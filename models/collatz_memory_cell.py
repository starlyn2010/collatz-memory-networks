"""CollatzMemoryCell: GRU con compuerta de actualización modulada por la
máscara Collatz determinista.

z_t = sigmoid(W_z [x_t; h_{t-1}]) * m_t
m_t = sigmoid(W_m · g_embedding(k_t))

donde k_t es la valuación Collatz (precomputada, nunca calculada aquí) y
g_embedding es el embedding de umbral:
  g(k) = [tanh(k/5), 1[k>=1], 1[k>=2], ..., 1[k>=kappa_max]]

Con m_t ≡ 1 la celda coincide con una GRU estándar (modo 2 de la ablación).
El canal Collatz se puede degradar de varias formas (modos 3-7) vía
gate_override; la máscara se pasa desde fuera, no se genera dentro.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def g_embedding(k, kappa_max=10):
    """Umbral embedding del valor de valuación k — vectorizado.

    k: escalar, (T,) o (B,) → salida con última dim 1+kappa_max."""
    k = torch.as_tensor(k, dtype=torch.float32)
    if k.dim() == 0:
        k = k.unsqueeze(0)
    base = torch.tanh(k / 5.0)
    ge = (k.unsqueeze(-1) >= torch.arange(1, kappa_max + 1, dtype=torch.float32)).float()
    return torch.cat([base.unsqueeze(-1), ge], dim=-1)


class CollatzMemoryCell(nn.Module):
    def __init__(self, input_size, hidden_size, kappa_max=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kappa_max = kappa_max
        self.W_x = nn.Linear(input_size, 3 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 3 * hidden_size)
        self.W_m = nn.Linear(kappa_max + 1, hidden_size)
        nn.init.zeros_(self.W_m.weight)
        nn.init.constant_(self.W_m.bias, 4.0)

    def compute_mask(self, valuations):
        """Máscara completa en una sola matmul vectorizada: (T, hidden).
        valuations: (T,) o (B, T)."""
        emb = g_embedding(valuations, self.kappa_max).to(self.W_m.weight.device)
        return torch.sigmoid(self.W_m(emb))

    def forward(self, x, h, m):
        """x: (B, input), h: (B, hidden), m: (B, hidden) máscara Collatz.
        Devuelve (h_new, z, m): z es la update gate hecho efectiva (post-m)."""
        gx = self.W_x(x)
        gh = self.W_h(h)
        r = torch.sigmoid(gx[:, : self.hidden_size] + gh[:, : self.hidden_size])
        z = torch.sigmoid(gx[:, self.hidden_size : 2 * self.hidden_size] + gh[:, self.hidden_size : 2 * self.hidden_size])
        n = torch.tanh(gx[:, 2 * self.hidden_size :] + r * gh[:, 2 * self.hidden_size :])
        z_eff = z * m
        h_new = (1.0 - z_eff) * h + z_eff * n
        return h_new, z_eff, m


class CollatzMemoryStack(nn.Module):
    """N capas de CollatzMemoryCell apiladas.

    valuations: tensor (T,) o (B, T) de valuaciones precomputadas, o None
    si se provee gate_override por el modo de ablación.
    gate_override: (B, T, hidden) para modos alterados (RandomGate, Sobol,
    etc.) o None para el modo índice puro (se deriva de valuations).
    """

    def __init__(self, input_size, hidden_size, n_layers=2, kappa_max=10):
        super().__init__()
        self.cells = nn.ModuleList(
            [CollatzMemoryCell(input_size if i == 0 else hidden_size, hidden_size, kappa_max) for i in range(n_layers)]
        )

    def forward(self, x_seq, valuations=None, gate_override=None, track=None):
        """x_seq: (T, B, input). valuations: (T,) o (B, T) o None.
        gate_override: (T, B, hidden) o None. track: lista opcional que
        recoge por posición (capa 0): (z_eff medio, m medio, norma de h)."""
        T, B, _ = x_seq.shape
        device = x_seq.device
        h = [torch.zeros(B, cell.hidden_size, device=device) for cell in self.cells]
        layer_masks = None
        if gate_override is not None:
            pass
        elif valuations is not None:
            layer_masks = [cell.compute_mask(valuations.to(device)).to(device) for cell in self.cells]
        out = [None] * T
        for t in range(T):
            inp = x_seq[t]
            for i, cell in enumerate(self.cells):
                if gate_override is not None:
                    m = gate_override[t] if gate_override.dim() == 3 else gate_override[:, t]
                elif layer_masks is not None:
                    mask_i = layer_masks[i]
                    m = mask_i[t] if mask_i.dim() == 2 else mask_i[:, t]
                else:
                    m = torch.ones(B, cell.hidden_size, device=device)
                h[i], z_eff, m_t = cell(inp, h[i], m)
                if track is not None and i == 0:
                    track.append(
                        (
                            z_eff.detach().mean().item(),
                            m_t.detach().mean().item(),
                            h[-1].detach().norm(dim=-1).mean().item(),
                        )
                    )
                inp = h[i]
            out[t] = h[-1]
        return torch.stack(out)