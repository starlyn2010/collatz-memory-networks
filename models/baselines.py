"""Baselines con parámetros comparables a CGMN (~85k con embed+pos).

Todos comparten: nn.Embedding(vocab, h, pad) + posicional aprendido (max_len, h)
+ proyección de salida. Así la comparación de parámetros es limpia.

- RNN vanilla (tanh), 2 capas h=112.
- GRU estándar, 2 capas h=64 (equivale a CMC con m_t=1).
- Transformer mini: 2 capas, 2 cabezas, d_model=56, ffn=112.
- CfC pura: ecuación cerrada de Hasani et al. (2021), 2 capas h=56, sin ACT.

CfC cerrado (por paso, x̃=[x;h]):
  K  = σ(W_k x̃ + b_k)
  Wτ = tanh(W_τ x̃ + b_τ); Wc = tanh(W_c x̃ + b_c); Wn = tanh(W_n x̃ + b_n)
  f  = σ(W_f x̃ + b_f)
  h' = f ⊙ (K ⊙ Wc ⊙ h + (1-K) ⊙ Wn)   + clamp(-10, 10) (estilo LNN Platformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _EmbedPosHead(nn.Module):
    """Embedding + posicional + salida compartidos por todos los baselines."""

    def __init__(self, vocab_size, hidden, max_len, pad_token_id=0, regress=False):
        super().__init__()
        self.hidden = hidden
        self.regress = regress
        out_dim = 1 if regress else vocab_size
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=pad_token_id)
        self.pos = nn.Parameter(torch.zeros(max_len, hidden))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)
        self.out = nn.Linear(hidden, out_dim, bias=False)
        self.out.bias = nn.Parameter(torch.zeros(out_dim))

    def forward_in(self, x):
        B, T = x.shape
        return self.embed(x) + self.pos[:T].unsqueeze(0)


class RNNBaseline(_EmbedPosHead):
    def __init__(self, vocab_size, hidden=112, n_layers=2, max_len=512, pad_token_id=0, regress=False):
        super().__init__(vocab_size, hidden, max_len, pad_token_id, regress=regress)
        self.rnn = nn.RNN(hidden, hidden, n_layers, batch_first=True, nonlinearity="tanh")

    def forward(self, x):
        h = self.rnn(self.forward_in(x))[0]
        return self.out(h)


class GRUBaseline(_EmbedPosHead):
    def __init__(self, vocab_size, hidden=64, n_layers=2, max_len=512, pad_token_id=0, regress=False):
        super().__init__(vocab_size, hidden, max_len, pad_token_id, regress=regress)
        self.gru = nn.GRU(hidden, hidden, n_layers, batch_first=True)

    def forward(self, x):
        h = self.gru(self.forward_in(x))[0]
        return self.out(h)


class TransformerBaseline(_EmbedPosHead):
    def __init__(self, vocab_size, d_model=56, n_layers=2, n_heads=2, ffn=112,
                 max_len=512, pad_token_id=0, regress=False, causal=True):
        super().__init__(vocab_size, d_model, max_len, pad_token_id, regress=regress)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            batch_first=True, dropout=0.0, activation="gelu", norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.causal = causal

    def forward(self, x):
        mask = (x == self.embed.padding_idx)
        attn_mask = None
        if self.causal:
            T = x.shape[1]
            attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.enc(self.forward_in(x), mask=attn_mask, src_key_padding_mask=mask)
        return self.out(h)


class CfCCell(nn.Module):
    """Ecuación cerrada de Hasani et al. (2021), sin ACT.

    Los 5 mapas lineales (W_k, W_tau, W_c, W_n, W_f) están fusionados en un
    único Linear con chunk (5·hidden), idéntico matemáticamente, ~4x más
    rápido en CPU."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.fused = nn.Linear(input_size + hidden_size, 5 * hidden_size)
        with torch.no_grad():
            self.fused.bias[hidden_size : 2 * hidden_size] = 4.0

    def forward(self, x, h):
        xh = torch.cat([x, h], dim=-1)
        out = self.fused(xh)
        Wk, Wf, Wtau, Wc, Wn = out.chunk(5, dim=-1)
        K = torch.sigmoid(Wk)
        f = torch.sigmoid(Wf)
        Wc = torch.tanh(Wc)
        Wn = torch.tanh(Wn)
        h_new = f * (K * Wc * h + (1.0 - K) * Wn)
        return torch.clamp(h_new, -10.0, 10.0)


class CfCBaseline(_EmbedPosHead):
    def __init__(self, vocab_size, hidden=56, n_layers=2, max_len=512, pad_token_id=0, regress=False):
        super().__init__(vocab_size, hidden, max_len, pad_token_id, regress=regress)
        self.cells = nn.ModuleList(
            [CfCCell(hidden, hidden) for _ in range(n_layers)]
        )

    def forward(self, x):
        inp = self.forward_in(x).transpose(0, 1)
        T, B, _ = inp.shape
        hs = [torch.zeros(B, self.hidden, device=x.device) for _ in self.cells]
        outs = []
        for t in range(T):
            for i, cell in enumerate(self.cells):
                hs[i] = cell(inp[t] if i == 0 else hs[i - 1], hs[i])
            outs.append(hs[-1])
        return self.out(torch.stack(outs, dim=1))


def build_baseline(name, vocab_size, max_len=512, regress=False):
    name = name.lower()
    if name == "rnn":
        return RNNBaseline(vocab_size, hidden=96, n_layers=2, max_len=max_len, regress=regress)
    if name == "gru":
        return GRUBaseline(vocab_size, hidden=64, n_layers=2, max_len=max_len, regress=regress)
    if name == "transformer":
        return TransformerBaseline(vocab_size, d_model=56, n_layers=2, n_heads=2,
                                   ffn=112, max_len=max_len, regress=regress)
    if name == "cfc":
        return CfCBaseline(vocab_size, hidden=53, n_layers=2, max_len=max_len, regress=regress)
    raise ValueError(f"baseline desconocido: {name}")