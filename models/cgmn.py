"""CGMN: arquitectura completa Collatz Memory Network.

Embedding + POSITIONAL + N capas CollatzMemoryCell + proyección de salida.
La secuencia Collatz (valuaciones) se precomputa en el Paso 1: o bien se
pasa como tensor en forward (datasets que ya la tienen), o bien se cargó
antes como buffer con set_valuations — NUNCA se recalcula aquí.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .collatz_memory_cell import CollatzMemoryStack  # noqa: E402


class CGMN(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=64,
        n_layers=2,
        pad_token_id=0,
        kappa_max=10,
        max_seq_len=512,
        regress=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.regress = regress
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.pos = nn.Parameter(torch.zeros(max_seq_len, hidden_size))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)
        self.stack = CollatzMemoryStack(hidden_size, hidden_size, n_layers, kappa_max)
        self.out = nn.Linear(hidden_size, 1 if regress else vocab_size, bias=False)
        self.out.bias = nn.Parameter(torch.zeros(1 if regress else vocab_size))
        self._valuations = None
        self.max_seq_len = max_seq_len

    def set_valuations(self, vals):
        """Buffer de valuaciones precomputadas (T,) o (B, T)."""
        self._valuations = vals
        self.register_buffer("valuations_buffer", torch.as_tensor(vals), persistent=False)

    def forward(self, x, valuations=None, gate_override=None, track=None):
        """x: (B, T) índices de token.

        valuations: (T,) | (B, T) — si None usa el buffer o m_id = 1.
        gate_override: (T, B, hidden) | None. track: ver CollatzMemoryStack.
        """
        B, T = x.shape
        emb = self.embed(x) + self.pos[:T].unsqueeze(0)
        emb = emb.transpose(0, 1)
        if valuations is None:
            if self._valuations is not None:
                v = torch.as_tensor(self._valuations, device=x.device)
                if v.dim() == 1:
                    v = v[:T].unsqueeze(0).expand(B, -1)
                else:
                    v = v[:, :T]
                valuations = v
            else:
                valuations = None
        h = self.stack(emb, valuations=valuations, gate_override=gate_override, track=track)
        logits = self.out(h)
        return logits.transpose(0, 1)