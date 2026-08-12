# %% [markdown]
# # CGMN vs Transformer — TinyStories: pelea justa + bóveda Collatz
#
# **Qué hace este notebook** (3 partes, todo automático):
# 1. **Pelea justa**: CGMN y Transformer con **el mismo número de parámetros**
#    (±1%) y **8 epochs** — para que CGMN tenga tiempo de estabilizarse.
# 2. **Bóveda Collatz (hipocampo)**: CGMN aprende sola (calentamiento de 200
#    pasos en memoria a+b) y luego sirve al Transformer como **memoria de
#    apuntes fijos (16)** entre bloques de 128. El Transformer lee el texto; la
#    puerta Collatz decide qué se guarda en la bóveda.
# 3. **Demo de coste a contexto 512 y 1024**: curvas de FLOPs/token que muestran
#    la "anti-burbuja" (la bóveda no crece con el contexto; el vanilla sí).
#
# **Instrucciones:** *Entorno de ejecución → Ejecutar todo* y te puedes ir.
# Descarga del dataset, entrenamiento y descarga de resultados automáticos.
# Configuración editable en `CONFIG` (celda 2).

# %%
# ========== Celda 1: keep-alive (evita desconexión por inactividad) ==========
try:
    import google.colab  # noqa: F401
    from IPython.display import Javascript, display

    display(Javascript(
        "function ClickConnect(){"
        "  const b=[...document.querySelectorAll('colab-connect-button')];"
        "  b.forEach(x=>x.click());"
        "  setTimeout(ClickConnect,60000);"
        "}"
        "ClickConnect();"
    ))
    print("keep-alive activado (click automático cada 60 s)")
except Exception:
    print("No es Colab: keep-alive omitido")

# %%
# ========== Celda 2: config + imports ==========
import json
import inspect
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=False)

CONFIG = {
    "SEQ": 128, "BS": 64, "VOCAB_SIZE": 10000, "MIN_FREQ": 2,
    "N_STORIES": 5000, "MAX_STORY": 250, "VAL_FRAC": 0.08,
    "EPOCHS": 8, "LR": 3e-4, "LR_MIN": 3e-5, "WD": 0.01, "CLIP": 1.0,
    "D_CGMN": 300, "N_LAYERS": 2, "HEADS": 4, "FFN": 1024,
    # bóveda
    "K_SEG": 3, "M_MEM": 16, "EPOCHS_VAULT": 4, "LR_VAULT": 3e-4,
    "D_VAULT": 256, "WARM_STEPS": 800, "WARM_LR": 2e-3, "WARM_LR_MIN": 3e-5,
    "KAPPA": 10, "K_COLLATZ": 50, "BASE_SEED": 42,
    "EVAL_EVERY": 45, "SEED": 0,
    "OUT_DIR": "/content",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(CONFIG["SEED"]); np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
try:
    _sig = inspect.signature(nn.TransformerEncoderLayer.__init__)
    ENC_EXTRA = {"enable_nested_tensor": False} if "enable_nested_tensor" in _sig.parameters else {}
except Exception:
    ENC_EXTRA = {}
print("device:", DEVICE)

# %%
# ========== Celda 3: generador Collatz + atención (port fiel del proyecto) ====
def syracuse_step(n, c=1):
    u = 3 * n + c
    v = (u & -u).bit_length() - 1
    return u >> v, v

def derive_seed(base_seed, t):
    M = (1 << 61) - 1
    x = (base_seed ^ (t * 0x9E3779B97F4A7C15)) & M
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & M
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & M
    x ^= x >> 31
    return (x | 1) + (1 << 62)

def collatz_valuations(T, c=1, base_seed=42, K=50):
    out = []
    for t in range(T):
        n = derive_seed(base_seed, t)
        v = 0
        for _ in range(K):
            n, v = syracuse_step(n, c)
        out.append(v)
    return out

def g_embedding(k, kappa_max):
    k = torch.as_tensor(k, dtype=torch.float32)
    base = torch.tanh(k / 5.0).unsqueeze(-1)
    ge = (k.unsqueeze(-1) >= torch.arange(1, kappa_max + 1, dtype=torch.float32, device=k.device)).float()
    return torch.cat([base, ge], dim=-1)

def collatz_mask(W_m, T, kappa_max):
    k = torch.as_tensor(collatz_valuations(T), dtype=torch.long)
    return torch.sigmoid(W_m(g_embedding(k, kappa_max).to(W_m.weight.device)))

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# %%
# ========== Celda 4: descarga automática del dataset ==========
def download_stories(n_stories, max_len, out_dir):
    cache = os.path.join(out_dir, "stories_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print("usando caché local:", len(z["train"]), "train,", len(z["val"]), "val")
        return z["train"].tolist(), z["val"].tolist()
    from datasets import load_dataset
    last = None
    for attempt in range(5):
        try:
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            it = iter(ds)
            texts = []
            for i, ex in enumerate(it):
                if i >= n_stories:
                    break
                words = ex["text"].split()
                if len(words) < 20:
                    continue
                texts.append(words[:max_len])
            rng = random.Random(CONFIG["SEED"])
            rng.shuffle(texts)
            n_val = max(1, int(len(texts) * CONFIG["VAL_FRAC"]))
            train, val = texts[n_val:], texts[:n_val]
            np.savez(cache, train=np.array(train, dtype=object), val=np.array(val, dtype=object))
            print(f"descargado: {len(train)} train / {len(val)} val historias")
            return train, val
        except Exception as e:
            last = e
            print(f"intento {attempt+1}/5 falló: {e}")
            time.sleep(10)
    raise RuntimeError("No se pudo descargar TinyStories: " + repr(last))

def build_tokenizer(stories, vocab_size, min_freq):
    counter = Counter(w for s in stories for w in s)
    words = [w for w, c in counter.most_common(vocab_size - 2) if c >= min_freq]
    itos = ["<pad>", "<unk>"] + words
    stoi = {w: i for i, w in enumerate(itos)}
    print(f"vocabulario: {len(itos)} palabras")
    return itos, stoi

def encode_pack(stories, stoi, seq_len, min_chunk=8):
    unk = stoi["<unk>"]
    xs, ys, ms = [], [], []
    for s in stories:
        ids = [stoi.get(w, unk) for w in s]
        for i in range(0, len(ids), seq_len):
            chunk = ids[i:i + seq_len]
            if len(chunk) < min_chunk:
                continue
            pad = [0] * (seq_len - len(chunk))
            xs.append(chunk + pad); ys.append(chunk[1:] + [0] + pad)
            ms.append([1.0] * len(chunk) + [0.0] * (seq_len - len(chunk)))
    return np.asarray(xs, dtype="int64"), np.asarray(ys, dtype="int64"), np.asarray(ms, dtype="float32")

def make_batches(x, y, m, bs, seed):
    idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield x[j], y[j], m[j]

def encode_books(stories, stoi, S, K):
    """Cada historia -> exactamente K bloques de S tokens (truncar/rellenar).
    Devuelve (B, K, S) int + máscara (B, K, S) real-token."""
    unk = stoi["<unk>"]
    B = len(stories)
    xs = np.zeros((B, K, S), dtype="int64")
    ms = np.zeros((B, K, S), dtype="float32")
    for bi, s in enumerate(stories):
        ids = [stoi.get(w, unk) for w in s][:K * S]
        for k in range(K):
            chunk = ids[k * S:(k + 1) * S]
            if not chunk:
                continue
            xs[bi, k, :len(chunk)] = chunk
            ms[bi, k, :len(chunk)] = 1.0
    return xs, ms

def make_book_batches(xs, ms, bs, seed):
    B = len(xs)
    idx = np.random.RandomState(seed).permutation(B)
    for i in range(0, B, bs):
        j = idx[i:i + bs]
        yield xs[j], ms[j]

# %%
# ========== Celda 5: modelos ==========
class CollatzCell(nn.Module):
    """GRU con update gate modulada por la valuación Collatz (port de CGMN)."""

    def __init__(self, hidden, kappa_max=10):
        super().__init__()
        self.hidden = hidden
        self.kappa_max = kappa_max
        self.W_x = nn.Linear(hidden, 3 * hidden)
        self.W_h = nn.Linear(hidden, 3 * hidden)
        self.W_m = nn.Linear(kappa_max + 1, hidden)
        nn.init.zeros_(self.W_m.weight)
        nn.init.constant_(self.W_m.bias, 4.0)

    def forward(self, x, h, m):
        gx = self.W_x(x)
        gh = self.W_h(h)
        r = torch.sigmoid(gx[:, :self.hidden] + gh[:, :self.hidden])
        z = torch.sigmoid(gx[:, self.hidden:2 * self.hidden] + gh[:, self.hidden:2 * self.hidden])
        n = torch.tanh(gx[:, 2 * self.hidden:] + r * gh[:, 2 * self.hidden:])
        z_eff = z * m
        return (1.0 - z_eff) * h + z_eff * n

class CGMN(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, kappa_max, seq_len, valuations):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(seq_len, hidden))
        nn.init.normal_(self.pos, 0.0, 0.02)
        self.cells = nn.ModuleList([CollatzCell(hidden, kappa_max) for _ in range(n_layers)])
        self.out = nn.Linear(hidden, vocab_size, bias=False)
        self.out.weight = self.embed.weight
        self.register_buffer("valuations", torch.as_tensor(valuations, dtype=torch.long))

    def forward(self, x):
        B, T = x.shape
        h = [torch.zeros(B, c.hidden, device=x.device) for c in self.cells]
        # máscaras Collatz en CADA forward (W_m entrenable, como en el proyecto)
        emb = g_embedding(self.valuations[:T].to(x.device), self.cells[0].kappa_max)
        masks = torch.stack([torch.sigmoid(c.W_m(emb)) for c in self.cells])
        out = []
        e = self.embed(x) + self.pos[:T].unsqueeze(0)
        for t in range(T):
            inp = e[:, t]
            for i, c in enumerate(self.cells):
                h[i] = c(inp, h[i], masks[i, t])
                inp = h[i]
            out.append(inp)
        return self.out(torch.stack(out, dim=1))

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, ffn, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(seq_len, d_model))
        nn.init.normal_(self.pos, 0.0, 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ffn,
            batch_first=True, dropout=0.0, activation="gelu", norm_first=True,
            **ENC_EXTRA)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.embed.weight

    def forward(self, x):
        T = x.shape[1]
        # máscara FLOAT aditiva (0.0/-1e9): las bool + padding dan NaN en eval
        attn = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = torch.zeros(T, T, device=x.device).masked_fill(attn, -1e9)
        h = self.enc(self.embed(x) + self.pos[:T].unsqueeze(0), mask=attn,
                     src_key_padding_mask=(x == 0).float())
        return self.out(h)

def create_model(name, cfg, d_model, seq_len):
    if name == "cgmn":
        vals = collatz_valuations(seq_len)
        m = CGMN(cfg["VOCAB_SIZE"], d_model, cfg["N_LAYERS"], cfg["KAPPA"], seq_len, vals)
    else:
        m = MiniTransformer(cfg["VOCAB_SIZE"], d_model, cfg["N_LAYERS"],
                            cfg["HEADS"], cfg["FFN"], seq_len)
    with torch.no_grad():
        nn.init.normal_(m.embed.weight, 0.0, 0.02)
        m.embed.weight[0].zero_()
    return m.to(DEVICE)

def tune_transformer_d(cfg, n_cgmn, lo=128, hi=512):
    """Dict para el transformer con d auto-ajustado para params = CGMN ±1%."""
    best = None
    for d in range(lo, hi + 1, 8):
        m = MiniTransformer(cfg["VOCAB_SIZE"], d, cfg["N_LAYERS"], cfg["HEADS"], cfg["FFN"], cfg["SEQ"])
        n = count_params(m)
        err = abs(n - n_cgmn) / n_cgmn
        if err < 0.01:
            return d, n
        if best is None or err < best[0]:
            best = (err, d, n)
    return best[1], best[2]

# %%
# ========== Celda 6: entrenamiento (parte 1 — pelea justa) ==========
def train_model(name, model, x, y, m, xv, yv, mv, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WD"])
    steps_per_epoch = max(1, len(x) // cfg["BS"])
    total = cfg["EPOCHS"] * steps_per_epoch
    step = 0
    hist = {"train_ce": [], "val_ppl": [], "val_ce": [], "step": []}
    t0 = time.time()
    model.train()
    for ep in range(cfg["EPOCHS"]):
        for xb, yb, mb in make_batches(torch.from_numpy(x), torch.from_numpy(y),
                                       torch.from_numpy(m), cfg["BS"], cfg["SEED"] + ep):
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            ce = -(torch.log_softmax(logits, -1).gather(-1, yb.unsqueeze(-1)).squeeze(-1) * mb).sum() / mb.sum().clamp(min=1)
            ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["CLIP"])
            opt.step()
            for g in opt.param_groups:
                g["lr"] = cfg["LR_MIN"] + 0.5 * (cfg["LR"] - cfg["LR_MIN"]) * (1 + math.cos(math.pi * step / max(total - 1, 1)))
            step += 1
            if step % cfg["EVAL_EVERY"] == 0 or step == total:
                model.eval()
                vce = vn = 0.0
                with torch.no_grad():
                    for xb, yb, mb in make_batches(torch.from_numpy(xv), torch.from_numpy(yv),
                                                   torch.from_numpy(mv), cfg["BS"], 999):
                        xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
                        logits = model(xb)
                        vce += float((-(torch.log_softmax(logits, -1).gather(-1, yb.unsqueeze(-1)).squeeze(-1) * mb)).sum())
                        vn += float(mb.sum())
                vce /= max(vn, 1)
                hist["train_ce"].append(float(ce.detach().item()))
                hist["val_ce"].append(vce); hist["val_ppl"].append(float(math.exp(vce)))
                hist["step"].append(step)
                with open(os.path.join(cfg["OUT_DIR"], f"checkpoint_{name}.json"), "w") as f:
                    json.dump(dict(hist, name=name), f, indent=2)
                print(f"[{name}] step {step}/{total} train_ce={float(ce.detach().item()):.4f} "
                      f"val_ce={vce:.4f} ppl={math.exp(vce):.1f} ({time.time()-t0:.0f}s)", flush=True)
                model.train()
    print(f"[{name}] FIN {time.time()-t0:.0f}s — ppl final {hist['val_ppl'][-1]:.1f}", flush=True)
    return hist

# %%
# ========== Celda 7: calentamiento del hipocampo (CGMN aprende sola) ==========
def delayed_memory_batch(bs, D, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 9, size=(bs, 1))
    b = rng.integers(1, 9, size=(bs, 1))
    x = np.concatenate([a, np.zeros((bs, D), dtype=np.int64), b, np.zeros((bs, D), dtype=np.int64)], axis=1)
    y = (a + b).astype("float32")
    return torch.from_numpy(x), torch.from_numpy(y)

def warmup_hippocampus(cfg, d_hipp, kappa_max, out_dir):
    """Entrena SOLA la celda Collatz en memoria a+b (D=20). Devuelve la celda."""
    vals = collatz_valuations(2 * 20 + 2)
    cell = CollatzCell(d_hipp, kappa_max).to(DEVICE)
    embed = nn.Embedding(9, d_hipp, padding_idx=0).to(DEVICE)
    head = nn.Linear(d_hipp, 1).to(DEVICE)
    with torch.no_grad():
        nn.init.normal_(embed.weight, 0.0, 0.02)
    opt = torch.optim.AdamW(list(cell.parameters()) + list(embed.parameters()) + list(head.parameters()),
                            lr=cfg["WARM_LR"], weight_decay=cfg["WD"])
    lr_min = cfg["WARM_LR_MIN"]
    t0 = time.time()
    D = 20
    for step in range(cfg["WARM_STEPS"]):
        for g in opt.param_groups:
            g["lr"] = lr_min + 0.5 * (cfg["WARM_LR"] - lr_min) * (
                1 + math.cos(math.pi * step / max(cfg["WARM_STEPS"] - 1, 1)))
        xb, yb = delayed_memory_batch(cfg["BS"], D, 1000 + step)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        h = torch.zeros(xb.shape[0], d_hipp, device=DEVICE)
        e = embed(xb)  # (B, T, d)
        T = e.shape[1]
        # máscara recalculada por paso: W_m se entrena durante el calentamiento
        mask = collatz_mask(cell.W_m, T, kappa_max).to(DEVICE)
        for t in range(T):
            h = cell(e[:, t], h, mask[t])
        loss = (((h @ head.weight.T + head.bias).squeeze(-1) - yb[:, 0]) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in cell.parameters()] + [p for p in embed.parameters()] + [p for p in head.parameters()], cfg["CLIP"])
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"[calentamiento] step {step+1}/{cfg['WARM_STEPS']} mse={loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[calentamiento] FIN — celda criada en {time.time()-t0:.0f}s "
          f"(mse final {loss.item():.3f}, objetivo <1)", flush=True)
    return cell

# %%
# ========== Celda 8: BÓVEDA — modelo (corteza + hipocampo) ==========
class VaultModel(nn.Module):
    """Transformer (corteza) + CGMN (hipocampo) que guarda 16 apuntes fijos.

    - Cada bloque de S tokens entra al Transformer; su hidden propone apuntes.
    - El hipocampo (celda Collatz) los filtra con su puerta m_t y actualiza su
      estado h (persistente entre bloques).
    - Los últimes M estados del hipocampo son las 'tarjetas' que el siguiente
      bloque consulta al inicio de su atención.
    - El hipocampo NO lee el texto: recibe las propuestas (corteza->hipocampo).
    """

    def __init__(self, vocab_size, d, n_layers, heads, ffn, S, M, kappa_max,
                 warm_cell=None):
        super().__init__()
        self.d, self.S, self.M = d, S, M
        self.kappa_max = kappa_max
        self.embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(S + M, d))
        nn.init.normal_(self.pos, 0.0, 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ffn,
            batch_first=True, dropout=0.0, activation="gelu", norm_first=True,
            **ENC_EXTRA)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, vocab_size, bias=False)
        self.out.weight = self.embed.weight
        # hipocampo
        self.hipp = CollatzCell(d, kappa_max) if warm_cell is None else warm_cell
        self.write = nn.Linear(d, d, bias=False)    # corteza -> propuesta
        self.read = nn.Linear(d, d, bias=False)     # estado -> tarjeta visible
        self.mem_init = nn.Parameter(torch.zeros(1, M, d))
        self.masks = None

    def _compute_masks(self, T_total, device):
        # recalculadas en CADA forward (W_m del hipocampo es entrenable)
        self.masks = collatz_mask(self.hipp.W_m, T_total, self.kappa_max).to(device)

    def forward(self, x_books):
        """x_books: (B, K, S). Procesa los K bloques con memoria persistente."""
        B, K, S = x_books.shape
        M = self.M
        mem = self.mem_init.expand(B, -1, -1)                 # (B, M, d)
        h = torch.zeros(B, self.d, device=x_books.device)
        logits, masks = [], []
        # máscara causal CORRECTA: token r atiende a columnas c <= r (tril(0))
        amask = torch.ones(S, S, dtype=torch.bool, device=x_books.device).tril(0)
        # memoria al inicio: todos los tokens pueden atender a las tarjetas
        attn_ok = torch.zeros(S + M, S + M, dtype=torch.bool, device=x_books.device)
        attn_ok[M:, M:] = amask       # S tokens bajo máscara causal
        attn_ok[M:, :M] = True        # tokens SI atienden a las tarjetas
        # las tarjetas se atienden ENTRE SÍ (salidas no usadas): ninguna fila
        # queda totalmente enmascarada -> evita NaN en softmax/softmax kernels
        attn_ok[:M, :M] = torch.eye(M, dtype=torch.bool, device=x_books.device)
        # máscara FLOAT aditiva: las bool + padding dan NaN en eval (torch)
        attn_f = torch.zeros(S + M, S + M, device=x_books.device).masked_fill(~attn_ok, -1e9)
        self._compute_masks(K * S + M, x_books.device)
        for k in range(K):
            xk = x_books[:, k]                                # (B, S)
            emb = self.embed(xk) + self.pos[M:M + S].unsqueeze(0)   # (B, S, d)
            mem = mem + self.pos[:M].unsqueeze(0)             # tarjetas: pos 0..M-1
            xseq = torch.cat([mem, emb], dim=1)               # (B, S+M, d)
            hidden = self.enc(xseq, mask=attn_f,
                              src_key_padding_mask=torch.cat(
                [torch.zeros(B, M, dtype=torch.float32, device=x_books.device), (xk == 0).float()], dim=1))
            # propuestas: corteza -> hipocampo (solo tokens reales, ceros para pad)
            props = self.write(hidden[:, M:])                 # (B, S, d)
            props = props * (xk != 0).unsqueeze(-1).float()
            outs = []
            base_mask_t = k * S
            for t in range(S):
                h = self.hipp(props[:, t], h, self.masks[base_mask_t + t])
                outs.append(h)
            hstack = torch.stack(outs, dim=1)                 # (B, S, d)
            mem = self.read(hstack[:, -M:])                    # últimes M fotos
            lg = torch.log_softmax(self.out(hidden[:, M:]), -1)
            logits.append(lg)
            masks.append((xk != 0))
        return torch.stack(logits, dim=1), torch.stack(masks, dim=1)  # (B,K,S,V),(B,K,S)

# %%
# ========== Celda 9: entrenamiento de la bóveda ==========
def train_vault(model, xs, ms, xvs, mvs, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR_VAULT"], weight_decay=cfg["WD"])
    B = len(xs)
    steps_per_epoch = max(1, B // cfg["BS"])
    total = cfg["EPOCHS_VAULT"] * steps_per_epoch
    step = 0
    hist = {"train_ce": [], "val_ppl": [], "step": []}
    t0 = time.time()
    model.train()
    for ep in range(cfg["EPOCHS_VAULT"]):
        for xb, mb in make_book_batches(xs, ms, cfg["BS"], cfg["SEED"] + ep):
            xb = torch.from_numpy(xb).to(DEVICE)
            mb = torch.from_numpy(mb).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits, lgmask = model(xb)
            ce = 0.0
            total_w = 0
            for k in range(cfg["K_SEG"]):
                lg = logits[:, k]                              # (B, S, V)
                yy = y if False else torch.cat([xb[:, k, 1:], torch.zeros((xb.shape[0], 1), dtype=torch.int64, device=DEVICE)], dim=-1)
                mk = (mb[:, k] > 0).float()
                mk = mk * torch.cat([mb[:, k, 1:], torch.zeros((xb.shape[0], 1), dtype=torch.float32, device=DEVICE)], dim=-1)
                ce = ce + (-lg.gather(-1, yy.unsqueeze(-1)).squeeze(-1) * mk).sum()
                total_w = total_w + mk.sum().clamp(min=1)
            loss = ce / total_w
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["CLIP"])
            opt.step()
            step += 1
            if step % cfg["EVAL_EVERY"] == 0 or step == total:
                model.eval()
                vce, vn = 0.0, 0
                with torch.no_grad():
                    vbatches = []
                    for xv, mv in make_book_batches(xvs, mvs, cfg["BS"], 999):
                        vbatches.append((torch.from_numpy(xv).to(DEVICE), torch.from_numpy(mv).to(DEVICE)))
                    for xv, mv in vbatches:
                        lg, lm = model(xv)
                        for k in range(cfg["K_SEG"]):
                            lgk = lg[:, k]
                            yy = torch.cat([xv[:, k, 1:], torch.zeros((xv.shape[0], 1), dtype=torch.int64, device=DEVICE)], dim=-1)
                            mkk = (mv[:, k] > 0).float() * torch.cat([mv[:, k, 1:], torch.zeros((xv.shape[0], 1), dtype=torch.float32, device=DEVICE)], dim=-1)
                            vce += float((-lgk.gather(-1, yy.unsqueeze(-1)).squeeze(-1) * mkk).sum())
                            vn += float(mkk.sum())
                vce /= max(vn, 1)
                hist["train_ce"].append(float(loss.item()))
                hist["val_ppl"].append(float(math.exp(vce)))
                hist["step"].append(step)
                with open(os.path.join(cfg["OUT_DIR"], "checkpoint_vault.json"), "w") as f:
                    json.dump(dict(hist, name="vault"), f, indent=2)
                print(f"[bóveda] step {step}/{total} train_ce={float(loss.item()):.4f} "
                      f"val_ce={vce:.4f} ppl={math.exp(vce):.1f} ({time.time()-t0:.0f}s)", flush=True)
                model.train()
    print(f"[bóveda] FIN {time.time()-t0:.0f}s — ppl final {hist['val_ppl'][-1]:.1f}", flush=True)
    return hist

# %%
# ========== Celda 10: demo de coste a contexto 512 y 1024 ==========
def flops_per_token(n_layers, d_model, L):
    """FLOPs de atención por token (aprox.): 4·L·d por capa por token.
    L = longitud atendida (contexto N para vanilla; S+M para bóveda)."""
    return 4 * n_layers * L * d_model

def build_books_ctx(val_toks, stoi, ctx_len, n_books=30, seed=5):
    """Crea n_books 'libros' de ctx_len tokens concatenando historias."""
    rng = random.Random(seed)
    unk = stoi["<unk>"]
    result = []
    while len(result) < n_books:
        book = []
        while len(book) < ctx_len:
            s = [stoi.get(w, unk) for w in rng.choice(val_toks)]
            book += s
        result.append(book[:ctx_len])
    return np.asarray(result, dtype="int64")

def demo_cost(cfg, stoi, val_toks, out_dir, vanilla, mem_model, mem_name, M):
    """ppl y FLOPs/token HONESTOS: vanilla entrenado (ventana fija de SEQ, sin
    memoria entre bloques) vs bóveda/archivador (memoria), sobre los MISMOS
    libros de contexto 512 y 1024. Ambos entrenados en bloques de SEQ."""
    out = {"flops": {"vanilla": {}, mem_name: {}}, "ppl": {"vanilla": {}, mem_name: {}}}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vanilla.eval()
    mem_model.eval()
    for ctx in [512, 1024]:
        books = build_books_ctx(val_toks, stoi, ctx)
        n_seg = ctx // cfg["SEQ"]
        xbooks = books.reshape(len(books), n_seg, cfg["SEQ"])
        # VANILLA: evalúa cada bloque AISLADO (no ve nada del pasado)
        vce = vn = 0
        with torch.no_grad():
            for i in range(0, len(xbooks), cfg["BS"]):
                xb = torch.from_numpy(xbooks[i:i + cfg["BS"]]).to(DEVICE)
                for k in range(n_seg):
                    xk = xb[:, k]
                    lg = torch.log_softmax(vanilla(xk), -1)
                    mb2 = (xk != 0).float()
                    yy = torch.cat([xk[:, 1:], torch.zeros((xk.shape[0], 1), dtype=torch.int64, device=DEVICE)], dim=-1)
                    mkk = mb2 * torch.cat([mb2[:, 1:], torch.zeros((xk.shape[0], 1), dtype=torch.float32, device=DEVICE)], dim=-1)
                    vce += float((-lg.gather(-1, yy.unsqueeze(-1)).squeeze(-1) * mkk).sum())
                    vn += float(mkk.sum())
        out["ppl"]["vanilla"][ctx] = float(math.exp(vce / max(vn, 1)))
        # coste HONESTO de arquitectura: para conectar el pasado sin memoria,
        # el vanilla de contexto completo paga atención sobre TODO el libro
        out["flops"]["vanilla"][ctx] = flops_per_token(cfg["N_LAYERS"], cfg["D_VAULT"], ctx)
        # MEMORIA: mismo libro, con memoria entre bloques
        vce = vn = 0
        with torch.no_grad():
            for i in range(0, len(xbooks), cfg["BS"]):
                xb = torch.from_numpy(xbooks[i:i + cfg["BS"]]).to(DEVICE)
                lg, lm = mem_model(xb)
                for k in range(n_seg):
                    lgk = lg[:, k]
                    mb2 = (xb[:, k] != 0).float()
                    yy = torch.cat([xb[:, k, 1:], torch.zeros((xb.shape[0], 1), dtype=torch.int64, device=DEVICE)], dim=-1)
                    mkk = mb2 * torch.cat([mb2[:, 1:], torch.zeros((xb.shape[0], 1), dtype=torch.float32, device=DEVICE)], dim=-1)
                    vce += float((-lgk.gather(-1, yy.unsqueeze(-1)).squeeze(-1) * mkk).sum())
                    vn += float(mkk.sum())
        out["ppl"][mem_name][ctx] = float(math.exp(vce / max(vn, 1)))
        out["flops"][mem_name][ctx] = flops_per_token(cfg["N_LAYERS"], cfg["D_VAULT"], cfg["SEQ"] + M)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ctxs = [512, 1024]
    ax1.plot(ctxs, [out["flops"]["vanilla"][c] / 1e6 for c in ctxs], "o-", color="#C44E52", label="vanilla (FLOPs/token)")
    ax1.plot(ctxs, [out["flops"][mem_name][c] / 1e6 for c in ctxs], "s-", color="#55A868", label=f"{mem_name} (FLOPs/token)")
    ax1.set_xlabel("contexto (tokens)"); ax1.set_ylabel("FLOPs de atención / token (M)")
    ax1.set_title("La bóveda no crece con el contexto (anti-burbuja)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_yscale("symlog")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_cost_contexto.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(out_dir, "demo_cost.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(">> demo de coste:", json.dumps(out))
    return out

# %%
# ========== Celda 11: ejecutar todo ==========
def run_all(cfg):
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)
    print(">> descargando TinyStories (automático)...")
    train_s, val_s = download_stories(cfg["N_STORIES"], cfg["MAX_STORY"], cfg["OUT_DIR"])
    itos, stoi = build_tokenizer(train_s + val_s, cfg["VOCAB_SIZE"], cfg["MIN_FREQ"])
    x, y, m = encode_pack(train_s, stoi, cfg["SEQ"])
    xv, yv, mv = encode_pack(val_s, stoi, cfg["SEQ"])
    print(f"parte 1 dataset: {len(x)} train, {len(xv)} val")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {"config": cfg}
    # ---- PARTE 1: pelea justa ----
    print("=== PARTE 1: PELEA JUSTA (params ±1%, 8 epochs) ===")
    cgmn = create_model("cgmn", cfg, cfg["D_CGMN"], cfg["SEQ"])
    n_cgmn = count_params(cgmn)
    d_tf, n_tf = tune_transformer_d(cfg, n_cgmn)
    print(f"   CGMN d={cfg['D_CGMN']} params={n_cgmn/1e6:.2f}M | Transformer d={d_tf} params={n_tf/1e6:.2f}M "
          f"({abs(n_tf-n_cgmn)/n_cgmn*100:.1f}% diff)")
    for name, model in [("cgmn", cgmn), ("transformer", (transformer_model := MiniTransformer(
            cfg["VOCAB_SIZE"], d_tf, cfg["N_LAYERS"], cfg["HEADS"], cfg["FFN"], cfg["SEQ"]).to(DEVICE)))]:
        with torch.no_grad():
            nn.init.normal_(model.embed.weight, 0.0, 0.02)
            model.embed.weight[0].zero_()
        t0 = time.time()
        hist = train_model(name, model, x, y, m, xv, yv, mv, cfg)
        results[name] = {"nparams": count_params(model), "sec": time.time() - t0, **hist}

    # ---- PARTE 2+3: calentamiento + bóveda ----
    print("=== PARTE 2/3: HIPOCAMPO (calienta sola) + BÓVEDA ===")
    cell = warmup_hippocampus(cfg, cfg["D_VAULT"], cfg["KAPPA"], cfg["OUT_DIR"])
    t0 = time.time()
    vault = VaultModel(cfg["VOCAB_SIZE"], cfg["D_VAULT"], cfg["N_LAYERS"], cfg["HEADS"],
                       cfg["FFN"], cfg["SEQ"], cfg["M_MEM"], cfg["KAPPA"], warm_cell=cell).to(DEVICE)
    with torch.no_grad():
        nn.init.normal_(vault.embed.weight, 0.0, 0.02)
        vault.embed.weight[0].zero_()
    xs, ms = encode_books(train_s, stoi, cfg["SEQ"], cfg["K_SEG"])
    xvs, mvs = encode_books(val_s, stoi, cfg["SEQ"], cfg["K_SEG"])
    print(f"bóveda dataset: {xs.shape[0]} libros × {cfg['K_SEG']} bloques")
    hist_v = train_vault(vault, xs, ms, xvs, mvs, cfg)
    torch.save(vault.state_dict(), os.path.join(cfg["OUT_DIR"], "vault_best.pt"))
    results["vault"] = {"nparams": count_params(vault), "sec": time.time() - t0, **hist_v}
    n_hipp = count_params(cell)
    results["hippocampus"] = {"nparams_warm": n_hipp}

    # ---- PARTE 4: demo de coste ----
    print("=== PARTE 4: DEMO DE COSTE 512 vs 1024 ===")
    results["demo_cost"] = demo_cost(cfg, stoi, val_s, cfg["OUT_DIR"], transformer_model, vault, "bóveda", cfg["M_MEM"])

    with open(os.path.join(cfg["OUT_DIR"], "results_tinystories.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ["cgmn", "transformer", "vault"]:
        if name in results:
            ax.plot(results[name]["step"], results[name]["val_ppl"], marker="o", label=name)
    ax.set_xlabel("step"); ax.set_ylabel("perplejidad de validación")
    ax.set_title("Pelea justa + bóveda — TinyStories (menor = mejor)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["OUT_DIR"], "tinystories_ppl.png"), dpi=150)
    print(">> guardado: results_tinystories.json, tinystories_ppl.png, fig_cost_contexto.png, vault_best.pt")
    print("RESULTADO FINAL:")
    for name in ["cgmn", "transformer", "vault"]:
        r = results.get(name)
        if r:
            print(f"  {name:12s} ppl_final={r['val_ppl'][-1]:.1f} params={r['nparams']/1e6:.2f}M sec={r['sec']:.0f}")
    return results

if "get_ipython" in globals():
    RESULTS = run_all(CONFIG)

# %%
# ========== Celda 12: descarga automática de resultados ==========
def auto_download(cfg):
    try:
        from google.colab import files
        for f in ["results_tinystories.json", "tinystories_ppl.png", "fig_cost_contexto.png", "demo_cost.json"]:
            p = os.path.join(cfg["OUT_DIR"], f)
            if os.path.exists(p):
                files.download(p)
        print(">> descarga automática iniciada")
    except Exception as e:
        print("No se pudo descargar automáticamente:", e)

if "get_ipython" in globals():
    auto_download(CONFIG)

# %%
# ========== Celda 13: FIN ==========
print("Notebook terminado. Revisa /content/ para los resultados.")

# %% [no-notebook]
# ========== Smoke test local (python3 colab_tinystories.py) — NO va al .ipynb ===
if __name__ == "__main__":
    def _smoke():
        print("SMOKE TEST (datos sintéticos, sin red)")
        outdir = "/tmp/opencode/tinystories_smoke"
        os.makedirs(outdir, exist_ok=True)
        cfg = {**CONFIG, "SEQ": 16, "BS": 8, "VOCAB_SIZE": 60, "MIN_FREQ": 1,
               "N_STORIES": 40, "MAX_STORY": 60, "EPOCHS": 1, "EVAL_EVERY": 2,
               "D_CGMN": 32, "FFN": 64, "HEADS": 2, "D_VAULT": 32, "WARM_STEPS": 10,
               "EPOCHS_VAULT": 1, "M_MEM": 4, "K_SEG": 3, "OUT_DIR": outdir}
        words = [f"w{i}" for i in range(40)]
        rng = random.Random(1)
        toks = [rng.choices(words, k=24) for _ in range(cfg["N_STORIES"])]
        itos, stoi = build_tokenizer(toks, cfg["VOCAB_SIZE"], cfg["MIN_FREQ"])
        x, y, m = encode_pack(toks, stoi, cfg["SEQ"])
        assert x.shape[1] == cfg["SEQ"] and (x == 0).sum() > 0 and m.sum() > 0
        # parte 1
        cgmn = create_model("cgmn", cfg, 32, cfg["SEQ"])
        assert cgmn.embed.weight is cgmn.out.weight
        d_tf, n_tf = tune_transformer_d(cfg, count_params(cgmn), lo=16, hi=96)
        h = train_model("cgmn", cgmn, x, y, m, x[:4], y[:4], m[:4], cfg)
        assert h["val_ppl"]
        # bóveda
        cell = warmup_hippocampus(cfg, 32, cfg["KAPPA"], outdir)
        v2 = VaultModel(60, 32, 2, 2, 64, 16, 4, cfg["KAPPA"], warm_cell=cell)
        xs, ms = encode_books(toks, stoi, 16, 3)
        logits, lm = v2(torch.from_numpy(xs))
        assert logits.shape == (xs.shape[0], 3, 16, 60), logits.shape
        assert torch.isfinite(logits).all() and torch.isfinite(lm).all(), "NaN/inf en bóveda (segmento todo-pad)"
        h2 = train_vault(v2, xs, ms, xs[:8], ms[:8], cfg)
        assert h2["val_ppl"]
        # REGRESIÓN ANTI-FUGA: con palabras aleatorias sin estructura, la ce de
        # validación NO puede colapsar (~0) salvo que el modelo vea el futuro.
        assert h2["val_ppl"][-1] > math.exp(1.0), f"¿fuga temporal? val_ppl={h2['val_ppl'][-1]:.3f}"
        tf = MiniTransformer(60, 32, 2, 2, 64, 16)
        with torch.no_grad():
            tf(torch.zeros(4, 16, dtype=torch.int64))
        demo = demo_cost(cfg, stoi, toks, outdir, tf, v2, "bóveda", 4)
        assert 512 in demo["flops"]["vanilla"] and "fig_cost_contexto.png" in os.listdir(outdir)
        print("SMOKE OK")
    _smoke()
