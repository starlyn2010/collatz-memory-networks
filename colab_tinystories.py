# %% [markdown]
# # CGMN vs Transformer — TinyStories (prueba de lenguaje)
#
# **Qué hace este notebook:** descarga solo el dataset TinyStories, construye un
# vocabulario de ~10k palabras, entrena **CGMN** (red con memoria Collatz) y un
# **Transformer mini** del mismo tamaño, y compara la perplejidad (ppl) de
# validación. No requiere ninguna interacción: al final se descargan solos los
# resultados (results.json + gráfico).
#
# **Instrucciones:** menú *Entorno de ejecución → Ejecutar todo* y puedes irte.
# La descarga del dataset, el entrenamiento y la descarga de resultados son
# automáticos. Si se corta la sesión, el último checkpoint queda en
# `/content/checkpoint_cgmn.json` y `/content/checkpoint_transformer.json`.
#
# Configuración editable abajo (celda 2, `CONFIG`).

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
    "SEQ": 128,            # tokens por secuencia de entrenamiento
    "BS": 64,              # batch size
    "VOCAB_SIZE": 10000,   # tope de palabras del vocabulario
    "MIN_FREQ": 2,         # palabras con frecuencia < MIN_FREQ -> <unk>
    "N_STORIES": 5000,     # historias usadas de TinyStories (streaming)
    "MAX_STORY": 250,      # tope de tokens por historia
    "VAL_FRAC": 0.08,      # fracción de historias de validación
    "EPOCHS": 2,           # pasadas sobre el corpus
    "LR": 3e-4, "LR_MIN": 3e-5, "WD": 0.01, "CLIP": 1.0,
    "D_MODEL": 256,        # ancho de CGMN y del Transformer (paridad)
    "N_LAYERS": 2, "HEADS": 4, "FFN": 1024,
    "KAPPA": 10, "K_COLLATZ": 50, "BASE_SEED": 42,
    "EVAL_EVERY": 90,      # pasos entre evaluaciones de ppl
    "SEED": 0,
    "OUT_DIR": "/content",  # dónde guardar checkpoints y resultados
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(CONFIG["SEED"]); np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
print("device:", DEVICE)

# %%
# ========== Celda 3: generador Collatz (port fiel del proyecto) ==========
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
    """Valuaciones por índice: P(v2(3n+c)=k) = 2^-k. Deterministas."""
    out = []
    for t in range(T):
        n = derive_seed(base_seed, t)
        v = 0
        for _ in range(K):
            n, v = syracuse_step(n, c)
        out.append(v)
    return out

# %%
# ========== Celda 4: descarga automática del dataset ==========
def download_stories(n_stories, max_len, out_dir):
    """TinyStories en streaming: solo descarga los shards necesarios.

    Con reintentos (5) para no morir por cortes de red. Cachea el resultado
    tokenizado para que re-ejecutar no vuelva a descargar."""
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
                t = ex["text"]
                words = t.split()
                if len(words) < 20:
                    continue
                texts.append(words[:max_len])
            rng = random.Random(CONFIG["SEED"])
            rng.shuffle(texts)
            n_val = max(1, int(len(texts) * CONFIG["VAL_FRAC"]))
            train = texts[n_val:]
            val = texts[:n_val]
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
    print(f"vocabulario: {len(itos)} palabras (frecuencia min {min_freq})")
    return itos, stoi

def encode_pack(stories, stoi, seq_len, min_chunk=8):
    """Tokeniza y parte en bloques de seq_len (bloques con >= min_chunk tokens;
    los más cortos se descartan para no entrenar sobre ruido)."""
    unk = stoi["<unk>"]
    xs, ys, ms = [], [], []
    for s in stories:
        ids = [stoi.get(w, unk) for w in s]
        for i in range(0, len(ids), seq_len):
            chunk = ids[i:i + seq_len]
            if len(chunk) < min_chunk:
                continue
            pad = [0] * (seq_len - len(chunk))
            x = chunk + pad
            y = chunk[1:] + [0] + pad
            m = [1.0] * len(chunk) + [0.0] * (seq_len - len(chunk))
            xs.append(x); ys.append(y); ms.append(m)
    return np.asarray(xs, dtype="int64"), np.asarray(ys, dtype="int64"), np.asarray(ms, dtype="float32")

def make_batches(x, y, m, bs, seed):
    idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield x[j], y[j], m[j]

# %%
# ========== Celda 5: modelos (mismos embedding/posición/head) ==========
def g_embedding(k, kappa_max):
    k = torch.as_tensor(k, dtype=torch.float32)
    base = torch.tanh(k / 5.0).unsqueeze(-1)
    ge = (k.unsqueeze(-1) >= torch.arange(1, kappa_max + 1, dtype=torch.float32)).float()
    return torch.cat([base, ge], dim=-1)

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
        self.out.weight = self.embed.weight  # head atado (paridad con Transformer)
        with torch.no_grad():
            emb = g_embedding(torch.as_tensor(valuations, dtype=torch.long), kappa_max)
            masks = torch.stack([torch.sigmoid(c.W_m(emb)) for c in self.cells])
        self.register_buffer("mask", masks)  # (n_layers, SEQ, hidden)

    def forward(self, x):
        B, T = x.shape
        h = [torch.zeros(B, c.hidden, device=x.device) for c in self.cells]
        out = []
        e = self.embed(x) + self.pos[:T].unsqueeze(0)
        for t in range(T):
            inp = e[:, t]
            for i, c in enumerate(self.cells):
                h[i] = c(inp, h[i], self.mask[i, t])
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
            batch_first=True, dropout=0.0, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.embed.weight

    def forward(self, x):
        T = x.shape[1]
        attn = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.enc(self.embed(x) + self.pos[:T].unsqueeze(0), mask=attn,
                     src_key_padding_mask=(x == 0))
        return self.out(h)

def create_model(name, cfg):
    if name == "cgmn":
        vals = collatz_valuations(cfg["SEQ"], base_seed=cfg["BASE_SEED"], K=cfg["K_COLLATZ"])
        m = CGMN(cfg["VOCAB_SIZE"], cfg["D_MODEL"], cfg["N_LAYERS"], cfg["KAPPA"],
                 cfg["SEQ"], vals)
    else:
        m = MiniTransformer(cfg["VOCAB_SIZE"], cfg["D_MODEL"], cfg["N_LAYERS"],
                            cfg["HEADS"], cfg["FFN"], cfg["SEQ"])
    # Embedding pequeño (N(0, 0.02)): con head atado, el init por defecto N(0,1)
    # del embedding explota los logits (std ~sqrt(d)·|h|·1). El pad queda a 0.
    with torch.no_grad():
        nn.init.normal_(m.embed.weight, 0.0, 0.02)
        m.embed.weight[0].zero_()
    return m.to(DEVICE)

# %%
# ========== Celda 6: entrenamiento ==========
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
                vce = 0.0
                vn = 0
                with torch.no_grad():
                    for xb, yb, mb in make_batches(torch.from_numpy(xv), torch.from_numpy(yv),
                                                   torch.from_numpy(mv), cfg["BS"], 999):
                        xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
                        logits = model(xb)
                        vce += float((-(torch.log_softmax(logits, -1).gather(-1, yb.unsqueeze(-1)).squeeze(-1) * mb)).sum())
                        vn += float(mb.sum())
                vce /= max(vn, 1)
                hist["train_ce"].append(float(ce.detach().item()))
                hist["val_ce"].append(vce)
                hist["val_ppl"].append(float(math.exp(vce)))
                hist["step"].append(step)
                ck = {k: hist[k] for k in hist}
                ck["name"] = name
                with open(os.path.join(cfg["OUT_DIR"], f"checkpoint_{name}.json"), "w") as f:
                    json.dump(ck, f, indent=2)
                print(f"[{name}] step {step}/{total} train_ce={float(ce.detach().item()):.4f} "
                      f"val_ce={vce:.4f} ppl={math.exp(vce):.1f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                model.train()
    print(f"[{name}] FIN {time.time()-t0:.0f}s — ppl final {hist['val_ppl'][-1]:.1f}", flush=True)
    return hist

# %%
# ========== Celda 7: ejecutar todo ==========
def run_all(cfg):
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)
    print(">> descargando TinyStories (automático)...")
    train_s, val_s = download_stories(cfg["N_STORIES"], cfg["MAX_STORY"], cfg["OUT_DIR"])
    itos, stoi = build_tokenizer(train_s + val_s, cfg["VOCAB_SIZE"], cfg["MIN_FREQ"])
    x, y, m = encode_pack(train_s, stoi, cfg["SEQ"])
    xv, yv, mv = encode_pack(val_s, stoi, cfg["SEQ"])
    print(f"dataset listo: {len(x)} secuencias train, {len(xv)} val "
          f"(~{len(x)*cfg['SEQ']/1e3:.0f}k tokens)")
    assert len(x) > 100, "dataset de train demasiado pequeño"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}
    for name in ["cgmn", "transformer"]:
        print(f">> entrenando {name.upper()}...")
        model = create_model(name, cfg)
        nparams = sum(p.numel() for p in model.parameters())
        print(f"   parámetros: {nparams/1e6:.2f}M")
        hist = train_model(name, model, x, y, m, xv, yv, mv, cfg)
        results[name] = {"nparams": nparams, **hist}

    with open(os.path.join(cfg["OUT_DIR"], "results_tinystories.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        ax.plot(r["step"], r["val_ppl"], marker="o", label=name)
    ax.set_xlabel("step"); ax.set_ylabel("perplejidad de validación")
    ax.set_title("CGMN vs Transformer mini — TinyStories (menor = mejor)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["OUT_DIR"], "tinystories_ppl.png"), dpi=150)
    print(f">> guardado: {os.path.join(cfg['OUT_DIR'], 'results_tinystories.json')} y "
          f"{os.path.join(cfg['OUT_DIR'], 'tinystories_ppl.png')}")
    print("RESULTADO FINAL:")
    for name, r in results.items():
        print(f"  {name:12s} ppl_final={r['val_ppl'][-1]:.1f}")
    return results

# En Colab (get_ipython existe) se ejecuta todo automáticamente al pulsar
# "Ejecutar todo". Fuera de Colab no se ejecuta nada (import seguro).
if "get_ipython" in globals():
    RESULTS = run_all(CONFIG)

# %%
# ========== Celda 8: descarga automática de resultados ==========
def auto_download(cfg):
    try:
        from google.colab import files
        for f in ["results_tinystories.json", "tinystories_ppl.png"]:
            p = os.path.join(cfg["OUT_DIR"], f)
            if os.path.exists(p):
                files.download(p)
        print(">> descarga automática iniciada (revisa tu carpeta de Descargas)")
    except Exception as e:
        print("No se pudo descargar automáticamente:", e)
        print("Los archivos quedaron en /content/ dentro de la sesión")

if "get_ipython" in globals():
    auto_download(CONFIG)

# %%
# ========== Celda 9: FIN ==========
print("Notebook terminado. Si no ves la descarga, entra a /content/ y descarga "
      "results_tinystories.json y tinystories_ppl.png manualmente.")

# %% [no-notebook]
# ========== Smoke test local (python3 colab_tinystories.py) — NO va al .ipynb ===
if __name__ == "__main__":
    def _smoke():
        print("SMOKE TEST (datos sintéticos, sin red)")
        outdir = "/tmp/opencode/tinystories_smoke"
        os.makedirs(outdir, exist_ok=True)
        cfg = {**CONFIG, "SEQ": 16, "BS": 8, "VOCAB_SIZE": 60, "MIN_FREQ": 1,
               "N_STORIES": 40, "MAX_STORY": 60, "EPOCHS": 1, "EVAL_EVERY": 2,
               "D_MODEL": 32, "FFN": 64, "HEADS": 2, "OUT_DIR": outdir}
        words = [f"w{i}" for i in range(40)]
        rng = random.Random(1)
        toks = [rng.choices(words, k=40) for _ in range(cfg["N_STORIES"])]
        itos, stoi = build_tokenizer(toks, cfg["VOCAB_SIZE"], cfg["MIN_FREQ"])
        x, y, m = encode_pack(toks, stoi, cfg["SEQ"])
        assert x.shape[1] == cfg["SEQ"] and len(x) > 5
        assert (x == 0).sum() > 0 and m.sum() > 0
        for name in ["cgmn", "transformer"]:
            model = create_model(name, cfg)
            assert model.embed.weight is model.out.weight, "head no atado"
            hist = train_model(name, model, x, y, m, x[:4], y[:4], m[:4], cfg)
            assert hist["val_ppl"], "sin métricas"
            ck = os.path.join(outdir, f"checkpoint_{name}.json")
            assert os.path.exists(ck), "checkpoint no guardado"
        print("SMOKE OK")
    _smoke()
