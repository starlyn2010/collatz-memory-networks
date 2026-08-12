# Spec — Sub-proyecto B: Benchmarks sintéticos (Colab GPU)

**Fecha:** 2026-08-12 · **Estado:** propuesto y aprobado por el investigador
**Propósito:** validar, en tareas sintéticas que NO son memorización pura,
si la **puerta de actualización modulada por valuaciones Collatz**
("reloj de posición 2^-k" de la teoría) aporta sobre (a) una GRU estándar y
(b) una compuerta aleatoria. Se busca aislar el efecto del reloj Collatz del
efecto de "tener memoria" (que ya se validó en TinyStories).

---

## 1. Alcance y criterios de éxito

**Incluye:** 5 tareas sintéticas + 7 arquitecturas (3 variantes de puerta Collatz
+ 4 baselines), 5 semillas, entrenamiento en GPU Colab T4, métricas de
exactitud media ± std, figura y JSON de resultados, registro en BITACORA.

**Excluye (fuera de esta spec):** generación de español (idea A), conexión a
modelos preentrenados (idea C), datasets grandes (solo datos sintéticos
generados en memoria), entrenamiento a >1.5 h por corrida.

**Criterio de éxito:** la celda pasa sus propias regresiones anti-fuga y los
resultados se registran en la bitácora. No exige "ganar todo": la hipótesis
nula ("la puerta Collatz no ayuda vs RandomGate") es un resultado científico
válido.

---

## 2. Tareas (S = 128 tokens universales)

Todas sobre vocabulario pequeño (bits / símbolos {a,b,c} / clases), generadas
en memoria con `random.Random(seed)`. Cada tarea: 4096 ejemplos train, 1024
val, 1024 test; train/val/test disjointos.

| # | Tarea | Input (S=128) | Target | Formato | Clases |
|---|-------|---------------|--------|---------|--------|
| 1 | XOR / Paridad | 128 bits | paridad del prefijo 1..t | per-position | 2 |
| 2 | Contar | 128 símbolos {a,b,c} | # de 'a' en la secuencia | many-to-one (último token) | 129 (0..128) |
| 3 | Reverse | 128 símbolos | secuencia invertida | per-position | V_sim |
| 4 | Suma binaria | a(0-255) SEP b(0-255) pad... | a+b | many-to-one | 511 (0..510) |
| 5 | Primera aparición | símbolos objetivo T | índice = posición del primer T | many-to-one | 128 (0..127) |

Detalles de formato:
- **per-position:** `y[t] = x[t+1]` shift; loss solo sobre tokens reales.
- **many-to-one:** el modelo lee la secuencia; la predicción se toma del hidden
  del **último token real** proyectado a clases. La "clase" se codifica como un
  token especial `<CLS_k>` anclado al vocabulario (o un head lineal separado si
  las clases no son tokens).
- SEP y PAD = tokens reservados (`<pad>`, `<unk>`, `<sep>`, más `<cls_*>`).

---

## 3. Arquitecturas (7 modelos, ~120K-180K params)

- **CGMN (CollatzFix1)** — CollatzMemoryStack(2 capas, d=128, kappa_max=10) +
  valuaciones precomputadas (modo índice) + `W_m` entrenable. La puerta real.
- **CGMN NoCollatz** — igual, pero `m_t = 1` (GRU estándar sobre la misma
  arquitectura de pila). Control directo.
- **CGMN RandomGate** — igual, pero `m_t ~ Bernoulli(0.5)` reproducible por
  semilla (via `mask_random`). Pregunta clave: ¿la estructura temporal de 2^-k
  le importa a la perfomance vs azar con la MISMA media 0.5?
- **RNN** — `nn.RNN` (tanh), 2 capas, d=128. Baseline clásico.
- **GRU** — `nn.GRU`, 2 capas, d=128. Baseline clásico.
- **Transformer mini** — 2 capas, 2 cabezas, d=128, FFN 512, máscara causal
  tril(0) + padding float (el que ya validé es libre de fuga).
- **CfC** — celda de orden continuous-time (Hasani) versión cerrada, sin ACT.
  Implementación inline (ver nota de dependencias).

Gate override implementation: `CollatzMemoryStack.forward(x_seq, valuations=None,
gate_override=None)`. Para RandomGate/NoCollatz se pasa un `m` (T,1,hidden) o
se deja `valuations=None` (→ ones) / se inyecta `mask_random(T,B,hidden,seed)`
como `gate_override`. La cabeza de salida (per-position / many-to-one) se
comparte per-model, salvo Transformer (usa su propio forward).

---

## 4. Protocolo de entrenamiento

- **Optimizador:** AdamW, lr=3e-4 base.
- **LR:** decaimiento coseno 3e-4 → 3e-5, restart cálido desactivado.
- **Batch:** 64. **Steps máx:** 500 (early-stop: val_acc no mejora 30 steps).
- **Clip:** 1.0. **Dropout:** 0.1 (solo en Transformer/CfC; GRU/CGMN sin
  dropout para no confundir el efecto de la puerta).
- **5 semillas** (0..4) por modelo×tarea → media ± std.
- **Eval cada 25 steps**, checkpoint solo del mejor val_acc.
- **Tamaño secuencia:** S=128 siempre (uniformiza, como acordado).

**Presupuesto (T4):** ~30-60 s/corrida × 7×5×5 = 175 corridas ≈
**30-60 min** total. Cada corrida con S=128, d=128 se mantiene <90 s (la
máscara Collatz es barata: O(T·d) por paso, no atención cuadrática).

---

## 5. Métricas y salidas

- **Métrica primaria:** exactitud final en TEST (media ± std sobre 5 seeds).
- **Tabla en BITACORA (Sesión 13):** por tarea, 7 modelos, acc mean±std;
  filas de CGMN-/-NoCollatz/-RandomGate subrayadas para la pregunta científica.
- **Figura 1:** barras por tarea (7 modelos), std como error bar; colores:
  CGMN = verde Collatz, RandomGate = naranja "ruido", baselines = gris.
- **Figura 2 (diagnóstico):** para CGMN, trazas de `z_eff` media, `m_t` media y
  norma de `h` sobre tiempo (track opcional del stack) — visualiza el reloj.
- **JSON:** `outputs/exp13_benchmarks.json` con acc/media/std por tarea×modelo.
- **Colab:** `colab/colab_benchmarks.py` + `.ipynb`; auto-descarga de
  `results_benchmarks.json`, `fig_bench.png`, `fig_reloj.png`.

**Interpretación esperada (para la bitácora):**
- Si CGMN > RandomGate ≥ NoCollatz en tareas posicionales (paridad, primera
  aparición) pero no en Contar/Reverse/Suma → el reloj Collatz aporta
  **información posicional**.
- Si RandomGate ≈ CGMN → no hay señal temporal; hay que replantear el diseño
  de la puerta (hipótesis refutada).

---

## 6. Regresión anti-fuga (obligatoria)

- En el smoke local: datos sintéticos sin estructura (tokens uniformes) →
  `val_acc` no puede ser ~1.0 para ningún modelo con máscara causal correcta.
  `assert val_ppl > e^1` (ya usado en los smokes del vault/slots).
- Para CGMN: verificar que el attention-free stack procesa `x_seq (T,B,d)`
  con `T=128`, `valuations (T,)` → shape `(T,B,hidden)` consistente, y que
  `gate_override=None` con valuaciones produce `m` ≠ constante (la puerta
  varía con t) — comprueba que la máscara Collatz está conectada.

---

## 7. Notas técnicas / decisiones

- **CfC inline:** se reusa una implementación mínima de la ecuación cerrada
  (tanh con time-scale constante γ), sin ACT ni dependencias raras. Si CfC
  añadiera >2 héroes de complejidad, se documenta en la bitácora y se marca
  como "CfC-ausente" (los 6 modelos restantes no se ven afectados).
- **No se usan** `mask_collatz_fix2/fix3` (RQMC / entropy scheduling) — esas
  son variantes de la *fuente* de valuaciones, no del modelo. La ablación
  aquí pesa la PUERTA (real vs no-puerta vs aleatoria), usando valuaciones
  modo-índice fija para CGMN. Se puede extender en una futura spec.
- **Vocabulario de salida** se construye por tarea (dinámico): para many-to-one
  se añade `|n_clases|` tokens `<cls_k>` al vocab y el loss mira solo la posición
  del último token real; para per-position se reusan los tokens del input.

---

## 8. Orden de sesiones (plan de fases)

- **Sesión 13 (B):** este spec → colab_benchmarks.py + .ipynb → correr T4.
- **Sesión 14 (A):** colab de español (TinyStories-ES / corpus infantil pequeño)
  entrenando archivador ganador + vanilla@512; samples de generación.
- **Sesión 15 (C):** conectar memoria a modelo preentrenado → usar
  `PlanTL-GOB-ES/gpt2-small-spanish` (o DistilGPT2) como corteza congelada +
  bóveda/archivador como compresor de prompt blando; evaluar ppl en texto largo.

## 9. Self-review

- [x] Sin placeholders: todas las tareas, modelos y thresholds están definidos.
- [x] Coherente: arquitectura (CollatzMemoryStack + mask) con el forward real
  de `models/collatz_memory_cell.py`.
- [x] Scoping: S=128, d=128, 500 steps → budget T4 respetado (<90 s/corrida).
- [x] Ambigüedad resuelta: many-to-one usa clases `<cls_k>` ancladas al vocab.
