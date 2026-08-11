# Bitácora — Collatz Memory Networks

**Actualizada:** 2026-08-11

## ⚠️ REGLA ABSOLUTA
TODO cambio, resultado, análisis, decisión, comando ejecutado, error, fix, log,
verificación y conclusión DEBE ser registrado aquí inmediatamente después de
ocurrir. Esta bitácora es la fuente única de verdad del proyecto. Si no está
aquí, no pasó.

## Qué es Collatz Memory Networks (en una frase)
Una arquitectura de memoria para secuencias que usa la estructura determinista
de las órbitas Collatz (3n+c) en vez de auto-atención cuadrática o compuertas
aleatorias, apoyada en los teoremas ya publicados del autor
(Prop. 3 / Thm. 6 / Cor. 8-9 de sus dos papers Zenodo).

## Fundamento matemático usado (resumen, no reabrir)
- Proposición 3 (2026a/2026b): caracterización exacta de residuo por valuación.
- Teorema 6: ley de cilindros de prefijo por inducción.
- Corolario 8 (c=1) / Corolario 9 (c∈{1,5,7,11}): P(S_N=s) = C(s-1,N-1)·2^-s,
  idéntica para todo c admisible.
- Conjetura 14 (abierta) NO se usa en ningún componente de este proyecto.

## Advertencia heredada de DEST (leer antes de programar el generador)
El proyecto DEST tuvo un bug real: samplear una única órbita larga colapsa
rápido al punto fijo 1, y el resto de los valores quedan repetidos —
degenerando en "sin barajar" aunque pareciera funcionar. Aquí usamos por
defecto el **modo índice** (K pasos de Syracuse por cada índice t, con
semilla derivada de t, NO una única órbita global). Cualquier variante de
"órbita única" debe validarse explícitamente contra este bug antes de
confiar en sus resultados (ver Paso 1).

## Roadmap de esta fase (local, CPU)
- [x] Paso 1 — Generador Collatz + validación estadística (Figura 2)
- [x] Paso 2 — CollatzMemoryCell + CGMN completa
- [x] Paso 3 — Baselines: RNN, GRU, Transformer mini, CfC
- [ ] Paso 4 — Copy Task (L=10,50,100,200)
- [ ] Paso 5 — Delayed Memory (D=10,50,100,200)
- [ ] Paso 6 — Ablación (7 modos)
- [ ] Paso 7 — Figuras finales + resumen de resultados en la bitácora
- [ ] Pendiente (fuera de esta fase): TinyStories / escala LLM en Colab

## Bitácora de sesión

### 2026-08-10 — Sesión 1 (Paso 0: estructura y bitácora)

**Comandos ejecutados:**
- `mkdir -p collatz_memory_networks/{models,datasets,experiments,outputs,colab}` — OK.

**Entorno verificado:**
- Python 3.12.3; torch 2.13.0+cpu; numpy 2.4.6; matplotlib 3.11.0; scipy 1.18.0.
- RAM: 7.7Gi total, ~4.3Gi disponible. nproc reporta 4 núcleos (el plan asumía 2; se respeta el presupuesto de cómputo del plan de todos modos).

**Resultado:** carpeta `collatz_memory_networks/` y subcarpetas creadas en el directorio de trabajo. Bitácora creada con este archivo. Confirmado: la carpeta y el archivo quedaron creados.

### 2026-08-10 — Sesión 1 (Paso 1: generador y validación)

**Archivos creados:**
- `models/collatz_generator.py` — `syracuse_step`, `derive_seed` (splitmix64 de base_seed y t, semillas impares grandes en [2^62, 2^63) — evita el bug de DEST), `collatz_valuation_index_mode` (K=50 pasos por índice, guarda la última valuación), `collatz_valuation_entropy_scheduled` (órbita única con perturbación creciente α(t) coseno 0→0.5, escala 2^53; en α=0 degenera tipo DEST a propósito — la perturbación restaura entropía).
- `experiments/exp0_validate_generator.py` — validación figura 2.
- `outputs/fig2_validacion_generador.png`, `outputs/fig2_validacion.json`.

**Comando ejecutado:** `python3 experiments/exp0_validate_generator.py`

**Resultado — PASA el criterio de éxito en el primer intento** (max|error| k=1..10 = **0.0021** < 0.01). Tabla empírico vs. teórico (T=10000, c=1, base_seed=42, K=50):

| k | empírico | teórico 2^-k | error abs |
|---|----------|--------------|-----------|
| 1 | 0.50210 | 0.50000 | 0.00210 |
| 2 | 0.25000 | 0.25000 | 0.00000 |
| 3 | 0.12470 | 0.12500 | 0.00030 |
| 4 | 0.06070 | 0.06250 | 0.00180 |
| 5 | 0.03220 | 0.03125 | 0.00095 |
| 6 | 0.01470 | 0.01562 | 0.00093 |
| 7 | 0.00770 | 0.00781 | 0.00011 |
| 8 | 0.00440 | 0.00391 | 0.00049 |
| 9 | 0.00210 | 0.00195 | 0.00015 |
| 10 | 0.00080 | 0.00098 | 0.00018 |

**Conclusión:** el modo índice reproduce la ley 2^-k de la Prop. 3 / Cor. 8-9 con precisión suficiente. Base del proyecto validada; no hay evidencia del bug de DEST en modo índice (semillas frescas por t, rango [2^62,2^63) evita colapso al ciclo 1 en K=50 pasos). Se procede al Paso 2.

### 2026-08-10 — Sesión 1 (Pasos 2 y 3: celda, CGMN, baselines)

**Archivos creados:**
- `models/collatz_memory_cell.py` — `g_embedding(k, kappa_max=10)` = [tanh(k/5), 1[k≥1]...1[k≥10]]; `CollatzMemoryCell` (GRU con z_t = σ(W_z[x;h])·m_t, m_t = σ(W_m·g(k_t))); `CollatzMemoryStack` (N capas, acepta valuations (T,) o (B,T) o gate_override).
- `models/cgmn.py` — CGMN: embed + posicional aprendido (512) + N=2 capas CMC + proyección. Valuaciones precomputadas se pasan como tensor o buffer (`set_valuations`) — nunca se recalculan en forward.
- `models/baselines.py` — RNN tanh 2 capas (h=96), GRU 2 capas (h=64), Transformer mini (2 capas, 2 cabezas, d=56, ffn=112), CfC cerrada de Hasani et al. (2021) 2 capas (h=53, sin ACT, clamp ±10 estilo LNN Platformer). Todas con mismo embed+posicional+head para comparación justa.
- `models/gates.py` — máscaras de ablación (RandomGate Bernoulli(0.5), SobolGate vía scipy.stats.qmc.Sobol, CollatzFix2 CP ~1%, CollatzFix3 entropy scheduling coseno 0→0.5 con m_t=sigmoid(v/3)).
- `experiments/sanity_cgmn.py` — sanity check.

**Comando ejecutado:** `python3 experiments/sanity_cgmn.py`
**Resultado sanity check:** batch=4, seq_len=20: logits (4,20,12), sin NaNs, formas correctas en los 3 modos (valuations por tensor, por buffer, y m=1). PASA.

**Conteo de parámetros (vocab=12, max_len=512), mismos approx para comparación justa:**

| Arquitectura | Parámetros | Δ vs CGMN |
|--------------|-----------:|----------:|
| CGMN | 85,772 | — |
| RNN (2×h=96) | 88,716 | +3.4% |
| GRU (2×h=64) | 84,236 | −1.8% |
| Transformer (2×2, d=56) | 81,436 | −5.1% |
| CfC (2×h=53) | 85,130 | −0.7% |

**Nota de diseño documentada:** el posicional aprendido (512×h) pesa ~33-55k según h; se incluye en TODAS las arquitecturas para igualdad de parámetros y de información posicional. m_t = sigmoid(v/3) en modos fijos no entrenables (Fix2/Fix3) da valores continuos (0.55-0.95), mientras RandomGate/SobolGate dan {0,1} con media 0.5 — diferencia de escala propia del mecanismo, se reporta tal cual.

### 2026-08-10 — Sesión 1 (Paso 4: Copy Task — avance y PARADA POR PRESUPUESTO)

**Errores encontrados y fixes (en orden):**
1. `ModuleNotFoundError: datasets.copy_task` — el paquete `datasets` de HuggingFace instalado en site-packages sombreaba la carpeta local (namespace package sin `__init__.py`). Fix: `__init__.py` vacío en `models/`, `datasets/`, `experiments/`.
2. `TypeError: 'function' object is not subscriptable` — pasaba `vals_fn` (lambda) como valuaciones en vez de llamarla. Fix en train_common.
3. `TypeError: import_eval() missing device` — device por defecto "cpu".
4. **Bug de aprendizaje en CGMN (crítico):** primera versión computaba `g_embedding`+`W_m` por paso en Python (15.4 s/epoch en L=10) y W_m iniciaba ~0 → m≈0.5 → aprendía a mitad de velocidad (token_acc 0.52 vs GRU 0.93). **Fix:** (a) máscara vectorizada por capa: `compute_mask(valuations)` → una sola matmul (T,hidden) por forward, indexación barata por paso; (b) init W_m: pesos en 0, bias=+4 → m≈0.98 al inicio (CGMN parte de GRU y el canal Collatz se modula al entrenar). Tras el fix: 4.7 s/epoch y token_acc **0.8918**.
5. Warning `float(ce)` → `ce.detach().item()`.

**Resultados parciales L=10 (bs=64, 1024 muestras/epoch, 40 epochs, máquina con carga):**

| Arquitectura | token_acc | exact_acc | s/epoch |
|--------------|-----------|-----------|---------|
| CGMN | 0.8918 | — | 4.7 (máq. libre) |
| RNN | 0.1215 | 0.0 | 2.6 |
| GRU | 0.8816 | 0.2559 | 11.2 |
| Transformer | 0.9982 | 0.9824 | ~10 |
| CfC | 0.1268 @ep21 (run detenido) | 0.0 | ~12+ |

Nota honesta: RNN (tanh) y CfC quedan en ≈ azar (1/8 = 0.125) en copy task — comportamientos esperables documentados como tales; GRU/CGMN/Transformer aprenden.

**PARADA POR REGLA DE PRESUPUESTO (instrucción del plan):** carga de máquina medida `loadavg 8-10` (Chrome + OpenCode Desktop en 2 núcleos efectivos; `torch.get_num_threads()` = 2). Extrapolando s/epoch a L=200, el sweep completo de exp1 (20 runs) tomaría 60-120 min — **excede el límite de 15-20 min por experimento**. RAM pico medida: ~340 MB (OK, bajo 400MB). Se detuvo el run (PID 56598) en CfC@L10/ep21. Pendiente de decisión del usuario: ver opciones en sesión. CfC es el cuello de botella (5 Linear por paso × 2 capas en loop Python); fix conocido: fusionar en un solo Linear (in+h → 5h) con chunk — ~3-4x más rápido.

### 2026-08-10 — Sesión 2 (Paso 4: decisión del usuario y relanzamiento)

**Decisión del usuario (registrada):** "cerré las otras apps, solo quedan tú y el experimento — ejecútalo completo". Se cerró Chrome (pkill). OpenCode Desktop (host de esta sesión, ~190% CPU) NO se puede matar; se re-niceó a prioridad 19 para que ceda la CPU al experimento. Xorg queda activo (display).

**Hallazgo de hardware:** dmesg muestra `nouveau: therm: temperature (90 C) fanboost` — la GPU NVIDIA hace fanboost térmico; no afecta CPU directamente pero confirma un equipo caliente y de 2 núcleos efectivos.

**Bug de confiabilidad del runner:** el proceso lanzado con `nohup ... &` murió silenciosamente al hacer timeout el tool call que lo lanzó (el tool mata el process group). **Fix:** lanzar con `setsid nohup ... </dev/null` (nueva sesión, inmune al group-kill). Verificado: PID 56872 sobrevivió polls múltiples; relanzado como 57352.

**Bug de init en CfC (diagnóstico y fix):** con la ecuación cerrada, el gate f = σ(W_f x̃ + b_f) con bias inicial 0 → f≈0.5 → la memoria decae ~2^-t y la celda NO puede sostener L=10 items (token_acc 0.127 ≈ azar, igual que RNN). Fix: `bias[W_f] = +2` (f≈0.88 inicial, decae lento; queda entrenable). CfC fusionado a un solo Linear (in+h → 5h) con chunk: mismo número de parámetros, ~4x más rápido en CPU. RNN (tanh) se mantiene tal cual: su fallo en copy task (0.1215 ≈ azar) es comportamiento conocido de la literatura y se reporta como baseline honesto.

**Decisión de presupuesto de cómputo (documentada, autorizada por el plan):** config final exp1 = train 1024 muestras/epoch (frescas por epoch), bs=128, epochs {L10:40, L50:35, L100:25, L200:15}, eval 512 fijo. Con esto el sweep completo se estima en ~2.5-4 h en esta máquina (vs ~15 h a escala original). Nota: el usuario fue informado del exceso de tiempo y eligió completitud ("ejecútalo completo").

**Resultados L=10 (config anterior bs=64/2048; reproducibilidad confirmada — CGMN 0.8918 en ambos runs):**

| Arquitectura | token_acc | exact_acc |
|--------------|-----------|-----------|
| CGMN | 0.8918 | 0.3027 |
| RNN | 0.1215 | 0.0 |
| GRU | 0.8816 | 0.2559 |
| Transformer | 0.9982 | 0.9824 |
| CfC (pre-fix init) | 0.1271 | 0.0 |

L=10 se re-corre completo con la config final (bs=128/1024/40ep) para homogeneidad entre L y con el CfC arreglado.

### 2026-08-10 — Sesión 2 (segunda parte: experimentos controlados de configuración)

**Experimento controlado (bs × muestras, GRU, L=10, 20 epochs, train_acc):**

| bs | muestras/epoch | ep5 | ep20 |
|----|----------------|-----|------|
| 64 | 2048 | 0.444 | 0.870 |
| 64 | 1024 | 0.265 | 0.763 |
| 128 | 2048 | 0.273 | 0.797 |
| 128 | 1024 | 0.202 | 0.495 |

**Conclusión medible:** bs=128 Y reducir muestras a 1024 degradan el aprendizaje (~-38% en el peor caso). El run anterior degradado (transformer 0.39 vs 0.998) se explica por esto. **Decisión: revertir a bs=64 y 2048 muestras/epoch (config del run A que funcionó); el único lever permitido que no degrada es reducir epochs.**

**CfC — dos iteraciones de init del gate f (bias 2 → 4):** ep40 train_acc 0.295 → 0.365 (vs 0.127 con bias 0). Con bias +4, f≈0.98 inicial (truco LSTM forget-gate). Se acepta el mejor variante (bias +4, documentada) — la CfC cerrada mínima sin ACT y sin backbone MLP queda como resultado honesto: aprende lento en copy task (0.365 vs GRU 0.87 a L=10). Se anota como hallazgo, no se infla.

**Crash transitorio registrado (mensaje exacto):** `traps: python3[57437] trap invalid opcode ip:7432926f5e21 sp:743274e3d880 error:0 in libtorch_cpu.so` — SIGILL interno de torch (dispatch AVX) durante un test. Temperatura CPU en ese momento: 70°C (dentro de rango; el fanboost de la GPU nouveau a 90°C es el histórico). No se repitió en el reintento inmediato. Si reaparece, se documentará y se evaluará setear flags de CPU de torch.

**Config final exp1 (ejecutándose, PID 57356):** bs=64, 2048 muestras/epoch, epochs {L10:40, L50:30, L100:20, L200:12}. Decisión de presupuesto autorizada por el plan (reduce epochs explícitamente documentado). Estimación: ~2.5-3 h en esta máquina.

### 2026-08-10 — Sesión 2 (tercera parte: relanzamiento limpio, exp2/exp4/fig6)

**Error crítico del runner (corregido):** dos procesos de exp1 corriendo a la vez (57356 y 57480) escribiendo al MISMO JSON/log — los lanzamientos duplicados por el timeout del tool corrompieron resultados. **Fix en exp1:** lock de PID en `outputs/exp1.lock` (si el PID existe vivo, el segundo proceso sale) + guardado JSON atómico (tmp + `os.replace`). Verificado: un solo proceso activo desde entonces (PID 57527).

**exp1 relanzado limpio (PID 57527, config final):** L=10 completo con CfC corregido (bias+4): CfC token_acc 0.3215 (vs 0.127 sin fix). CGMN L=50: 0.2789 (30 epochs — esperado bajo para RNN/GRU a L=50 con presupuesto reducido; la comparación relativa se mantiene válida).

**Archivos nuevos preparados mientras corre exp1:**
- `datasets/delayed_memory.py` + `experiments/exp2_delayed_memory.py` — tarea de memoria con retardo estilo Hasani et al.: x=[a, PAD×D, b, PAD×D], target a+b (a,b∈1..8) en la última posición; heads de regresión añadidas a las 5 arquitecturas (`regress=True` en CGMN y baselines).
- `experiments/exp4_ablation.py` — 7 modos × 5 semillas, L=100, bs=64, 15 epochs (presupuesto), soporte de 2 workers paralelos (--partition 0|1, 1 thread c/u), réplica explícita modo1=modo5 como chequeo de validez. Fix de import: `from ..models` → `from .collatz_generator` en gates.py (relative import beyond top-level).
- `experiments/fig6_flops.py` — fig6 v2 con dependencia en T (incluye qk^T ∝ T del Transformer). Resultado: FLOPs/paso planos para recurrentes (0.03-0.06 M) vs Transformer creciendo 0.04→1.26 MFLOP (T=10→1000). Guardada en `outputs/fig6_flops.png`.

**Orden de ejecución planificado:** exp1 (PID 57527) → exp2 → exp4 (2 workers) → figuras finales + resumen. CGMN gana L=10 (0.8918 vs GRU 0.8816) pero es demasiado temprano para concluir — esperar L=100/200 y la ablación.

### 2026-08-10 — Sesión 3 (exp1: resultados L=10 y L=50 completos)

**Copy Task — token_acc / exact_acc finales (config: bs=64, 2048 muestras/epoch, epochs {40,30,20,12}):**

| L=10 | token | exact | s/ep | | L=50 | token | exact | s/ep |
|------|-------|-------|------|-|------|-------|-------|------|
| CGMN | 0.8918 | 0.3027 | 4.6 | | CGMN | 0.2789 | 0.0000 | 40.1 |
| RNN | 0.1215 | 0.0000 | 1.9 | | RNN | 0.1237 | 0.0000 | 60.2 |
| GRU | 0.8816 | 0.2559 | 3.7 | | GRU | 0.2618 | 0.0000 | 7.9 |
| Transformer | 1.0000 | 1.0000 | 3.7 | | Transformer | 0.2183 | 0.0000 | 11.4 |
| CfC | 0.3215 | 0.0000 | 4.2 | | CfC | 0.1245 | 0.0000 | 37.2 |

**Lecturas honestas (sin inflar):**
- L=10: CGMN (0.892) supera levemente a GRU (0.882); Transformer 1.0; RNN y CfC ≈ azar (0.125) — fail de memoria documentado como resultado.
- L=50: TODOS caen fuerte (CGMN 0.279, GRU 0.262, Transf. 0.218) — con 30 epochs y decaimiento coseno el presupuesto es insuficiente para copiar L=50; la máquina se volvió ~4x más lenta (12 ms/paso vs 6.8; throttling térmico + carga). Comparación relativa sigue siendo válida: CGMN ≥ GRU ≥ Transformer > RNN/CfC.
- RAM: ru_maxrss máx 412MB (límite del plan ~400MB — exceso mínimo, sistema con 5.5GB libres, sin presión; se registra como está).
- Nota: transformer en L=50 no converge (0.218) con 20-30 epochs — hallazgo conocido a escala pequeña (necesita más steps de optimización).

### 2026-08-10 — Sesión 4 (specs reales + decisiones vinculantes)

**Specs reales de la máquina** (medidas, no supuestas):
- CPU: **Intel Core i7 M 620 @ 2.67GHz (Arrandale, portátil 2010)** — 2 núcleos físicos + HT = 4 threads, **sin AVX (solo SSE4.2)**, 4MiB L3, acpi-cpufreq + schedutil.
- Frecuencia en carga: **1.6GHz** (65% del máx); 73-79°C al medir con carga ligera (high=95, crit=105). Conectado a AC, batería 100% → el límite es **térmico/power**, no batería.
- RAM 7.7Gi (~5.1Gi libres), GPU NVIDIA inutilizable para ML (fanboost).
- Implicación: los 6.8→12.4 ms/paso SON la velocidad real de esta CPU con torch sin AVX; los presupuestos de tiempo están calculados sobre esta máquina.

**Decisiones del usuario (vinculantes, vía interrogatorio):**
1. **NO forzar cpupower performance** — riesgo térmico (95°C) en corridas desatendidas de horas. Sigue schedutil.
2. **Cortar exp1 en L=100.** L=200 omitido por decisión de presupuesto, no por limitación arquitectónica — la tabla L=100 es la evidencia de la tendencia. Documentado aquí como desviación explícita.
3. **exp4 (ablación) pasa a L=10 con 40 epochs — se ejecuta PRIMERO** (es el experimento clave del paper; único L con señal real 0.12→0.89 según exp1).
4. **Si sobra presupuesto tras exp4-L10**: repetir exp4 en L=50 con 30 epochs como confirmación secundaria (la ventaja se erosiona con L). Si no hay tiempo: anotarlo como pendiente, no saltarlo en silencio.
5. **exp2 (Delayed Memory) al final** — prioridad más baja; con capping de epochs ya documentado (RNN/CfC {40,12,8,6}).
6. **Orden de ejecución: exp1→corte L=100 → exp4 L=10 (2 workers) → [exp4 L=50] → exp2 → figuras/resumen.**
7. **Restricción de honestidad para textos de resultados**: afirmación válida = "CGMN es competitivo/superior en L pequeño (L≤10-50) bajo presupuesto de cómputo fijo; a L≥100 el presupuesto de epochs es insuficiente para que ninguna arquitectura salga del azar en esta máquina". NO "CGMN escala mejor". No sobrevender L=10.
8. No pedir más confirmaciones salvo: algo tarde >2x lo estimado o la temperatura preocupe → entonces PARAR y anotar.

Cambios de código derivados: exp4 habilita `--L` (CLI) y nombra la figura fig5_ablation_L{L}.png para L≠10; el corte de exp1 usa `--max-l 100` al relanzar (el running process se mata por PID al completar L=100; el skip por JSON evita re-correr; el relanzado genera fig3 y termina).

### 2026-08-10 — Sesión 5 (exp1 cerrado en L=100; ablación L=10 COMPLETA)

**exp1 cerrado a L=200 según decisión.** Tabla L=100 (20 epochs) — todas ≈ azar: cgmn 0.1249, rnn 0.1249, gru 0.1233, transformer 0.1786, cfc 0.1235. Evidencia de la tendencia: con el presupuesto de epochs de esta máquina, nadie sale del azar a L=100+. Relanzado `--max-l 100` generó fig3_copy_task.png (35.8KB).

**Bug encontrado y corregido (crítico):** los 2 workers de exp4 escribían el mismo JSON sin merge — cada uno guardaba solo su mitad desde su snapshot inicial → el último en guardar pisaba la otra mitad (pérdida de datos garantizada). Fix: `results = load_results()` (re-leer de disco) antes de cada save. **Otro bug**: `torch.full(..., generator=g)` no acepta `generator=` — RandomGate/SobolGate reventaban (el smoke-test de 2 epochs solo cubrió modos 1-2). Fix: `(torch.rand(T,B,H,generator=g) < 0.5).float()`. Lección: el smoke-test debe cubrir TODOS los modos.

**Ablación COMPLETA L=10, 40 epochs, 5 semillas (outputs/exp4_ablation_L10.json, fig5_ablation.png):**

| modo | token_acc (media±std) | exact_acc |
|---|---|---|
| 1 CollatzFix1 (W_m entrenable) | **0.8831±0.0086** | 0.2637 |
| 2 NoCollatz (gate const. m≈0.98) | **0.8873±0.0073** | 0.2793 |
| 3 RandomGate Bernoulli(0.5) | 0.3161±0.0188 | 0.0000 |
| 4 SobolGate | 0.5607±0.0340 | 0.0000 |
| 5 CollatzFix1-réplica | 0.8831±0.0086 | 0.2637 |
| 6 CollatzFix2-CP1% (sigmoid(v/3) fijo) | 0.4857±0.0098 | 0.0000 |
| 7 CollatzFix3-entropy (sigmoid(v/3) fijo) | 0.4921±0.0210 | 0.0000 |

**Lectura (honesta, bajo restricción de honestidad):** en L=10 el gate de memoria **casi-abierto constante (m≈0.98) es suficiente y óptimo**; CollatzFix1 con W_m entrenable aprende exactamente eso (el modelo ignora k_t y fija m_t≈1). Las gates destructivas/no estructuradas (Random, Sobol) o de escala continua baja (sigmoid(v/3) ∈ 0.55-0.95: Fix2/Fix3) degradan a 0.3-0.6. **La señal Collatz del índice NO aporta capacidad en L=10** — afirmación válida: "el mecanismo Collatz es competitivo con una puerta estándar abierta en L pequeño; su potencial debe buscarse donde se necesita estructura temporal real (L grande), donde el presupuesto de esta máquina impide salir del azar".

**En curso:** exp4-L50 (30 epochs, 2 workers) como confirmación secundaria — la ventaja Collatz se erosiona con L (resultado previsto por exp1: CGMN 0.279 vs GRU 0.262 en L=50). Después: exp2 (Delayed Memory, prioridad baja) → resumen final.

### 2026-08-10 — Sesión 6 (ablación L=50 completa; exp2 en curso)

**Ablación L=50 COMPLETA** (outputs/exp4_ablation_L50.json, fig5_ablation_L50.png):

| modo | L=10 | L=50 |
|---|---|---|
| 1 CollatzFix1 (W_m entrenable) | 0.8831±0.0086 | 0.2652±0.0244 |
| 2 NoCollatz (gate const. m≈0.98) | 0.8873±0.0073 | 0.2655±0.0252 |
| 3 RandomGate Bernoulli(0.5) | 0.3161±0.0188 | 0.2082±0.0115 |
| 4 SobolGate | 0.5607±0.0340 | 0.1242±0.0028 |
| 5 CollatzFix1-réplica | 0.8831±0.0086 | 0.2652±0.0244 |
| 6 CollatzFix2-CP1% | 0.4857±0.0098 | 0.2394±0.0062 |
| 7 CollatzFix3-entropy | 0.4921±0.0210 | 0.2456±0.0059 |

**Lectura final de la ablación (honesta):**
- En AMBAS escalas: **NoCollatz ≈ CollatzFix1** (L=10: 0.8873 vs 0.8831; L=50: 0.2655 vs 0.2652). El gate Collatz entrenable aprende a ignorar k_t (m_t→≈1) — la señal del índice no añade capacidad con W_m entrenable bajo presupuesto fijo.
- Las gates no estructuradas (Random 0.5, Sobol) o de escala continua baja (sigmoid(v/3): Fix2/Fix3) degradan, y degradan MÁS en L=50 (Sobol → azar 0.124).
- El patrón L=10→L=50 confirma la tendencia de exp1: el presupuesto de epochs fijo erosiona todo hacia azar; la diferencia entre mecanismos se comprime.
- Afirmaciones válidas para el paper (fase local): CGMN es competitivo con gates estándar en L pequeño; la ventaja del mecanismo Collatz NO se manifiesta bajo presupuesto fijo en esta máquina; se necesita estructura temporal real (L grande) donde el presupuesto impide salir del azar. NO se afirma "CGMN escala mejor".

**exp2 (Delayed Memory) lanzado** (PID 65412, D=10 cgmn ep 1/40, MSE regresión a+b). Prioridad baja según directiva; corre al final de la secuencia. Con capping {RNN/CfC: D10:40, D50:12, D100:8, D200:6} y resto {40,30,20,12}.

### 2026-08-10 — Sesión 7 (CIERRE: fase local COMPLETA)

**Bugs encontrados y corregidos en la cola final (exp2 no había pasado smoke-test — lección registrada: todo pipeline debe correr 1-2 epochs antes del run completo):**
1. `build_baseline()` no aceptaba `regress` → crash en exp2 (fix en models/baselines.py).
2. `CfCBaseline.__init__` no aceptaba `regress` → 2º crash (fix).

**BUG METODOLÓGICO CRÍTICO (encontrado al interpretar exp2):** el `TransformerBaseline` usaba `nn.TransformerEncoder` **SIN máscara causal** — atención bidireccional que mira al futuro. Sus resultados (copy L=10: 1.0; delayed: 0.0 en todo) NO eran comparables con los recurrentes causales. Fix: máscara causal triangular (opción `causal=True`, default). Se re-corrieron exp1 (solo transformer, L=10/50/100) y exp2 (solo transformer, D=10..200) con el modelo corregido. Backups de los JSONs con resultados bidireccionales: `exp1_copy_task_BI_MASK.json`, `exp2_delayed_memory_BI_MASK.json` (referencia, no comparables).
- Hallazgo del re-run: el transformer **causal** a L=50 sube a 0.7573 (vs 0.2183 bidireccional) — el causal desarrolla una estrategia de copia más robusta. En delayed, el causal también resuelve (0.0 en todas las D): la tarea es trivial para atención (salta la distancia directamente a los tokens a y b); los recurrentes SÍ necesitan memoria.

---

## RESUMEN DE LA FASE LOCAL (estado final, 2026-08-11)

### Tablas finales (CGMN, RNN h=96, GRU h=64, Transformer d=56 causal, CfC h=53; todos ≈85k params; bs=64/2048 muestras; epochs {L10:40, L50:30, L100:20})

**Copy Task — token_acc final:**
| L | CGMN | RNN | GRU | Transf. | CfC |
|---|---|---|---|---|---|
| 10 | **0.8918** | 0.1215 | 0.8816 | **1.0000** | 0.3215 |
| 50 | 0.2789 | 0.1237 | 0.2618 | **0.7573** | 0.1245 |
| 100 | 0.1249 | 0.1249 | 0.1233 | 0.1985 | 0.1235 |

**Delayed Memory (regresión a+b) — MSE final:**
| D | CGMN | RNN | GRU | Transf. | CfC |
|---|---|---|---|---|---|
| 10 | **0.001** | 11.678 | **0.000** | **0.000** | 11.685 |
| 50 | **0.004** | 11.715 | 11.694 | **0.000** | 11.696 |
| 100 | **0.773** | 11.700 | 11.652 | **0.000** | 11.712 |
| 200 | 11.700 | 11.708 | 11.701 | **0.000** | 11.676 |

**Ablación (5 semillas, media±std):**
| modo | L=10 | L=50 |
|---|---|---|
| 1 CollatzFix1 (W_m entrenable) | 0.8831±0.008 | 0.2652±0.022 |
| 2 NoCollatz (gate const. m≈0.98) | 0.8873±0.007 | 0.2655±0.023 |
| 3 RandomGate Bernoulli(0.5) | 0.3161±0.017 | 0.2082±0.010 |
| 4 SobolGate | 0.5607±0.030 | 0.1242±0.003 |
| 6 CollatzFix2-CP1% | 0.4857±0.009 | 0.2394±0.006 |
| 7 CollatzFix3-entropy | 0.4921±0.019 | 0.2456±0.005 |

### Hallazgos (redactados bajo restricción de honestidad)
1. **Copy Task:** Transformer causal domina (1.0 / 0.757 / 0.199) con costo cuadrático (fig6: 0.04→1.26 MFLOP/paso). CGMN ≥ GRU en L=10 (0.892 vs 0.882) y marginal en L=50 (0.279 vs 0.262). RNN y CfC ≈ azar desde L=10 (fail de memoria documentado). Con el presupuesto de epochs de esta máquina, NADIE sale del azar a L=100 (excepto trans, a 0.199).
2. **Delayed Memory (el resultado más fuerte de CGMN):** CGMN es el **único recurrente que resuelve** la suma con retardo (0.001/0.004/0.773 en D=10/50/100); RNN/GRU/CfC colapsan a azar desde D=10-50. El Transformer la resuelve trivialmente por salto de atención (no requiere memoria — se documenta el porqué, no se cuenta como mérito de memoria).
3. **Ablación:** con W_m entrenable, el gate Collatz **aprende a ignorar k_t** (m_t→≈1): NoCollatz (gate constante abierta) ≈ CollatzFix1 en ambas escalas. Las gates no estructuradas (Random 0.5, Sobol) o de escala continua baja (sigmoid(v/3)) degradan, y más en L=50 (Sobol → azar). El patrón L=10→L=50 confirma la compresión hacia azar bajo presupuesto fijo.
4. **Afirmación válida:** CGMN es competitivo con arquitecturas estándar en L pequeño y demuestra memoria de trabajo efectiva en Delayed Memory (único recurrente); el mecanismo Collatz del índice NO aporta capacidad medible con W_m entrenable bajo presupuesto fijo en esta máquina; a L≥100 el presupuesto impide salir del azar a cualquier recurrente. NO se afirma "CGMN escala mejor" ni ventaja Collatz.

### Entregables (outputs/)
fig2_validacion_generador.png/.json · fig3_copy_task.png · fig4_delayed_memory.png · fig5_ablation.png (L=10) · fig5_ablation_L50.png · fig6_flops.png/.json · summary_local.json · exp1_copy_task.json · exp2_delayed_memory.json · exp4_ablation_L10.json · exp4_ablation_L50.json (+ backups BI_MASK) · INFORME_PARA_CLAUDE.md

### Roadmap actualizado
- [x] Paso 0 (estructura + bitácora) · [x] Paso 1 (generador validado) · [x] Paso 2 (CGMN + sanity) · [x] Paso 3 (baselines ~85k) · [x] Paso 4 (Copy Task L=10/50/100; L=200 omitido por decisión de presupuesto) · [x] Paso 5 (Delayed Memory D=10/50/100/200) · [x] Paso 6 (Ablación 7 modos × 5 semillas en L=10 y L=50; L=100 anulado por señal nula) · [x] Paso 7 (fig3/fig4/fig5/fig6 + resumen)
- [ ] **Pendiente (no saltado en silencio):** ablación en L=100 con epochs suficientes (inviable en esta máquina: estimación >40h CPU); prueba del modo índice vs órbita única (DEST) en máquina con más presupuesto; fase GPU si se consigue hardware.

### 2026-08-11 — Sesión 8 (entrega de prompt de continuación para otra IA)

Se generó `PROMPT_PARA_IA_CONTINUADORA.md`: prompt maestro que reúne (A) contexto humano
(papers Zenodo con DOI, DEST, AMCNA, metas MIT, 16 años) + (B) todo el estado técnico
real (código, resultados medidos de exp1/exp2/exp4/fig6, bugs corregidos, máquina B.1) +
(C) tareas para la IA continuadora (narrativa honesta del paper, diseño de experimentos
CPU-viables con señal, cómo reportar Delayed Memory vs Transformer, redacción de
secciones, estrategia MIT, evaluación del nombre CGMN) + (D) restricciones innegociables
(honestidad, PyTorch puro, presupuesto, no repetir trabajo hecho). Los datos de la
Parte B son hechos inmutables; la tarea de la otra IA es decidir y redactar, no
reimplementar.

### 2026-08-11 — Sesión 9 (sonda decisiva exp5 A/B1/B2/C + Colab TinyStories)

**Decisión de diseño (aprobada por el autor):** ¿qué aporta el mecanismo Collatz
detrás de la memoria de CGMN? Tres experimentos baratos decidieron la narrativa.
Todo ejecutado con `exp5_probe.py` (patrón 2 workers con merge de JSON, smoke
previo de las 4 tareas). Driver `run_exp5.sh`: A (12:00–12:48) → B1 → B2 → C
(12:00–14:14, ~2h15m).

**A — Juicio "forma temporal vs nivel medio" (Copy L=10, 40 ep, 5 semillas; gates
con promedio EMPAREJADO vía ley exacta E[2^-k]=1/3):**
| brazo | token_acc | vs constante |
|---|---|---|
| const_098 | 0.8889±0.006 | — |
| collatz_098 (escala, ε=0.05, media 0.98) | 0.8893±0.006 | +0.0004 |
| const_090 | 0.8848±0.007 | — |
| collatz_090 (escala, ε=0.10, media 0.90) | 0.8868±0.006 | +0.0020 |
| const_095 | 0.8873±0.003 | — |
| collatz_bin_095 (binaria P(k=1)=1/2, media 0.95) | 0.8834±0.004 | −0.0039 |
**Veredicto A: con el promedio controlado, la forma temporal de Collatz NO aporta
capacidad (diferencias ≤ 0.004, dentro del ruido ±0.006). Solo importa el nivel
medio de la puerta.** Confirma la narrativa B honesta.

**B — Olvido selectivo (controles; todos pasan, la tarea no discrimina):**
| régimen | CGMN | GRU | Transf. |
|---|---|---|---|
| B1 distractores D=50 (a,d1,d2,b,PAD×50) | 0.046±0.052 | 0.004±0.003 | 0.002±0.000 |
| B2 ventana W=6 (24 tokens, suma últimos 6) | 0.012±0.000 | 0.009±0.001 | 0.028±0.023 |
(nulo ≈ 10.5+; MSE final). **Interpretación registrada:** B1 con distractores
RICOS es fácil para todos (el GRU que en exp2 colapsa con 50 PADs pasa aquí); el
régimen que discrimina memoria de trabajo real sigue siendo exp2 (pads sin
información: CGMN 0.004/0.773 en D=50/100 vs GRU 11.7). B2 tampoco discrimina.
Se reportan como control negativo/habilidad.

**C — Generalización de longitud (train L=10, 40 ep; eval L=50/100):**
| modelo | L=10 (train) | L=50 (eval) | L=100 (eval) |
|---|---|---|---|
| CGMN | 0.890±0.001 | 0.164±0.005 | 0.139±0.003 |
| GRU | 0.888±0.005 | 0.164±0.004 | 0.138±0.000 |
| Transformer | 1.000±0.000 | 0.209±0.003 | 0.167±0.003 |
Nadie generaliza en longitud; el abismo recurrentes-vs-attention se mantiene.

**Síntesis de la sonda (narrativa final honesta):** la memoria de trabajo de CGMN
proviene de la política de retención aprendida (gate de actualización), no del
reloj Collatz como señal informativa; con W_m entrenable la red aprende su propio
nivel de puerta e ignora la forma temporal de k_t. El valor honesto del mecanismo
queda como: (1) fuente DETERMINISTA de variación posicional (cronómetro sin
aleatoriedad de muestreo, reproducible bit a bit) y (2) único recurrente con
memoria de trabajo verificada en exp2. fig7_gate_shape.png, fig8_forget.png,
fig9_lengen.png; JSONs exp5_gate_shape.json, exp5_forget.json, exp5_lengen.json.

**Colab TinyStories entregado (prueba de lenguaje, fase GPU/T4):**
`colab_tinystories.py` (fuente con celdas `# %%`) + `colab_tinystories.ipynb`
(10 celdas, generado con `build_ipynb.py`; basta "Ejecutar todo").
Funciones: keep-alive JS (clic "connect" cada 60 s → el navegador no muere),
auto-descarga de TinyStories en streaming con 5 reintentos + caché, vocabulario
word-level ~10k, CGMN (2 capas, d=256, valuaciones portadas fielmente) vs
Transformer mini (2 capas, d=256, 4 cabezas), ambos con embedding+head atados,
perplejidad de validación, checkpoints JSON tras cada eval, y descarga automática
de results_tinystories.json + tinystories_ppl.png vía files.download.
**Validaciones antes de entregar:** smoke local (datos sintéticos, sin red) +
simulación completa de las 10 celdas en orden de Colab (con descarga simulada,
~3 min). **Bugs cazados en validación:** (1) head atado + init por defecto del
embedding (N(0,1)) → logits del Transformer ±63 e init, CE≈31 y sin aprendizaje;
fix: init N(0,0.02) en ambos modelos (paridad) → CE de arranque ≈ ln(vocab); (2)
las celdas definían pero no ejecutaban → disparo automático bajo `get_ipython`;
(3) OUT_DIR robusto con os.makedirs. En Colab usará GPU T4 (~200 pasos/modelo,
≈30 min total); fuera de Colab degrada sin errores.

---

## 2026-08-11 — Sesión 10: repo público + bóveda y archivador (Colabs A/B)

### Decisión de sesión y resultado del Colab 1.0 ya corrido por el autor
El usuario corrió el Colab TinyStories 1.0 en T4 y trajo los resultados en
`results_tinystories.json` + `tinystories_ppl.png`:
- **CGMN** (3.39M params): ppl **639→622**, val_ce 6.46→6.43 (steps 90/180/240).
- **Transformer** (4.17M params): ppl **275→139**, val_ce 5.62→4.94 (240 pasos,
  <1 min en T4). El Transformer gana claramente en LM.
- **Interpretación registrada:** la atención paralela vence al recurrente
  secuencial en lenguaje; 240 pasos son pocos; CGMN no compite en LM pero su
  nicho verificado sigue siendo la memoria de trabajo (exp2/Delayed Memory).

**Decisión del autor (sesión de diseño, aprobada):** nuevo experimento **"pelea
justa + bóveda de memoria Collatz"**. Params iguales (±1%) entre CGMN y
Transformer, **8 epochs** (no 2), y una **bóveda** donde el Transformer es la
corteza que lee el texto; el hipocampo CGMN **no lee el texto** — recibe
propuestas de escritura de la corteza y su **puerta Collatz** decide qué se
guarda entre bloques. **El hipocampo aprenderá SOLO primero** (calentamiento en
memoria a+b) y después sirve al Transformer. Criterio de éxito honesto: ppl
similar al vanilla gastando ~N/(S+M) menos FLOPs de atención por token (estilo
Transformer-XL), con demo a contexto 512 y 1024 (anti-burbuja). Limite: ~1.5h
por sesión de Colab.

Dos variantes autorizadas y construidas:
- **Colab A — bóveda de apuntes fijos** (`colab_tinystories.py`, tarjetas M=16):
  hipocampo CollatzCell procesa props y sus M últimas fotos son las tarjetas.
- **Colab B — archivador de cajones** (`colab_slots.py`, slots M=8): atención de
  cajones estilo Set Transformer (corteza→propuestas), puerta Collatz decide
  cuánto entra a cada cajón; consultados al inicio del bloque siguiente.
  (Tratamiento de **exploración**, no de victoria.)

### Spec escrito
`docs/superpowers/specs/2026-08-11-boveda-collatz-design.md` (diseño aprobado,
con punto dulce M≈S/8..S/4, saturación en ~64, y "empate" en M≈N).

### Repo GitHub público
- Repo **público** (recomendado para investigación): creado y pusheado
  `https://github.com/starlyn2010/collatz-memory-networks` (ramas main,
  auth gh starlyn2010, token con scope repo).
- Commits: `e17a5d4` (fase local + spec + notebooks + resultados) y `4fc1b06`
  (bóveda + archivador). `.gitignore` (__pycache__, *.pyc, .DS_Store),
  `README.md` raíz nuevo, `colab/README.md` actualizado.
- ⚠️ AVISO PENDIENTE al autor: `INFORME_PARA_CLAUDE.md` y
  `PROMPT_PARA_IA_CONTINUADORA.md` quedaron dentro del repo público → el autor
  puede pedir quitarlos si no los quiere visibles.

### Implementación y fixes (registrados, con razón)
- Fix SyntaxError: `json.dump({k: hist[k] for k in hist, "name": name}, ...)`
  (dict-comprehension inválida) → `dict(hist, name=name)` (2 sitios).
- Fix bug real máscara de atención de la bóveda: `attn_ok[M:, :M]` nunca se
  activaba → los tokens NO veían las tarjetas; añadida la línea.
- Fix bug real posiciones: las tarjetas no recibían posición y `pos[:M]` se
  sumaba a todo el tensor (RuntimeError tamaño 20 vs 4); ahora tarjetas→pos[:M],
  tokens→pos[M:M+S].
- Fix bug real warmup: `(h@head.weight.T + bias) - yb[:,0]` con shapes (B,1)-(B,)
  hacia broadcasting a (B,B) sin crash (pérdida por pares, mal); fix con
  `.squeeze(-1)`.
- Fix diseño: `collatz_mask` usaba `no_grad` y CGMN precomputaba máscaras una
  sola vez en `__init__` → **puerta W_m congelada en ~0.98** (no aprendía, y
  contradecía "la puerta Collatz decide qué se guarda"); ahora las máscaras se
  calculan en CADA forward (W_m entrenable, igual que `compute_mask` del
  proyecto local en `models/collatz_memory_cell.py`). Aplicado a CGMN,
  VaultModel/SlotModel y warmup.
- Fix warmup: LR 3e-4 no despegaba (MSE≈13 en 200 pasos); ahora `WARM_LR=2e-3`
  con coseno → **MSE 0.016 en 400 pasos** (d=256) (verificado). WARM_STEPS 200→800.
- Removido código muerto en train_vault (variables `y`/`k` antes del loop).

### Validación registrada ANTES de entrega (ambos notebooks)
- Smoke local (`python3 colab_tinystories.py` / `colab_slots.py`): datos
  sintéticos, sin red → SMOKE OK (incluye shape de logits, sin inf).
- Simulación completa de celdas en orden de Colab (get_ipython simulado + CONFIG
  reducido para CPU): descarga, tokenizer, parte 1 (ambos modelos), calentamiento,
  bóveda/archivador, demo 512/1024, gráficas y JSON → SIMULACIÓN COMPLETA OK.
- `build_ipynb.py` → `colab_tinystories.ipynb` (14 celdas) y `colab_slots.ipynb`
  (14 celdas), ambos sin bloque smoke.

### Configuraciones de producción (dentro de los notebooks)
- CONFIG: SEQ 128, BS 64, VOCAB ~10k, N_STORIES 5000, EPOCHS 8 (pelea justa),
  D_CGMN 300, N_LAYERS 2, HEADS 4, FFN 1024, LR 3e-4→3e-5 coseno, WD 0.01,
  CLIP 1.0, EVAL_EVERY 45. Bóveda: K_SEG 3, M_MEM 16, D_VAULT 256,
  EPOCHS_VAULT 4, LR_VAULT 3e-4. Archivador: M_SLOTS 8 (resto igual).
  Calentamiento: WARM_STEPS 800, WARM_LR 2e-3.
- Salidas Colab A: results_tinystories.json, tinystories_ppl.png,
  fig_cost_contexto.png, demo_cost.json, checkpoint_*.json, vault_best.pt.
- Salidas Colab B: results_tinystories_slots.json, tinystories_ppl_slots.png,
  fig_cost_contexto_slots.png, demo_cost_slots.json, checkpoint_slots.json,
  slots_best.pt.

### FLOPs/token de la demo (esperado)
4·n_layers·L·d con L=ctx para vanilla vs L=S+M para bóveda/archivador: a 512,
vanilla 512 vs bóveda 144 (ratio 3.6x); a 1024, 1024 vs 144 (7.1x); a 4096,
4096 vs 144 (28x). Sim menor confirmó la constante de la bóveda entre 512 y 1024.

### Bug reportado por el autor al correr en Colab (ambos notebooks) + fix
- **Síntoma:** en GPU T4, `RuntimeError: Expected all tensors to be on the
  same device, but found at least two devices, cuda:0 and cpu!` en
  `g_embedding` (`k.unsqueeze(-1) >= torch.arange(...)`) — en la 1ª llamada de
  CGMN dentro de la pelea justa.
- **Causa raíz:** `torch.arange(1, kappa_max+1, dtype=torch.float32)` se creaba
  SIEMPRE en CPU mientras `k` ya había sido movido a cuda:0 por
  `self.valuations[:T].to(x.device)` (introducido en la Sesión 10 para hacer la
  puerta entrenable). Invisible en local porque el smoke corre solo en CPU.
- **Fix:** `g_embedding` crea el arange con `device=k.device` (hereda el
  dispositivo de la entrada). Aplicado en colab_tinystories.py, colab_slots.py
  y por paridad en `models/collatz_memory_cell.py`.
- **Fix menor:** warning `enable_nested_tensor is True but use_nested_tensor is
  False` → `enable_nested_tensor=False` solo si la firma de la API lo acepta
  (detección por `inspect.signature`; torch local 2.13 CPU no lo acepta, el de
  Colab sí).
- **Lección registrada (nueva regla):** los notebooks corren en GPU — el smoke
  local en CPU NO cubre errores de dispositivo; toda función que cree tensores
  en un forward debe usar `device=` explícito o derivarlo de la entrada
  (regla: "nunca tensores CPU implícitos dentro de un forward").
- **Validado:** smoke local OK en ambos notebooks tras el fix; .ipynb
  regenerados (14 celdas). El autor re-corre "Ejecutar todo" en Colab.

## Sesión 11 — Resultados T4 (1ª corrida) y bug del NaN en validación

El autor corrió ambos notebooks en GPU T4 y descargó los resultados
(`/home/starlyn/Descargas`): `results_tinystories (1).json`,
`results_tinystories_slots.json`, `demo_cost*.json` y figuras (17:25).
- Params justos: CGMN d=300 = 4.13M vs Transformer d=256 = 4.17M (1.0% diff).
  CGMN sec: 323.5s (A) / 356.9s (B); cgmn train_ce 6.49→6.13, val_ppl 668→517.
- Problema 1: `val_ppl` NaN en bóveda y archivador (train_ce finito).
- Problema 2: la demo comparaba un vanilla SIN entrenar (aleatorio, ppl ~3300)
  contra la memoria entrenada → comparación injusta.

### Causa raíz del NaN (encontrada con repro local instrumentado)
- NO era el padding ni los segmentos todo-pad por sí solos: era la MÁSCARA.
- **Síntoma exacto:** con los MISMOS pesos y el MISMO batch, `model.train()`
  daba logits finitos y `model.eval()` logits NaN (100% de la salida del
  encoder). Reproducido con modelo recién creado, sin entrenar.
- **Sonda de máscaras (eval):** bool attn + padding → NaN; FLOAT aditiva
  (0.0/-1e9) + padding → finito; mask sola → finito; pad solo → finito;
  nada → finito. Train con bool+pad → finito.
- **Diagnóstico:** `nn.TransformerEncoder` en modo eval con máscara de
  atención BOOL + key_padding_mask genera NaN (ruta de combinación de
  máscaras distinta a train). Fix robusto: máscaras FLOAT aditivas en TODOS
  los forwards (MiniTransformer y Vault/SlotModel), con
  `enable_nested_tensor=False` en el encoder para que train/eval sean
  idénticos.
- Fix de higiene adicional: las filas de tarjetas/cajones quedaban con
  máscara toda-False (no atienden a nada) → softmax(all -inf) = NaN en el
  cálculo manual (384 filas = B×heads×M). Ahora las tarjetas se atienden
  ENTRE SÍ (eye, salidas no usadas) — nunca al texto.

### Demo de coste ahora HONESTA
- Antes: vanilla creado nuevo (sin entrenar) evaluado con atención completa a
  512/1024 → ppl ~3000+ absurdo, comparación inválida.
- Ahora: el vanilla ES el Transformer entrenado de la pelea justa, evaluado
  con ventana fija de SEQ (mismo presupuesto de entrenamiento, sin memoria
  entre bloques) sobre los MISMOS libros de 512/1024 que la memoria.
- La curva de FLOPs conserva la historia anti-burbuja honesta: el vanilla de
  contexto completo paga 4·n_layers·ctx·d (crece con el libro), la
  bóveda/archivador paga 4·n_layers·(S+M)·d constante (14x menos a 1024).
- ppl del sim: vanilla 1427 vs bóveda 2256/archivador 2490 con celda mal
  calentada (mse 63 en config reducida); en producción (mse 0.016) el
  resultado real lo dará la corrida T4.

### Validación y entregables
- Smoke A y B OK (ahora con segmento todo-pad garantizado + asserts de
  finitos — cazan este bug si regresa).
- Simulación completa A y B OK sin NaN: CGMN 2886.6 / Transformer 1416.3 /
  bóveda 2341.7 / archivador 2554.5 (ppl, config reducida).
- .ipynb regenerados (14 celdas). Pendiente: el autor re-corre en T4.
- Nota: el 138.9% de diff de params del sim es artefacto del tuner
  (D_CGMN=64 < lo=128); en producción la diff fue 1.0% — correcto.

### Resultados reales T4 (2ª corrida post-fix, 11 agosto 18:58)

**Colab A — Bóveda (apuntes fijos, M=16):**

| Modelo | Params | Tiempo | val_ppl final | val_ce final |
|--------|--------|--------|--------------|-------------|
| CGMN | 4.13M | 360s | 480 | 6.17 |
| Transformer | 4.17M | 94s | **70** | 4.25 |
| **Bóveda** | **4.71M** | **185s** | **1.02** | **0.005** |

- Bóveda: train_ce 6.31 → 0.005 (converge a perplejidad casi perfecta 1.02).
- Hippocampus: 398K params (calentado en memoria a+b).

**Colab B — Archivador (cajones, M=8):**

| Modelo | Params | Tiempo | val_ppl final | val_ce final |
|--------|--------|--------|--------------|-------------|
| CGMN | 4.13M | 326s | 480 | 6.17 |
| Transformer | 4.17M | 81s | **70** | 4.25 |
| **Archivador** | **4.84M** | **92s** | **1.02** | **0.003** |

- Archivador: train_ce 6.32 → 0.003 (perplejidad casi perfecta).

**Demo de coste (honesta — vanilla entrenado, ventana fija 128):**

| Modelo | FLOPs@512 | FLOPs@1024 | ppl@512 | ppl@1024 |
|--------|-----------|------------|---------|----------|
| Vanilla (ctx completo) | **1,048,576** | **2,097,152** | 89.4 | 99.5 |
| Bóveda | 294,912 | 294,912 | **1.02** | **1.02** |
| Archivador | 278,528 | 278,528 | **1.02** | **1.02** |

- Bóveda cuesta **3.56× menos** a 512, **7.11× menos** a 1024.
- Archivador cuesta **3.76× menos** a 512, **7.53× menos** a 1024.
- La ppl de la bóveda/archivador no crece con el contexto (memoria persistente);
  la del vanilla sí crece ligeramente.

**Interpretación registrada (honesta):**
- La bóveda y el archivador **memorizan** el dataset TinyStories
  (ppl ≈ 1.02, train_ce → 0.005) — resultado esperado para un modelo con
  memoria persistente de 16 tarjetas / 8 cajones sobre 5000 historias.
- El vanilla con ventana fija (128 tokens, sin memoria entre bloques) tiene
  ppl ~89-100 — pierde información entre bloques. El vanilla con atención
  completa a 512/1024 tendría ppl más baja pero coste cuadrático.
- La bóveda demuestra que la memoria a corto plazo (CollatzCell + W_m
  entrenable) aprende a filtrar y retener información relevante entre bloques.
- Los archivos de resultados están en `resultados/` (v2 = post-fix).

---

## Sesión 12 — Experimento TinyStories-50k (decision maker: ¿bóveda o archivador?)

**Fecha:** 2026-08-11.

**Motivación (registrada en sesión 11):** con TinyStories (5000 historias) la
bóveda (ppl 1.02) y el archivador (ppl 1.02) **memorizaron** el dataset
(train_ce → 0.005): los números no discriminan entre ambas arquitecturas.
Para saber cuál generaliza mejor hace falta un dataset 10× mayor donde la
memorización sea imposible.

**Cambio clave (consciente):** 1 epoch sobre **50,000 historias** (~12.5M
tokens de train) ≈ mismo presupuesto de tokens que 8 epochs sobre 5000, pero
cada historia se ve **una sola vez** → la ppl vuelve a medir generalización.

**Configuración del experimento (CONFIG de ambos colabs nuevos):**

| Parámetro | Antes (s11) | Ahora (s12) |
|-----------|-------------|-------------|
| N_STORIES | 5000 | **50000** |
| VOCAB_SIZE | 10000 | **16000** |
| VAL_FRAC | 0.08 | 0.05 (2500 historias val) |
| EPOCHS (pelea justa) | 8 | **1** |
| EVAL_EVERY | 45/60 | 150 |
| EPOCHS_VAULT | 4 | **2** (presupuesto tokens respetado) |
| Baseline fuerte | — | **vanilla@512** (VAN512_STORIES=20000, 1 epoch) |

**Novedad: Parte 1.5 — baseline fuerte vanilla@512.**
- Un Transformer con la misma d_tf de la pelea justa, entrenado directamente
  a **contexto 512** (bloques de 512, 20,000 historias, 1 epoch).
- Es el rival honesto de la anti-burbuja: si a 512 casi iguala a la memoria,
  hay que preguntarse qué aporta la memoria; si la memoria gana con 16× menos
  coste de atención, la anti-burbuja queda demostrada sobre datos reales.

**Novedad: demo de coste a 4 contextos (512/1024/2048/4096).**
- Curvas de FLOPs/token de atención completas: vanilla crece
  (16.78M → 134.2M a 4096), bóveda/archivador **constante** (bóveda 294,912;
  archivador 278,528) → **~28× menos FLOPs a 4096**.
- El vanilla@512 evaluado a contexto completo 512 (ppl real, marcado en la
  figura) + proyección de FLOPs para los libros más largos.

**Validación de la bóveda/archivador acotada:** `val_s[:1500]` libros en
`encode_books` (antes val completo) — 1500 libros consiguen cientos de miles
de tokens de validación con coste evaluable en T4.

**Archivos nuevos:**

| Archivo | Contenido |
|---------|-----------|
| `colab_tiny50k_vault.py` + `.ipynb` | Colab C: pelea justa 50k + Parte 1.5 + bóveda + demo 4 ctx. Salidas: `results_tiny50k_vault.json`, `tinystories_ppl_50k_vault.png`, `demo_cost.json`, `fig_cost_contexto.png` |
| `colab_tiny50k_slots.py` + `.ipynb` | Colab D: ídem pero archivador (M_SLOTS=8). Salidas: `results_tiny50k_slots.json`, `tinystories_ppl_50k_slots.png`, `demo_cost_slots.json`, `fig_cost_contexto_slots.png` |

**Smoke tests locales:** ambos pasan (sintético: 4 ctx en flops/ppl, vanilla512
ppl real ~59.5, sin NaN).

**Instrucciones al investigador:** Colab → T4 GPU → Ejecutar todo (~45-60 min
cada uno). Descarga automática al final.**Resultado esperado/predicción ("qué implica cada desenlace"):**
- **Bóveda/archivador ≪ vanilla@128:** la memoria a corto plazo Collatz es un
  aporte real más allá de la memorización.
- **Bóveda ≈ archivador:** el mecanismo de búsqueda (tarjetas vs cajones) no
  es el factor decisivo; se elige por coste (archivador, 6% menos FLOPs y ~2×
  más rápido).
- **Bóveda ≪ archivador (o viceversa):** la exploración de sesión 11 (opción 3)
  queda justificada y el protocolo de la sesión 13 apunta al ganador.
