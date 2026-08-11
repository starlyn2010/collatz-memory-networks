# Diseño — Bóveda Collatz (memoria hipocampal) + Archivador de cajones

**Fecha:** 2026-08-11 · **Proyecto:** CGMN (Collatz Memory Networks) · **Estado:** aprobado por el autor

## Contexto y motivación

El Colab TinyStories previo mostró: CGMN ppl 622 vs Transformer 139 (params 3.39M vs 4.17M, 240 pasos). El Transformer domina el modelado de lenguaje por paralelismo; CGMN no compite ahí. Su fortaleza verificada es la **memoria de trabajo** (Delayed Memory D=100: único recurrente que resuelve).

Este diseño ataca el punto débil del Transformer: **contexto largo con costo cuadrático**. Se construye una "bóveda de memoria" Collatz anexa al Transformer, para que en lugar de releer todo el pasado (O(N²)) consulte una memoria compacta (O(N·(S+M))).

## Objetivos

1. **Pelea justa**: re-correr TinyStories con parámetros igualados (±1%) y 10 epochs, para dar a CGMN una oportunidad limpia de estabilizarse.
2. **Bóveda coláctica (Colab A)**: Transformer con memoria hipocampal Collatz-KV; demostrar «perplejidad similar al vanilla con ~4x menos cómputo» en contexto de 512 y 1024 tokens.
3. **Archivador de cajones (Colab B)**: Opción 3 exploratoria (memoria por slots con puerta Collatz) — misma corpus y métricas.
4. **Métricas comunes**: params, FLOPs/token (conteo de atención), tiempo real de entrenamiento, ppl final + curvas.

## Arquitectura — Bóveda (Colab A)

### Concepto «corteza + hipocampo»
- **La corteza (Transformer) lee el texto** en bloques de S=128 tokens (atención self dentro del bloque). Es la única red que lee el libro.
- **El hipocampo (CGMN de 1 capa, d=256)** NO lee el texto: recibe **vectores de escritura** propuestos por el Transformer en cada posición (proyección lineal de su hidden state).
- **La puerta Collatz decide si el apunte se guarda**: la celda CGMN aplica `h_new = (1 - z_eff)·h + z_eff·propuesta` con `z_eff = z·m_t` y `m_t = sigmoid(W_m·g(k_t))` — la misma mecánica verificada localmente (W_m init bias=4.0, k_t = valuaciones por índice, `E[2^-k]=1/3`).
- **Lectura**: al procesar el bloque i+1, el Transformer atiende a `[128 tokens propios + M=16 apuntes]`, donde los apuntes son los últimos M estados del hipocampo tras el bloque i. El estado del hipocampo **persiste entre bloques** (nunca se reinicia).
- La puerta sigue un **calentamiento**: CGMN entrena SOLA ~200 pasos en una tarea de memoria (a+b con retardo, sintética) antes de unirse al Transformer.

### Costos (por token, por capa)
- Vanilla: N slots · costo 2N·d
- Bóveda: (S+M) = 144 slots · costo 2·(128+M)·d → ahorro = N/144
- Ratio demostrado a N=512: 3.6x; N=1024: 7.1x; N=4096: 28x
- **Fuera de utilidad**: M > ~64 (fotos redundantes de un estado d-dimensional). **Fuera de ahorro**: M ≈ N (empata con vanilla). Punto dulce: M = S/8..S/4 (16-32 para S=128).

### Demostración de 2 longitudes
Evaluación de val ppl + FLOPs/token sobre historias de 512 y 1024 tokens, tres modelos:
1. Vanilla Transformer 512/1024 (ventana completa) — referencia cara.
2. Bóveda 4×128 / 8×128 con M=16 — esperado: ppl similar, costo ~4x/7x menor.
3. (Repite curvas de costo por token para mostrar la «anti-burbuja».)

## Arquitectura — Archivador de cajones (Colab B, exploratorio)

- M=8 slots (vectores d=256). Por bloque:
  - **Lectura**: atención cruzada segmento→slots (query = tokens, KV = slots).
  - **Escritura**: atención cruzada slots→segmento (query = slots, KV = tokens) produce el delta; la **puerta Collatz** escala cuánto se escribe: `slot += m_t·α·tanh(delta)`; m_t = sigmoid(W_m·g(k_t)) (fijada o entrenable).
  - Slots persisten entre bloques.
- Riesgo alto de colapso de entrenamiento → se reporta como exploración, con números reales (ppl, params, FLOPs, tiempo).

## Dataset y entrenamiento

- Corpus: TinyStories (streaming, 5000 historias, vocab word-level 10k, min_freq 2, bloques ≥8 tokens) — la infraestructura validada del Colab anterior.
- Pelea justa: CGMN d=300 (params ≈ 4.12M) vs Transformer d auto-ajustado para ±1% (≈4.16M); AdamW 3e-4 coseno→3e-5, clip 1.0, bs 64, 10 epochs, eval cada 45 pasos.
- Bóveda: CGMN hipocampo d=256, 1 capa, M=16, calentamiento 200 pasos de Delayed Memory sintética (D=20) antes del entrenamiento conjunto.
- Archivador: M=8 slots, misma pipeline.

## Notebooks y entrega

- **Colab A**: `colab_tinystories.py` ampliado (pelea justa + bóveda + demo 2 longitudes) + regenerado a `.ipynb` con `build_ipynb.py`.
- **Colab B**: nuevo `colab_slots.py` + `.ipynb`.
- Ambos: keep-alive JS, auto-descarga con reintentos+caché, checkpoints JSON por eval, descarga automática de results+figs, disyuntor de tiempo de sesión (~80 min) con guardado antes de cortar.
- **Validación obligatoria**: smoke local (sintético) + simulación completa de celdas en orden de Colab, con descarga simulada — antes de entregar. Bugs conocidos a vigilar: init del embedding con head atado (N(0,0.02)), API torch 2.13 renombrada (`src_mask`, `key_padding_mask`), carreras de JSON.

## Honestidad (restricciones innegociables)

1. Si la bóveda NO iguala al vanilla a 4x menos: se reporta igual (la curva de costo es el dato real).
2. CGMN en lenguaje general seguirá detrás del Transformer — no se maquilla.
3. Archivador puede no entrenar: es exploración documentada.
4. FLOPs = conteo de atención aproximado; el tiempo real medido es el proxy.
5. Números finales inmutables como en sesiones anteriores.