# Collatz Memory Networks (CGMN)

Red recurrente con memoria de trabajo modulada por el mecanismo Collatz (3n+c).
Investigación en curso (PyTorch; fase CPU local + notebooks Colab/T4).

## Estado de la investigación

- **Fase local completa** (CPU): Copy Task, Delayed Memory, ablación del gate de
  memoria, coste FLOPs. Resultados medidos e inmutables en `BITACORA.md` y
  `outputs/` (JSONs + fig3..fig9).
- **Hallazgo central**: CGMN es el único recurrente que resuelve Delayed Memory
  (suma a+b con retardo D=50-100; MSE 0.004/0.773 vs ~11.7 del resto). Con el
  gate entrenable, la forma temporal de Collatz no aporta capacidad extra en Copy
  Task (ablación con promedios emparejados) — la memoria proviene de la política
  de retención aprendida.
- **En curso**: bóveda de memoria Collatz para Transformer (memoria hipocampal
  con costo lineal sobre contexto largo) y archivador de cajones (slots).
  Notebooks: `colab_tinystories.py/.ipynb` (pelea justa + bóveda) y
  `colab_slots.py/.ipynb` (slots).

## Estructura

- `models/` — CGMN (celda + stack), generador Collatz, gates, baselines.
- `datasets/` — Copy Task y Delayed Memory (sintéticos, sin descargas).
- `experiments/` — exp1/exp2/exp4/exp5 + utilidades compartidas.
- `outputs/` — resultados y figuras.
- `colab/` + notebooks raíz — fase GPU.
- `docs/superpowers/specs/` — diseños.

## Notas

- Bitácora autorizada: `BITACORA.md` (regla de honestidad de resultados).
- PyTorch puro sobre CPU (máquina local) o T4 (Colab).
