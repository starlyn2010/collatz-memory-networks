#!/usr/bin/env python3
"""Convierte colab_tinystories.py (celdas '# %%') en colab_tinystories.ipynb.
Celdas marcadas con '# %% [no-notebook]' se excluyen (smoke test local)."""
import sys

import nbformat as nbf

SRC = sys.argv[1] if len(sys.argv) > 1 else "colab_tinystories.py"
OUT = sys.argv[2] if len(sys.argv) > 2 else SRC.replace(".py", ".ipynb")

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
cells = []
for block in open(SRC).read().split("# %%")[1:]:
    lines = block.strip("\n").split("\n")
    first = lines[0].strip()
    if first == "[no-notebook]":
        continue
    is_md = first == "[markdown]"
    body = "\n".join(lines[1:] if is_md else lines)
    if not body.strip():
        continue
    if is_md:
        md = "\n".join(
            l[2:] if l.startswith("# ") else l[1:] if l.startswith("#") else l
            for l in body.split("\n")
        )
        cells.append(nbf.v4.new_markdown_cell(md.strip("\n")))
    else:
        cells.append(nbf.v4.new_code_cell(body))
nb.cells = cells
nbf.write(nb, OUT)
print(f"{OUT}: {len(cells)} celdas ({sum(1 for c in cells if c.cell_type == 'markdown')} md)")
