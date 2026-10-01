# -*- coding: utf-8 -*-
"""
Convierte los notebooks fuente (.py, formato 'percent' de jupytext) a .ipynb
y los ejecuta de principio a fin. Uso de prueba (desarrollo):

    CATASTRO_CSV=tests/predios_sinteticos.csv CATASTRO_SINTETICO=1 \
        python tests/ejecutar_notebooks.py [01 02 ...]

Los .ipynb que quedan en notebooks/ van SIN salidas; las copias ejecutadas se
guardan en tests/salidas/ solo para revisar errores y tiempos.
"""

import os
import sys
import time
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
NB = RAIZ / "notebooks"
SALIDAS = RAIZ / "tests" / "salidas"
SALIDAS.mkdir(exist_ok=True)

filtro = sys.argv[1:]
fuentes = sorted(NB.glob("[0-9][0-9]_*.py"))
if filtro:
    fuentes = [f for f in fuentes if any(f.name.startswith(x) for x in filtro)]

for src in fuentes:
    nb = jupytext.read(src)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    destino = src.with_suffix(".ipynb")
    nbformat.write(nb, destino)  # versión limpia (sin salidas) para el libro
    if os.environ.get("SOLO_CONVERTIR") == "1":
        print(f"convertido {destino.name}")
        continue
    t0 = time.time()
    cliente = NotebookClient(nb, timeout=3600, kernel_name="python3",
                             resources={"metadata": {"path": str(NB)}})
    try:
        cliente.execute()
        estado = "OK"
    except Exception as e:
        estado = f"ERROR: {type(e).__name__}: {str(e)[-1500:]}"
    nbformat.write(nb, SALIDAS / destino.name)
    print(f"{src.name}: {estado}  ({time.time() - t0:.0f}s)", flush=True)
