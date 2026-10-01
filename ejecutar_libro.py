# -*- coding: utf-8 -*-
"""
Ejecuta los capítulos del libro EN ORDEN (guardando las salidas dentro de
cada .ipynb) y luego construye el Jupyter Book en _build/html/index.html.

    python ejecutar_libro.py              # todo: capítulos 01-12 + build
    python ejecutar_libro.py 05 06        # solo esos capítulos (sin build)
    python ejecutar_libro.py --solo-build # solo construir el HTML

El orden importa: 01 (limpieza) -> 02 (partición, guarda el dataset de
modelado) -> 03..09 (EDA en train, registran decisiones) -> 10 (modelo, lee
las decisiones). Si cambias algo en un capítulo del EDA, vuelve a correr
desde ese capítulo hasta el 10.
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parent
NB = RAIZ / "notebooks"


def ejecutar(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    t0 = time.time()
    print(f"-> {nb_path.name} ...", flush=True)
    cliente = NotebookClient(nb, timeout=7200, kernel_name="python3",
                             resources={"metadata": {"path": str(NB)}})
    try:
        cliente.execute()
    except Exception as e:
        nbformat.write(nb, nb_path)  # guarda hasta donde llegó para ver el error
        print(f"\nERROR en {nb_path.name}. Ábrelo en Jupyter para ver la celda que falló.\n{e}")
        sys.exit(1)
    nbformat.write(nb, nb_path)
    print(f"   OK ({time.time() - t0:.0f} s)", flush=True)


def construir():
    print("\nConstruyendo el libro...")
    # equivalente a escribir en la terminal:  jupyter-book build .
    from jupyter_book.cli.main import main as jb
    try:
        jb(["build", str(RAIZ)], standalone_mode=False)
    except SystemExit:
        pass
    print(f"\nListo: {RAIZ / '_build' / 'html' / 'index.html'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--solo-build"]:
        construir()
        sys.exit(0)
    cuadernos = sorted(NB.glob("[0-9][0-9]_*.ipynb"))
    if args:
        cuadernos = [c for c in cuadernos if any(c.name.startswith(a) for a in args)]
    for c in cuadernos:
        ejecutar(c)
    if not args:
        construir()
