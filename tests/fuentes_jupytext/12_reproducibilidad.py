# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Apéndice B. Reproducibilidad
#
# - **Semilla única:** `SEED = 42` en `src/config.py`, usada en la partición,
#   los folds, los muestreos, K-Means, t-SNE, Isolation Forest, los bootstrap y
#   los modelos.
# - **Código ejecutable de principio a fin:** `python ejecutar_libro.py`
#   ejecuta los capítulos 1–10 en orden y construye el libro.
# - **Dependencias:** `requirements.txt`. Abajo se imprimen las versiones
#   exactas con que se ejecutó esta copia del libro.
# - **Datos:** el CSV se genera con `01_descarga_predios.py` (descarga del
#   servicio ArcGIS con caché). *[COMPLETAR: fecha de descarga y enlace si lo
#   publicas en otro lugar]*.

# %%
# Huella del entorno de ejecución: versión de Python, sistema operativo y versiones
# exactas de las librerías clave. Pequeños cambios de versión (p. ej. en
# scikit-learn o pandas) pueden alterar resultados, así que quedan documentados.
import platform
import sys
# importlib.metadata lee la versión INSTALADA de cada paquete sin importarlo.
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Raíz del libro en el path para importar src, se ejecute desde notebooks/ o la raíz.
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))
from src import config as C

print(f"Python {platform.python_version()} — {platform.system()} {platform.release()}")
# Se usan los nombres de distribución de pip (scikit-learn, jupyter-book), no los de
# importación (sklearn), porque version() busca por nombre de paquete instalado.
for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "folium", "pyarrow", "jupyter-book"]:
    try:
        print(f"{p:<14} {version(p)}")
    # Los opcionales (folium, pyarrow) pueden faltar sin romper el libro.
    except PackageNotFoundError:
        print(f"{p:<14} (no instalado)")
# Semilla global y nombre del CSV efectivamente usado (permite detectar si se corrió
# con el dataset real o con otro archivo).
print(f"\nSEED = {C.SEED} | CSV = {C.ruta_csv().name}")

# %% [markdown]
# ## Registro de decisiones del EDA usado por el modelo

# %%
# Tabla completa del registro de decisiones (decisiones_eda.json): cada parámetro
# que el modelo toma del EDA (buffer, variables, transformaciones...) junto con el
# hallazgo que lo motivó. Es la trazabilidad EDA -> modelado en un solo lugar.
import pandas as pd

dec = C.leer_decisiones()
pd.set_option("display.max_colwidth", 120)  # motivos largos visibles sin truncar
# str(valor): algunos valores son listas o diccionarios; como texto caben en una celda.
# .T: una fila por decisión, con columnas "valor" y "motivo".
pd.DataFrame({k: {"valor": str(v["valor"]), "motivo": v["motivo"]} for k, v in dec.items()}).T
