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
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# - **Semilla única:** `SEED = 42` en `src/config.py`, usada en la partición,
#   los folds, los muestreos, K-Means, t-SNE, Isolation Forest, los bootstrap y
#   los modelos.
# - **Código ejecutable de principio a fin:** `python ejecutar_libro.py`
#   ejecuta en orden los cuadernos de los capítulos 1–10 y de este apéndice, y
#   construye el libro.
# - **Partición independiente de la versión de scikit-learn:** la partición
#   train/test (cap. 2) y los folds de la validación cruzada espacial (cap. 10)
#   se generan con la función propia `folds_estratificados_por_grupo` de
#   `src/particion.py`. Usa el mismo algoritmo voraz de `StratifiedGroupKFold`,
#   pero con un generador de numpy de semilla fija, porque la asignación de
#   bloques a folds de `StratifiedGroupKFold(shuffle=True)` cambia entre
#   versiones de scikit-learn: en Google Colab, con otra versión, el test quedó
#   con una distribución de estratos muy distinta (ver el docstring del
#   módulo). Así, con la misma semilla, los folds son los mismos en cualquier
#   entorno.
# - **Reglas de limpieza:** los rangos plausibles y las reglas de alcance del
#   cap. 1 (área mínima de 10 m², registros agregados, incoherencias dentro de
#   la fila y techo de 60 para el piso de ubicación) se fijaron con la
#   herramienta `diagnostico_rangos.py`, en la raíz del libro:
#   `python diagnostico_rangos.py RUTA_AL_CSV [datos/procesados/dataset_modelado.parquet]`.
#   Con el parquet del cap. 2 restringe el diagnóstico a train; no modifica el
#   libro, solo escribe su evidencia en `diagnostico_rangos_reporte.txt`. Los
#   umbrales resultantes viven en `src/config.py`.
# - **Dependencias:** `requirements.txt`. Abajo se imprimen las versiones
#   exactas con que se ejecutó esta copia del libro.
# - **Datos:** el CSV se genera con `01_descarga_predios.py` (descarga del
#   servicio ArcGIS con caché). Fecha de descarga: 15 de septiembre de 2026.

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
# ```{admonition} Dónde se ejecutó esta copia del libro
# :class: note
# Según la salida anterior, el libro se ejecutó completo (todos los cuadernos,
# capítulos 1–10 y este apéndice) en un entorno Linux en la nube (kernel
# 6.18.44), con Python 3.11.15,
# scikit-learn 1.8.0, pandas 3.0.2 y numpy 2.4.4, sobre el CSV real
# (`predios_residenciales_barranquilla.csv`) y con `SEED = 42`. Gracias a la
# función propia de partición, ejecutarlo con otra versión de scikit-learn no
# cambia qué bloques quedan en train y en test; otras salidas (p. ej. el
# ajuste de los modelos) sí pueden variar ligeramente entre versiones, por eso
# se documentan aquí.
# ```

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
