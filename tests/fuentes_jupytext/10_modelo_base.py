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
# # 10. Modelo base (sección 3)
#
# | Requisito del rubric | Cómo se cumple aquí |
# |---|---|
# | Regresión logística | `LogisticRegression` multinomial con regularización L2 |
# | Definición del objetivo | `estrato_num` (1–6), clasificación multiclase ordinal (cap. 4) |
# | Línea base trivial | `DummyClassifier` (mayoritaria, estratificada, uniforme) + línea base espacial "clase más frecuente por zona" |
# | Partición acorde a la estructura, antes del preprocesamiento | bloques espaciales estratificados (cap. 2) + buffer (cap. 8) |
# | Pipeline y CV del mismo tipo | `Pipeline` + `GridSearchCV` con folds por bloques + buffer dentro de train |
# | Métricas | accuracy, precision, recall, F1, AUC, matriz de confusión, ROC, PR y calibración (hay desbalance) |
# | Intervalos de confianza | bootstrap **por bloques espaciales** |
# | Comparación con la línea base | tabla + diferencias pareadas con IC |
# | Diagnóstico de residuos | Moran's I de los residuos (ordinales) en test |
# | Curva de aprendizaje | con la misma validación espacial |
# | Interpretación | coeficientes y métricas |
#
# Todas las decisiones de preprocesamiento se leen de
# `datos/procesados/decisiones_eda.json`, escrito por los capítulos 3–9: así
# cada paso del Pipeline queda rastreado al hallazgo del EDA que lo motivó.

# %% tags=["hide-input"]
# Preparación del entorno: importaciones, estilo de figuras y lectura del registro
# de decisiones del EDA (decisiones_eda.json). Mostrar esa tabla al inicio deja
# explícito qué hallazgo del EDA justifica cada paso del Pipeline (trazabilidad).
import json
import sys
import time
import warnings
from pathlib import Path

# La raíz del libro depende de si el notebook se ejecuta desde notebooks/ o desde
# la raíz; se agrega a sys.path para poder importar el paquete local src.
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))
# Se silencian avisos (p. ej. de convergencia o de categorías infrecuentes) para que
# el libro sea legible; los resultados no dependen de ellos.
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, roc_curve, auc)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline

from src import config as C
from src import espacial as S
from src import graficos as G
from src import modelo as M
from src import particion as P

# Estilo común del libro (fuentes grandes, paleta viridis por estrato).
G.estilo()
# Si se corre con el CSV sintético de pruebas, imprime una advertencia visible
# para que esos números nunca se confundan con resultados reales.
C.aviso_sintetico()
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
# Diccionario {clave: {"valor", "motivo"}} escrito por los capítulos 3–9 del EDA.
dec = C.leer_decisiones()
# Tabla decisión -> valor -> motivo (str() para que listas y dicts se muestren bien).
pd.DataFrame({k: {"valor": str(v["valor"]), "motivo": v["motivo"]} for k, v in dec.items()}).T

# %% [markdown]
# ## 10.1 Datos, variables y buffer

# %%
# Carga el dataset con la partición train/test por bloques espaciales hecha ANTES de
# cualquier preprocesamiento (cap. 2), aplica el buffer entre train y test y define
# los dos conjuntos de variables a comparar. Todos los parámetros vienen del EDA.
datos = P.cargar_particion()
# Variables admitidas tras la auditoría de fuga (cap. 9): numéricas, categóricas y
# espaciales por separado, para poder armar los conjuntos A y B.
vm = dec["variables_modelo"]["valor"]
# Numéricas muy asimétricas (cap. 5) que recibirán log1p en el Pipeline.
num_log = dec["num_log"]["valor"]
# Distancia mínima train-test, derivada del alcance del correlograma (cap. 8).
buffer_km = dec["buffer_km"]["valor"]
# Cuantiles de winsorización: se APRENDEN en train dentro del Pipeline, no aquí.
q_inf, q_sup = dec["winsorizacion_cuantiles"]["valor"]
# Frecuencia mínima para que una categoría tenga su propia columna one-hot.
min_frec = dec["onehot_min_frecuencia"]["valor"]

train_total = datos[datos.particion == "train"].reset_index(drop=True)
test = datos[datos.particion == "test"].reset_index(drop=True)
# mascara_buffer (KDTree) marca True las unidades de train a >= buffer_km de TODA
# unidad de test. Las más cercanas se descartan: por la autocorrelación espacial
# del estrato serían "casi copias" de test y inflarían el desempeño medido.
mantener = P.mascara_buffer(train_total[["x_km", "y_km"]].to_numpy(), test[["x_km", "y_km"]].to_numpy(), buffer_km)
train = train_total[mantener].reset_index(drop=True)
print(f"Train: {len(train_total):,} -> {len(train):,} tras excluir {(~mantener).sum():,} unidades a menos de "
      f"{buffer_km} km de test | Test: {len(test):,} unidades en {test.bloque.nunique()} bloques")

# Dos conjuntos anidados: A (solo atributos físicos de la vivienda) y B (A + x, y y
# distancia al centro). Comparar A y B mide cuánto aporta la ubicación por sí sola.
conjuntos = {
    "A_fisicas": {"num": vm["numericas"], "cat": vm["categoricas"]},
    "B_fisicas+ubicacion": {"num": vm["numericas"] + vm["espaciales"], "cat": vm["categoricas"]},
}
for k, v in conjuntos.items():
    print(f"{k}: {len(v['num'])} numéricas + {len(v['cat'])} categóricas")
# Objetivo: estrato_num (1–6) como arreglo numpy para indexarlo por posición.
y_tr, y_te = train[C.OBJETIVO].to_numpy(), test[C.OBJETIVO].to_numpy()


def pipeline_logistica(conj, **kw):
    # Pipeline preprocesamiento + logística. Al ir TODO dentro del Pipeline, en cada
    # fold la winsorización, la imputación, el escalado y el one-hot se ajustan solo
    # con el train de ese fold: no hay fuga de información de validación ni de test.
    nums, cats = conj["num"], conj["cat"]
    # construir_preprocesador: numéricas en num_log -> winsorizar, imputar mediana
    # (+ indicador de faltante), log1p y estandarizar; resto de numéricas igual pero
    # sin log1p; categóricas -> imputar "faltante" y one-hot con categorías raras
    # agrupadas en "infrequent".
    prep = M.construir_preprocesador([c for c in nums if c in num_log], [c for c in nums if c not in num_log],
                                     cats, q_inf=q_inf, q_sup=q_sup, min_frecuencia=min_frec)
    # Con 6 clases y el solver por defecto (lbfgs) la logística es multinomial
    # (softmax) con penalización L2; max_iter=1000 evita cortes por no convergencia.
    # **kw permite fijar C o class_weight desde fuera (aquí los fija GridSearchCV).
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000, **kw))])


# %% [markdown]
# ## 10.2 Validación cruzada espacial y búsqueda de hiperparámetros
#
# Folds: `StratifiedGroupKFold` por bloque dentro de train, excluyendo del
# entrenamiento de cada fold las unidades a menos de `buffer_km` del fold de
# validación. Se busca `C` (inverso de la regularización) y si conviene
# `class_weight="balanced"` por el desbalance, optimizando **F1 macro**.
#
# **Regla de "una desviación estándar" (1-SE).** No se elige a ciegas la
# combinación con el F1 medio más alto: se toma la **más regularizada** (menor
# `C`) cuyo F1 medio esté a menos de un error estándar
# ($\text{SE} = \text{desv. entre folds}/\sqrt{k}$) de la mejor, y entre
# ellas la de mayor F1 (Breiman et al., 1984; Hastie, Tibshirani y Friedman,
# 2009, §7.10). Motivo: en la CV espacial la superficie de F1 es muy plana y
# la desviación entre folds (~0.09) es mucho mayor que las diferencias entre
# combinaciones. En la segunda ejecución con datos reales, un cambio mínimo en
# los datos (tratar el piso 99 como faltante) hizo que la "mejor" combinación
# del conjunto B saltara de C = 0.01 balanceado (F1 0.315) a C = 10 sin pesos
# (0.317): una diferencia de 0.002, ruido puro, que llevaba a un modelo casi
# sin regularización, con coeficientes enormes y peor calibrado. La regla 1-SE
# hace la elección estable y prefiere el modelo más simple cuando los datos no
# distinguen entre ellos.

# %%
# Búsqueda de hiperparámetros con CV ESPACIAL (mismo tipo de validación que la
# partición train/test) y selección final con la regla 1-SE. Se hace para A y para B;
# test no interviene en ningún momento de esta celda.
# Folds dentro de train: StratifiedGroupKFold por bloque de 2 km (nunca parte un
# bloque y equilibra las proporciones de estrato) y, en cada fold, se quitan del
# entrenamiento las unidades a < buffer_km de la validación. Se guardan como lista
# de índices para reutilizar EXACTAMENTE los mismos folds en todo el capítulo.
folds = P.folds_espaciales_con_buffer(train, buffer_km)
# C = inverso de la fuerza de regularización (C pequeño = coeficientes más
# encogidos). class_weight="balanced" pondera cada clase por 1/frecuencia para
# contrarrestar el desbalance; None deja todas las observaciones con igual peso.
grid = {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}


def regla_1se(cv_results):
    """Índice de la combinación más regularizada (menor C) a < 1 SE de la mejor."""
    # GridSearchCV acepta un callable en refit: recibe cv_results_ y devuelve el
    # índice de la combinación con la que se reentrena en todo train (best_index_).
    media = np.asarray(cv_results["mean_test_score"], float)
    # Error estándar de la MEDIA de k folds: SE = desv. entre folds / sqrt(k). Mide
    # cuánto podría moverse el F1 medio solo por qué zonas cayeron en cada fold.
    se = np.asarray(cv_results["std_test_score"], float) / np.sqrt(len(folds))
    # nanargmax: una combinación que falle en algún fold queda con NaN y se ignora.
    i_mejor = int(np.nanargmax(media))
    # Candidatas: combinaciones estadísticamente indistinguibles de la mejor
    # (su F1 medio está dentro de un SE del máximo).
    candidatos = np.flatnonzero(media >= media[i_mejor] - se[i_mejor])
    c_vals = np.asarray(cv_results["param_clf__C"], float)
    # Entre ellas se prefiere el modelo más simple = el de menor C (más regularizado).
    c_min = c_vals[candidatos].min()
    finalistas = [i for i in candidatos if c_vals[i] == c_min]
    # Desempate por class_weight: con ese C, la de mayor F1 medio.
    return int(finalistas[int(np.argmax(media[finalistas]))])


busquedas, f1_cv_elegido = {}, {}
for nombre, conj in conjuntos.items():
    t0 = time.time()
    # scoring="f1_macro": promedia el F1 de las 6 clases sin ponderar por tamaño, de
    # modo que los estratos minoritarios (5 y 6) pesan igual que los mayoritarios.
    # refit=regla_1se en lugar de refit=True (que tomaría el máximo a ciegas).
    # return_train_score=True guarda el F1 en train para diagnosticar sobreajuste.
    gs = GridSearchCV(pipeline_logistica(conj), grid, cv=folds, scoring="f1_macro", n_jobs=-1, refit=regla_1se,
                      return_train_score=True)
    gs.fit(train[conj["num"] + conj["cat"]], y_tr)
    busquedas[nombre] = gs
    r = gs.cv_results_
    # F1 medio en CV de la combinación ELEGIDA por la regla 1-SE (no del máximo):
    # es el que se usa después para escoger el modelo principal entre A y B.
    f1_cv_elegido[nombre] = float(r["mean_test_score"][gs.best_index_])
    # Se reporta también el máximo de la rejilla y su SE para mostrar que la
    # diferencia entre ambos está dentro del ruido entre folds.
    i_max = int(np.nanargmax(r["mean_test_score"]))
    print(f"{nombre}: elegido (regla 1-SE) {gs.best_params_} | F1 macro CV = {f1_cv_elegido[nombre]:.4f} "
          f"| máximo de la rejilla: {r['params'][i_max]} con {r['mean_test_score'][i_max]:.4f} "
          f"(SE = {r['std_test_score'][i_max] / np.sqrt(len(folds)):.4f}) ({time.time() - t0:.0f}s)")

# %% tags=["hide-input"]
# Mapa de calor del F1 macro medio en CV para cada combinación (class_weight x C),
# uno por conjunto. Sirve para ver qué tan plana es la superficie de F1: si lo es,
# justifica usar la regla 1-SE en vez de elegir el máximo.
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
for ax, (nombre, gs) in zip(axes, busquedas.items()):
    r = pd.DataFrame(gs.cv_results_)
    # dropna=False conserva la fila class_weight=None (pandas la trataría como NaN
    # y la descartaría del pivote).
    piv = r.pivot_table(index="param_clf__class_weight", columns="param_clf__C", values="mean_test_score",
                        dropna=False)
    # Etiqueta legible "None" para la fila sin pesos.
    piv.index = piv.index.map(lambda v: "None" if pd.isna(v) else str(v))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title(f"{nombre}: F1 macro (CV espacial)")
    ax.set_xlabel("C")
    ax.set_ylabel("class_weight")
fig.tight_layout()
# Se guarda en figuras/ para el informe, además de mostrarse en el libro.
G.guardar(fig, "10_grid")
plt.show()

# %% [markdown]
# ### Línea base espacial: tamaño de la zona
#
# La "moda por zona" depende del tamaño de la zona. Para no fijarlo a dedo, se
# elige con la misma validación cruzada espacial (así la línea base también
# está afinada y la comparación con la logística es justa).

# %%
# Afinado de la línea base espacial "moda por zona": se prueba el tamaño de celda
# con los mismos folds espaciales y la misma métrica que la logística, para que la
# comparación sea justa (ambos modelos tienen su hiperparámetro elegido en CV).
zonas = {}
for tam_z in [2.0, 4.0, 6.0, 8.0]:
    # BaselineModaZona divide el plano en celdas de tam_z km y, en fit, guarda la
    # distribución de estratos de cada celda en el train del fold; predice la clase
    # más frecuente de la celda (o la distribución global si la celda no existe en
    # train). Solo usa x_km e y_km: es "la ubicación sin atributos físicos".
    r = cross_validate(M.BaselineModaZona(tam_km=tam_z), train[["x_km", "y_km"]], y_tr, cv=folds,
                       scoring="f1_macro")
    zonas[tam_z] = float(r["test_score"].mean())
# Se elige el tamaño con mayor F1 macro medio en CV.
tam_zona = max(zonas, key=zonas.get)
print("F1 macro CV por tamaño de zona:", {k: round(v, 4) for k, v in zonas.items()}, "-> elegido:", tam_zona, "km")

# %% [markdown]
# ### Comparación en validación cruzada con las líneas base

# %%
# Comparación en CV espacial de las logísticas contra líneas base triviales (Dummy)
# y la espacial (moda por zona). Un modelo solo aporta si supera claramente a estas
# referencias; la desviación entre folds indica cuánto dependen de la zona evaluada.
# Cada entrada es (estimador, columnas que usa). Las Dummy ignoran X, pero sklearn
# exige al menos una columna: se les pasa x_km solo como relleno.
modelos_cv = {
    # Predice siempre la clase más frecuente: piso absoluto de accuracy.
    "Dummy (mayoritaria)": (DummyClassifier(strategy="most_frequent"), ["x_km"]),
    # Predice al azar con las proporciones de train: piso para F1 macro.
    "Dummy (estratificada)": (DummyClassifier(strategy="stratified", random_state=C.SEED), ["x_km"]),
    # Predice al azar con probabilidad 1/6 por clase.
    "Dummy (uniforme)": (DummyClassifier(strategy="uniform", random_state=C.SEED), ["x_km"]),
    "Moda por zona": (M.BaselineModaZona(tam_km=tam_zona), ["x_km", "y_km"]),
}
# best_estimator_ es el Pipeline reentrenado con la combinación elegida por la regla
# 1-SE; cross_validate lo clona, así que en cada fold se reajusta desde cero.
for nombre, gs in busquedas.items():
    modelos_cv[f"Logística {nombre}"] = (gs.best_estimator_, conjuntos[nombre]["num"] + conjuntos[nombre]["cat"])
filas = []
for nombre, (est, cols) in modelos_cv.items():
    # Mismos folds para todos los modelos: las diferencias no se deben a particiones
    # distintas. Además de F1 macro se registran accuracy (fácil de comunicar, pero
    # engañosa con desbalance) y balanced accuracy (recall medio por clase).
    r = cross_validate(est, train[cols], y_tr, cv=folds, n_jobs=-1,
                       scoring={"f1_macro": "f1_macro", "accuracy": "accuracy",
                                "balanced_accuracy": "balanced_accuracy"})
    filas.append({"modelo": nombre, **{f"{m} (media)": r[f"test_{m}"].mean() for m in ["f1_macro", "accuracy",
                                                                                        "balanced_accuracy"]},
                  # la desviación entre folds cuantifica la variabilidad por zona
                  "F1 macro (desv. entre folds)": r["test_f1_macro"].std()})
tabla_cv = pd.DataFrame(filas).set_index("modelo")
tabla_cv

# %% [markdown]
# ```{note}
# Los puntajes de las logísticas en esta tabla vienen de los mismos folds con
# que se eligieron sus hiperparámetros, así que son levemente optimistas frente
# a las Dummy (que no tienen hiperparámetros). La comparación definitiva es la
# del conjunto de prueba (10.4).
# ```
#
# ```{admonition} Resultado de la validación cruzada espacial
# :class: note
# - El buffer de 1 km excluye entre el **27 % y el 49 %** del entrenamiento de
#   cada fold (37.7 % en promedio); en el modelo final, train pasa de 219 628 a
#   **138 779** filas (80 849 excluidas por cercanía a test). Estos
#   porcentajes difieren un poco de los del cap. 8.4 (35.7 %) porque aquí
#   los folds se construyen sobre el train ya depurado de la zona de test.
# - Ambas logísticas eligen **C = 0.01** (la regularización más fuerte de la
#   rejilla) y **`class_weight="balanced"`**. Que gane la regularización más
#   fuerte es coherente con el cambio de dominio espacial: coeficientes más
#   pequeños generalizan mejor a zonas no vistas.
# - F1 macro en CV: Dummy mayoritaria 0.066, estratificada 0.142, moda por
#   zona 0.205, **logística A 0.303, logística B 0.314**. La desviación entre
#   folds (0.08–0.10) es del mismo orden que las diferencias entre modelos: el
#   desempeño depende mucho de la zona de validación.
# ```

# %% [markdown]
# ### ¿Cuán optimista sería una validación aleatoria?
#
# Mismo modelo principal (el de mejor F1 macro en la CV espacial), mismos
# datos, pero con `StratifiedKFold` aleatorio (sin bloques ni buffer).

# %%
# Cuantifica el "optimismo espacial": se evalúa el mismo Pipeline con folds
# aleatorios (StratifiedKFold) y se compara con la CV por bloques + buffer. La
# diferencia es cuánto se sobreestimaría el desempeño si se ignorara el espacio.
# el modelo principal se elige por F1 macro en la CV espacial, NUNCA mirando test
mejor_nombre = max(busquedas, key=lambda k: f1_cv_elegido[k])
# El otro conjunto se conserva para la comparación pareada en test (10.4).
otro_nombre = [k for k in busquedas if k != mejor_nombre][0]
print(f"Modelo principal (mejor F1 macro en CV espacial): {mejor_nombre}")
mejor = busquedas[mejor_nombre].best_estimator_
# Columnas del modelo principal (se llama cols_b porque en los datos reales gana B).
cols_b = conjuntos[mejor_nombre]["num"] + conjuntos[mejor_nombre]["cat"]
# Folds aleatorios: unidades vecinas (incluso del mismo edificio) quedan repartidas
# entre entrenamiento y validación, y el modelo "reconoce" la zona en vez de
# generalizar a zonas nuevas. Se usan la misma semilla y el mismo número de folds.
r_alea = cross_validate(mejor, train[cols_b], y_tr, scoring="f1_macro", n_jobs=-1,
                        cv=StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED))
optimismo = pd.DataFrame({
    "F1 macro CV": [r_alea["test_score"].mean(), f1_cv_elegido[mejor_nombre]],
    # para la CV espacial se toma la desviación entre folds de la combinación elegida
    "desv.": [r_alea["test_score"].std(),
              busquedas[mejor_nombre].cv_results_["std_test_score"][busquedas[mejor_nombre].best_index_]],
}, index=["aleatoria (StratifiedKFold)", "espacial (bloques + buffer)"])
# Diferencia positiva en la fila aleatoria = optimismo de ignorar la estructura espacial.
optimismo["diferencia vs espacial"] = optimismo["F1 macro CV"] - optimismo.loc["espacial (bloques + buffer)", "F1 macro CV"]
optimismo

# %% [markdown]
# ### Sensibilidad al buffer
#
# El tamaño de bloque y el buffer son "hiperparámetros de la evaluación"
# (notas de clase 9.1.11): el número exacto cambia según cómo se construyan.
# Se reporta el F1 macro de la CV del modelo principal con varios buffers.

# %%
# Sensibilidad de la CV espacial al tamaño del buffer: el buffer es un
# "hiperparámetro de la evaluación", así que se muestra cómo cambia el F1 (y cuánto
# train se pierde) entre no usar buffer y usar el alcance del correlograma.
# Alcance de la autocorrelación del estrato (cap. 8); si no está registrado, se usa
# el buffer elegido para no romper la celda.
alcance = dec.get("alcance_correlograma_km", {}).get("valor", buffer_km)
sens = []
# El set elimina duplicados (p. ej. si el buffer ya es 0.5 km) y el alcance se
# acota a 3 km porque buffers mayores dejan casi sin datos de entrenamiento.
for b in sorted({0.0, 0.5, float(buffer_km), float(min(alcance, 3.0))}):
    # Mismos bloques y semilla que `folds`: solo cambia el buffer aplicado.
    fb = P.folds_espaciales_con_buffer(train, b, verbose=False)
    r = cross_validate(mejor, train[cols_b], y_tr, cv=fb, scoring="f1_macro", n_jobs=-1)
    # % excluido por fold = 1 - (train que queda tras el buffer) / (train posible,
    # es decir, todo lo que no es validación); se promedia entre folds.
    sens.append({"buffer_km": b, "F1 macro CV": r["test_score"].mean(), "desv.": r["test_score"].std(),
                 "% train excluido": 100 * np.mean([1 - len(tr) / (len(train) - len(va)) for tr, va in fb])})
pd.DataFrame(sens)

# %% [markdown]
# ```{admonition} Cómo leer la sensibilidad
# :class: note
# A medida que crece el buffer, el F1 de validación **baja** por dos razones
# que se suman: (i) se elimina la "ayuda" de vecinos casi idénticos (el
# optimismo espacial que se quiere quitar) y (ii) se entrena con menos datos y
# se obliga al modelo a **extrapolar** a zonas más lejanas.
#
# Con los datos reales: **sin buffer 0.380 → 0.345 con 0.5 km → 0.314 con
# 1 km (elegido) → 0.230 con el alcance completo (2.9 km)**, que deja fuera el
# 82 % de train. Ese último valor es un escenario pesimista (el modelo casi no
# tiene datos), no una estimación realista. La caída de 0.066 entre 0 y 1 km
# mide cuánto del desempeño depende de tener vecinos del mismo barrio en
# entrenamiento. Comparado con la validación **aleatoria** (0.616, sección
# anterior), el optimismo total de ignorar el espacio es de **0.30 puntos de
# F1**: la validación aleatoria casi duplicaría el desempeño reportado.
# ```

# %% [markdown]
# ## 10.3 Curva de aprendizaje (validación espacial)

# %%
# Curva de aprendizaje con los MISMOS folds espaciales: F1 en entrenamiento y en
# validación al crecer el tamaño de train. Responde si más datos ayudarían (brecha
# que se cierra) o si el límite es el modelo o las variables (validación plana).
# train_sizes son fracciones del tamaño de entrenamiento de los folds (ya depurado
# por el buffer); `tam` devuelve los tamaños absolutos correspondientes.
# shuffle=True baraja antes de tomar cada subconjunto; sin barajar se tomarían las
# primeras filas, que pueden estar ordenadas por zona y sesgarían la curva.
tam, sc_tr, sc_va = learning_curve(mejor, train[cols_b], y_tr, cv=folds, scoring="f1_macro",
                                   train_sizes=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0], n_jobs=-1, shuffle=True,
                                   random_state=C.SEED)
fig, ax = plt.subplots(figsize=(10, 5))
# sc_tr y sc_va tienen forma (tamaños, folds): media por fila y banda de ±1 desv.
# entre folds para mostrar la variabilidad debida a la zona de validación.
for s, et, col in [(sc_tr, "entrenamiento", "#4c72b0"), (sc_va, "validación (espacial)", "#c44e52")]:
    ax.plot(tam, s.mean(1), "o-", color=col, label=et)
    ax.fill_between(tam, s.mean(1) - s.std(1), s.mean(1) + s.std(1), color=col, alpha=0.15)
# Escala log porque los tamaños van de ~5 % a 100 % (más de un orden de magnitud).
ax.set_xscale("log")
ax.set_xlabel("unidades de entrenamiento")
ax.set_ylabel("F1 macro")
ax.set_title(f"Curva de aprendizaje — logística {mejor_nombre}")
ax.legend()
G.guardar(fig, "10_curva_aprendizaje")
plt.show()
# Tabla con la brecha train − validación: grande y persistente sugiere sobreajuste o
# cambio de dominio; si ambas se aplanan, más filas no mejorarán el modelo.
pd.DataFrame({"n": tam, "F1 train": sc_tr.mean(1), "F1 validación": sc_va.mean(1),
              "brecha": sc_tr.mean(1) - sc_va.mean(1)})

# %% [markdown]
# ```{admonition} Cómo leer la curva
# :class: note
# - Si las curvas de entrenamiento y validación **convergen** y la validación
#   se aplana, más datos no ayudarán: el límite es el **sesgo del modelo**
#   (lineal) o la información de las variables, no el tamaño de muestra.
# - Una brecha grande y persistente indicaría **sobreajuste**; una brecha
#   moderada que no cierra puede venir del **cambio de dominio espacial**:
#   validar en barrios no vistos es más difícil que entrenar.
#
# **Con los datos reales:** la validación está **plana** entre 0.32 y 0.34
# desde 2 700 hasta 54 000 unidades: multiplicar por 20 los datos no mejora
# nada. La brecha (0.22–0.27) no cierra. Con C = 0.01 (regularización fuerte)
# y n ≫ p no es sobreajuste clásico: es **cambio de dominio**. Lo que el
# modelo aprende en unas zonas no se transfiere a otras (cap. 8.5:
# heterogeneidad espacial). **Conclusión:** para mejorar hace falta un modelo
# más flexible o mejores variables (del entorno), no más filas.
# ```

# %% [markdown]
# ## 10.4 Evaluación final en el conjunto de prueba (una sola vez)

# %%
# Evaluación final: cada modelo (con hiperparámetros ya fijados en CV) se entrena en
# TODO train (tras el buffer) y se evalúa UNA sola vez en test. Usar test una única
# vez, sin volver a ajustar nada después, es lo que hace que su estimación no esté
# sesgada por decisiones tomadas mirándolo.
cols_por_modelo = {k: v[1] for k, v in modelos_cv.items()}
finales = {}
for nombre, (est, cols) in modelos_cv.items():
    # fit devuelve el propio estimador; todo el preprocesamiento del Pipeline se
    # aprende aquí solo con train.
    finales[nombre] = est.fit(train[cols], y_tr)
pred, proba = {}, {}
for nombre, est in finales.items():
    X = test[cols_por_modelo[nombre]]
    # Probabilidades (para AUC, log loss, Brier, calibración) y clase predicha.
    proba[nombre] = est.predict_proba(X)
    pred[nombre] = est.predict(X)
# M.metricas calcula: accuracy, balanced accuracy, precision/recall/F1 macro, F1
# ponderado, métricas ORDINALES (MAE, accuracy ±1 estrato, kappa cuadrático, que
# penalizan menos equivocarse por un estrato) y, con probabilidades, AUC OvR macro
# (sobre las clases presentes en test), log loss y Brier multiclase.
resultados = pd.DataFrame({n: M.metricas(y_te, pred[n], proba[n]) for n in finales}).T
resultados

# %%
# Intervalos de confianza del 95 % para las métricas del modelo principal en test,
# por bootstrap de BLOQUES espaciales. Cuantifican cuánto variaría el resultado si
# hubieran caído otras zonas de la ciudad en test.
PRINCIPAL = f"Logística {mejor_nombre}"
# bootstrap_por_bloques remuestrea con reemplazo los bloques de test (no las filas):
# las unidades de un mismo bloque están autocorrelacionadas y no son observaciones
# independientes, así que un bootstrap por filas daría IC demasiado estrechos.
# En cada una de las B=500 réplicas recalcula accuracy, F1 macro, MAE ordinal y
# kappa cuadrático; el IC son los percentiles 2.5 y 97.5 (método percentil).
ic = M.bootstrap_por_bloques(y_te, pred[PRINCIPAL], proba[PRINCIPAL], test["bloque"].to_numpy(), B=500)
# Se antepone el valor puntual observado en test para leerlo junto a su intervalo.
ic.insert(0, "valor en test", [resultados.loc[PRINCIPAL, k] for k in ic.index])
print(f"IC 95 % por bootstrap de {test.bloque.nunique()} bloques espaciales (500 réplicas) — {PRINCIPAL}")
# La media ± desviación de la CV espacial se muestra como segunda referencia: con
# pocos bloques en test, una sola partición puede ser más fácil o más difícil.
print(f"Referencia complementaria (CV espacial en train): F1 macro {tabla_cv.loc[PRINCIPAL, 'f1_macro (media)']:.3f} "
      f"± {tabla_cv.loc[PRINCIPAL, 'F1 macro (desv. entre folds)']:.3f} entre folds")
ic

# %% [markdown]
# ```{admonition} Sobre la incertidumbre
# :class: important
# El bootstrap remuestrea **bloques** completos porque las filas de un mismo
# bloque no son independientes. Con solo **7 bloques de prueba**, los
# intervalos son muy anchos: F1 macro **0.435 con IC 95 % [0.14, 0.46]**,
# accuracy 0.568 [0.32, 0.79], kappa cuadrático 0.82 [0.17, 0.92]. Es la forma
# honesta de decir que el resultado depende mucho de qué zonas de la ciudad
# quedaron en test.
#
# El intervalo del F1 es **asimétrico** (el valor observado está casi en el
# límite superior): cuando una réplica bootstrap no incluye el bloque que
# concentra el estrato 6 o el 5, el F1 de esa clase cae a 0 y arrastra el
# promedio macro. Además, el F1 en test (0.435) es claramente mayor que el de
# la CV espacial (0.314 ± 0.079): el test resultó una zona "más fácil" (casi
# sin estrato 6, cap. 2). **La cifra más representativa del desempeño en zonas
# nuevas es la de la CV espacial**, y la de test es una realización
# favorable.
# ```

# %%
# Diferencias de F1 macro en test entre el modelo principal y cada referencia, con
# IC por bootstrap PAREADO por bloques. Comparar dos IC separados no basta: lo que
# importa es si la DIFERENCIA es distinta de 0 al evaluar ambos en las mismas zonas.
# zero_division=0: si una réplica no contiene alguna clase o no la predice, su F1
# cuenta como 0 en lugar de lanzar avisos.
f1m = lambda a, b: f1_score(a, b, average="macro", zero_division=0)
comparaciones = {}
for base in ["Dummy (mayoritaria)", "Dummy (estratificada)", "Moda por zona", f"Logística {otro_nombre}"]:
    # bootstrap_diferencia: en cada réplica elige los MISMOS bloques para ambos
    # modelos y calcula F1(principal) − F1(base). Al compartir las zonas, la
    # variabilidad común se cancela y el IC de la diferencia es más preciso.
    # También devuelve un p bootstrap bilateral (proporción de réplicas del otro lado
    # de 0), acotado inferiormente por 1/B.
    comparaciones[f"{PRINCIPAL} − {base}"] = M.bootstrap_diferencia(y_te, pred[PRINCIPAL], pred[base],
                                                                     test["bloque"].to_numpy(), f1m, B=500)
comp = pd.DataFrame(comparaciones).T
# Si el IC 95 % no contiene 0, la diferencia es significativa al 5 %.
comp["¿IC excluye 0?"] = (comp["IC95_inf"] > 0) | (comp["IC95_sup"] < 0)
print("Diferencia de F1 macro en test (bootstrap pareado por bloques):")
comp

# %% [markdown]
# ```{admonition} Nota crítica del rubric
# :class: warning
# Si el accuracy o el F1 superan 0.80–0.90 hay que sospechar: verificar fuga
# (cap. 9), comparar con la línea base y confirmar que la validación respeta
# la estructura espacial. La celda siguiente hace esa verificación automática.
# ```

# %%
# Alerta automática del rubric: un accuracy >= 0.80 en este problema sería señal de
# fuga (proxies del estrato o vecinos de test en train), no de un buen modelo. Se
# reporta además la ganancia sobre la Dummy mayoritaria, que es el piso de referencia.
acc = resultados.loc[PRINCIPAL, "accuracy"]
if acc >= 0.8:
    print(f"ALERTA: accuracy = {acc:.3f} ≥ 0.80. Revisar fuga espacial, variables proxy y la partición.")
else:
    print(f"accuracy = {acc:.3f} (< 0.80): no hay señal de desempeño sospechosamente alto.")
print(f"Ganancia sobre la mayoritaria: {acc - resultados.loc['Dummy (mayoritaria)', 'accuracy']:+.3f} en accuracy, "
      f"{resultados.loc[PRINCIPAL, 'f1_macro'] - resultados.loc['Dummy (mayoritaria)', 'f1_macro']:+.3f} en F1 macro")

# %% [markdown]
# ### Matriz de confusión y reporte por clase

# %% tags=["hide-input"]
# Matriz de confusión del modelo principal en test, en conteos y normalizada por
# fila. La normalizada muestra el recall de cada estrato y hacia qué estratos se
# desplazan los errores (si caen en clases adyacentes, el error es "ordinal leve").
etiquetas = [C.NOMBRES_CLASES[c] for c in C.CLASES]
fig, axes = plt.subplots(1, 2, figsize=(19, 7.5))
# normalize="true" divide cada fila por su total real: la diagonal es el recall.
for ax, norm, t in [(axes[0], None, "conteos"), (axes[1], "true", "normalizada por fila (recall)")]:
    # labels=C.CLASES fija las 6 filas/columnas aunque alguna clase falte en test.
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred[PRINCIPAL], labels=C.CLASES, normalize=norm),
                           display_labels=etiquetas).plot(ax=ax, cmap="Blues", colorbar=False,
                                                          values_format=".2f" if norm else ",d")
    ax.set_title(f"Matriz de confusión ({t})")
    ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
G.guardar(fig, "10_confusion")
plt.show()

# %%
# Precisión, recall, F1 y soporte por estrato en test. Complementa el F1 macro: dice
# qué clases concretas arrastran el promedio (p. ej. estratos intermedios o raros).
# zero_division=0 evita avisos si una clase nunca se predice o no aparece en test.
print(classification_report(y_te, pred[PRINCIPAL], labels=C.CLASES, target_names=etiquetas, digits=3,
                            zero_division=0))

# %% [markdown]
# ### Curvas ROC y precisión-recall (one-vs-rest)

# %% tags=["hide-input"]
# Curvas ROC y precisión-recall one-vs-rest: cada estrato contra el resto, usando la
# probabilidad predicha de esa clase. Evalúan la capacidad de ORDENAR (ranking) de
# las probabilidades, independiente del umbral implícito de predict (argmax).
fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
# La columna k de predict_proba corresponde a C.CLASES[k] porque las 6 clases están
# en train y sklearn las ordena igual (classes_ = [1, ..., 6]).
for k, c in enumerate(C.CLASES):
    # Etiqueta binaria "es estrato c" frente a "no lo es".
    yb = (y_te == c).astype(int)
    # Sin positivos en test la curva no está definida: se omite esa clase.
    if yb.sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(yb, proba[PRINCIPAL][:, k])
    axes[0].plot(fpr, tpr, color=G.PALETA_ESTRATO[c], label=f"{C.NOMBRES_CLASES[c]} (AUC {auc(fpr, tpr):.3f})")
    pr_, rc_, _ = precision_recall_curve(yb, proba[PRINCIPAL][:, k])
    # AP (average precision) resume la curva PR; su referencia "al azar" es la
    # prevalencia de la clase (base), no 0.5 como en ROC. Por eso se imprimen ambas.
    axes[1].plot(rc_, pr_, color=G.PALETA_ESTRATO[c],
                 label=f"{C.NOMBRES_CLASES[c]} (AP {average_precision_score(yb, proba[PRINCIPAL][:, k]):.3f}, "
                       f"base {yb.mean():.2f})")
# Diagonal = clasificador aleatorio (AUC 0.5).
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("tasa de falsos positivos")
axes[0].set_ylabel("tasa de verdaderos positivos (recall)")
axes[0].set_title("ROC one-vs-rest")
axes[0].legend(fontsize=10)
axes[1].set_xlabel("recall")
axes[1].set_ylabel("precisión")
axes[1].set_title("Precisión-recall one-vs-rest")
axes[1].legend(fontsize=10)
fig.tight_layout()
G.guardar(fig, "10_roc_pr")
plt.show()

# %% [markdown]
# ```{admonition} Cómo leer ROC vs. PR con desbalance
# :class: note
# El AUC-ROC de una clase rara puede verse alto aunque el modelo la prediga
# mal, porque la tasa de falsos positivos se diluye entre muchos negativos. La
# curva **precisión-recall** es más exigente: su referencia es la proporción
# de la clase ("base"), no 0.5. Para los estratos 5 y 6 (minoritarios) la PR
# es la curva a mirar.
# ```

# %% [markdown]
# ### Calibración
#
# ¿Las probabilidades significan lo que dicen? Curvas de fiabilidad
# one-vs-rest por clase (10 bins por cuantiles) y error de calibración
# esperado (ECE) de la clase predicha.

# %% tags=["hide-input"]
# Calibración: ¿cuando el modelo dice "70 % de ser estrato c", lo es ~70 % de las
# veces? Curvas de fiabilidad one-vs-rest por estrato y ECE de la clase predicha.
# Importa porque class_weight="balanced" altera las probabilidades a propósito.
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
# Un panel por estrato (2 x 3 = 6 clases).
for ax, (k, c) in zip(axes.ravel(), enumerate(C.CLASES)):
    yb = (y_te == c).astype(int)
    if yb.sum() == 0:
        ax.axis("off")
        continue
    # strategy="quantile": bins con el mismo número de unidades, así ningún punto de
    # la curva se apoya en muy pocas observaciones (las probabilidades de las clases
    # raras se concentran cerca de 0 y bins uniformes quedarían casi vacíos).
    # Devuelve (frecuencia observada, probabilidad media predicha) por bin.
    fr, mp = calibration_curve(yb, proba[PRINCIPAL][:, k], n_bins=10, strategy="quantile")
    # Diagonal = calibración perfecta; por debajo = el modelo sobreestima la clase.
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(mp, fr, "o-", color=G.PALETA_ESTRATO[c])
    ax.set_title(C.NOMBRES_CLASES[c])
    ax.set_xlabel("probabilidad predicha")
    ax.set_ylabel("frecuencia observada")
fig.suptitle(f"Curvas de calibración (one-vs-rest) — {PRINCIPAL}")
fig.tight_layout()
G.guardar(fig, "10_calibracion")
plt.show()
# ECE (expected calibration error) de la clase predicha: se agrupan las unidades por
# su confianza (probabilidad máxima) y en cada grupo se compara la tasa de acierto
# con la confianza media. ECE = sum_b (n_b / n) * |acierto_b − confianza_b|.
conf = proba[PRINCIPAL].max(1)
acierto = (pred[PRINCIPAL] == y_te).astype(float)
# Bordes en los deciles de la confianza (bins de igual tamaño); np.unique elimina
# bordes repetidos si muchas unidades comparten el mismo valor.
bins = np.unique(np.quantile(conf, np.linspace(0, 1, 11)))
# Solo los bordes interiores: digitize asigna el índice de bin 0..(nbins−1), con el
# mínimo y el máximo incluidos en el primer y el último bin.
ids = np.digitize(conf, bins[1:-1])
# (ids == b).mean() es el peso n_b / n de cada bin.
ece = sum(np.abs(acierto[ids == b].mean() - conf[ids == b].mean()) * (ids == b).mean()
          for b in np.unique(ids))
# El Brier multiclase (media de sum_k (p_k − 1[y=k])^2) combina calibración y
# discriminación; se imprime junto al ECE como segunda medida de las probabilidades.
print(f"ECE (clase predicha, 10 bins): {ece:.4f} | Brier multiclase: {resultados.loc[PRINCIPAL, 'brier_multiclase']:.4f}")

# %% [markdown]
# ## 10.5 Diagnóstico de residuos: ¿queda estructura espacial?
#
# Para una variable ordinal se define el residuo como
# $r_i = y_i - \hat{E}[y_i] = y_i - \sum_k k\,\hat p_{ik}$ (estrato observado
# menos estrato esperado según las probabilidades del modelo). Si los residuos
# están autocorrelacionados en el espacio, el modelo deja sin capturar
# información estructural (el entorno). Se calcula Moran's I de los residuos a
# nivel de edificio en test.

# %%
# Diagnóstico de residuos: Moran's I global de los residuos ordinales en test, para
# las dos logísticas. Si I > 0 y es significativo, los errores se agrupan en el
# espacio y al modelo le falta información del entorno (supuesto de independencia
# de los errores violado).
filas = []
res_mapa = None
for nombre in [f"Logística {k}" for k in busquedas]:
    # Estrato esperado E[y] = sum_k k * p_k: producto de la matriz de probabilidades
    # (n x 6) por el vector de clases [1..6]. Usa toda la distribución predicha, no
    # solo el argmax, así que el residuo es continuo y respeta el orden del estrato.
    esperado = proba[nombre] @ np.asarray(C.CLASES, float)
    # Residuo = y − E[y]: positivo si el modelo subestima, negativo si sobreestima.
    t = test.assign(residuo=y_te - esperado)
    # Una fila por edificio (residuo promedio): las unidades de un mismo edificio
    # comparten coordenada y casi siempre estrato, e inflarían artificialmente I.
    e = S.deduplicar_por_edificio(t[["npn_edificio", "x_km", "y_km", "residuo"]])
    # Matriz de pesos W de k vecinos más cercanos, estandarizada por filas: kNN y no
    # banda fija porque la densidad de edificios es muy desigual en la ciudad.
    Wt, _, _ = S.pesos_knn(e[["x_km", "y_km"]].to_numpy(), k=C.K_VECINOS)
    # I = (n / S0) * z'Wz / z'z con z centrado; p-valor por permutación (499 veces se
    # barajan los residuos entre edificios para obtener la distribución bajo H0 de
    # ausencia de autocorrelación), sin suponer normalidad.
    r = S.moran_global(e["residuo"].to_numpy(), Wt, n_perm=499, seed=C.SEED)
    # El residuo medio (a nivel de unidad) indica sesgo global: < 0 = sobreestima.
    filas.append({"modelo": nombre, "Moran I residuos": r["I"], "p (perm.)": r["p_perm"], "edificios": r["n"],
                  "residuo medio": t["residuo"].mean()})
    # Se guardan los residuos por edificio del modelo principal para el mapa.
    if nombre == PRINCIPAL:
        res_mapa = e
pd.DataFrame(filas).set_index("modelo")

# %% tags=["hide-input"]
# Mapa de residuos del modelo principal: hexágonos con el residuo medio de los
# edificios que contienen. Muestra DÓNDE se agrupan los errores que Moran detecta.
fig, ax = plt.subplots(figsize=(9, 8))
# Escala divergente simétrica (±1.5 estratos) centrada en 0; mincnt=3 oculta
# hexágonos con menos de 3 edificios, cuyo promedio sería demasiado ruidoso.
hb = ax.hexbin(res_mapa["x_km"], res_mapa["y_km"], C=res_mapa["residuo"], reduce_C_function=np.mean,
               gridsize=35, cmap="RdBu_r", vmin=-1.5, vmax=1.5, mincnt=3)
plt.colorbar(hb, label="residuo medio (estrato observado − esperado)")
G.ejes_mapa(ax, f"Residuos de {PRINCIPAL} en test\n(rojo: subestima, azul: sobreestima)")
G.guardar(fig, "10_residuos_mapa")
plt.show()

# %% [markdown]
# ```{admonition} Interpretación
# :class: note
# - **Los errores se agrupan por zonas**: Moran's I de los residuos = **0.67**
#   en el modelo B (p = 0.002) y 0.35 en el A. Hay barrios enteros que el
#   modelo sobreestima o subestima: el estrato depende del **entorno** (vías,
#   espacio público, servicios, historia del barrio) que las variables de la
#   vivienda no describen.
# - Que B tenga **más** autocorrelación residual que A no significa que la
#   ubicación empeore el modelo. A predice unidad por unidad con variables
#   físicas "ruidosas", y ese ruido diluye el patrón espacial de sus
#   residuos. B suaviza sus predicciones con la tendencia norte-sur, y lo que
#   queda como residuo son justamente las **desviaciones locales** respecto a
#   esa tendencia, que son muy espaciales. La tendencia lineal capta la gran
#   escala, pero no los barrios.
# - **Sesgo hacia arriba:** el residuo medio es **−0.29 estratos** (el modelo
#   sobreestima en promedio). En el mapa, la franja oriental del test (cerca
#   del centro y del río) queda sistemáticamente sobreestimada, y el borde
#   norte del bloque central, subestimado. Dos causas probables: el
#   `class_weight="balanced"` sube artificialmente la probabilidad de las
#   clases altas (minoritarias en train), y el test casi no tiene estrato 6.
# - Mejoras (entregable 2): rezagos espaciales de variables **físicas**
#   (nunca del objetivo), interacciones o modelos no lineales (árboles,
#   boosting) y variables del entorno.
# ```

# %% [markdown]
# ## 10.6 Interpretación de coeficientes
#
# En la logística multinomial hay un vector de coeficientes por clase. Como
# las numéricas están estandarizadas, cada coeficiente es el cambio en el
# log-odds por **una desviación estándar**. Para una lectura directa se
# muestra $\beta_{6} - \beta_{1}$: el cambio en el log-odds de ser estrato 6
# frente a estrato 1 por cada desviación estándar (o por pertenecer a la
# categoría, en las one-hot).
#
# Cautelas de lectura: (i) en las variables con log1p, "una DE" es una DE del
# valor winsorizado y transformado; (ii) `x_km`, `y_km` y `dist_centro_km`
# están relacionadas (la distancia es función de x e y), así que sus
# coeficientes no se interpretan por separado sino como una tendencia espacial
# conjunta; (iii) con variables categóricas redundantes (uso / condición /
# tipo de vivienda) el peso se reparte entre ellas.

# %%
# Coeficientes del modelo principal (entrenado en todo train): mapa de calor de la
# matriz clase x variable y tabla de β6 − β1, el contraste entre los extremos de la
# escala, que resume qué empuja hacia estratos altos o bajos.
clf = finales[PRINCIPAL].named_steps["clf"]
# Nombres de las columnas DESPUÉS del preprocesamiento (incluye indicadores de
# faltante y columnas one-hot), en el mismo orden que las columnas de coef_.
nombres = finales[PRINCIPAL].named_steps["prep"].get_feature_names_out()
# Se quita el prefijo del transformador ("num_log__", "num_lin__", "cat__").
nombres = [n.split("__", 1)[1] for n in nombres]
# coef_ tiene forma (6 clases, p variables): en la multinomial hay un vector de
# coeficientes por clase (identificable solo en diferencias entre clases).
coef = pd.DataFrame(clf.coef_, index=[C.NOMBRES_CLASES[c] for c in clf.classes_], columns=nombres)
# Ancho proporcional al número de variables, con tope para que la figura quepa.
fig, ax = plt.subplots(figsize=(min(24, 0.55 * len(nombres) + 4), 5.5))
sns.heatmap(coef, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "coeficiente (log-odds por DE)"})
ax.set_title(f"Coeficientes de la logística multinomial ({PRINCIPAL})")
ax.tick_params(axis="x", rotation=75, labelsize=10)
G.guardar(fig, "10_coeficientes")
plt.show()
# En la multinomial solo las DIFERENCIAS entre vectores de clase tienen
# interpretación: β6 − β1 es el cambio en log(P(6)/P(1)) por una DE de la variable
# (o por pertenecer a la categoría). Última fila = estrato 6, primera = estrato 1.
lectura = pd.DataFrame({"β6 − β1": coef.iloc[-1] - coef.iloc[0]})
# Odds ratio = exp(β6 − β1). Se recorta a [−50, 50] antes de exp para evitar
# desbordamiento numérico (exp(710) ya es inf) si algún coeficiente fuera enorme;
# con ese rango el OR sigue siendo claramente "extremo". Formato de 3 cifras.
lectura["odds ratio (6 vs 1)"] = np.exp(lectura["β6 − β1"].clip(-50, 50)).map(lambda v: f"{v:.3g}")
# Las 15 variables con mayor |β6 − β1|, sin importar el signo.
lectura.sort_values("β6 − β1", key=np.abs, ascending=False).head(15)

# %% [markdown]
# Importancia por permutación (caída de F1 macro al desordenar cada variable
# original, 3 repeticiones). Para no tocar el conjunto de prueba se calcula en
# el primer fold de la CV espacial: se entrena en su parte de entrenamiento y
# se permuta en su validación (submuestra de 20 000).

# %%
# Importancia por permutación sobre las variables ORIGINALES (antes del one-hot):
# cuánto cae el F1 macro al desordenar cada una. Se calcula en un fold de la CV
# espacial y no en test, para no volver a usar test después de la evaluación final.
from sklearn.base import clone
# Primer fold: índices de entrenamiento (ya sin el buffer) y de validación.
itr, iva = folds[0]
# clone crea una copia sin ajustar con los mismos hiperparámetros; se entrena solo
# con el train del fold para que la validación sea de zonas no vistas.
modelo_fold = clone(mejor).fit(train.iloc[itr][cols_b], y_tr[itr])
# Submuestra de hasta 20 000 unidades de validación (sin reemplazo, semilla fija)
# para acotar el costo: cada variable se re-predice n_repeats veces.
sub = np.random.default_rng(C.SEED).choice(iva, min(20_000, len(iva)), replace=False)
# Se permuta cada columna de entrada del Pipeline (una categórica se permuta
# completa, no por columna one-hot). Una caída negativa significa que la variable
# perjudica al generalizar a la zona de validación.
pi = permutation_importance(modelo_fold, train.iloc[sub][cols_b], y_tr[sub], scoring="f1_macro",
                            n_repeats=3, random_state=C.SEED, n_jobs=-1)
# Media y desviación de la caída entre las 3 repeticiones, ordenadas de mayor a menor.
imp = pd.DataFrame({"caída F1 macro": pi.importances_mean, "desv.": pi.importances_std},
                   index=cols_b).sort_values("caída F1 macro", ascending=False)
imp

# %% [markdown]
# ## 10.7 Resumen de resultados

# %%
# Resumen reproducible: se guardan en resultados_modelo.json las tablas clave (CV,
# test, IC, comparaciones, optimismo e hiperparámetros) para que el informe y los
# capítulos siguientes lean los números de un archivo y no se copien a mano.
salida = {
    "tabla_cv": tabla_cv.round(4).to_dict(),
    "resultados_test": resultados.round(4).to_dict(),
    "ic_modelo_principal": ic.round(4).to_dict(),
    # astype(str): la columna booleana "¿IC excluye 0?" (numpy.bool_) no es
    # serializable directamente por json.
    "comparaciones_f1": comp.round(4).astype(str).to_dict(),
    "optimismo_validacion_aleatoria": optimismo.round(4).to_dict(),
    # str() para que None y los floats de la rejilla se guarden sin problemas.
    "mejores_hiperparametros": {k: {kk: str(vv) for kk, vv in v.best_params_.items()} for k, v in busquedas.items()},
    "n_train_final": int(len(train)), "n_test": int(len(test)), "buffer_km": buffer_km,
}
(C.CARPETA_PROCESADOS / "resultados_modelo.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
# Tabla final compacta: métricas de clasificación, de ranking (AUC) y ordinales de
# todos los modelos en test, para leer de un vistazo la comparación con las bases.
resultados.loc[:, ["accuracy", "balanced_accuracy", "f1_macro", "AUC_OvR_macro", "MAE_ordinal", "accuracy_±1",
                   "kappa_cuadrático"]]

# %% [markdown]
# ```{admonition} Interpretación de los resultados
# :class: important
# **1. ¿Supera a las líneas base?** Sí a las triviales, no de forma
# concluyente a la espacial. La logística B (físicas + ubicación, C = 0.01,
# `class_weight="balanced"`) logra en test **F1 macro 0.435**, balanced
# accuracy 0.48, AUC OvR 0.88 y accuracy 0.568, frente a 0.075 de F1 de la
# Dummy mayoritaria (Δ = +0.36, IC 95 % [0.10, 0.39]) y 0.165 de la
# estratificada (Δ = +0.27, IC [0.01, 0.31]). Pero la **moda por zona de
# 4 km** (una regla sin variables físicas: "predice el estrato más común de tu
# zona en train") obtiene F1 0.346, **mayor accuracy (0.607)**, menor MAE
# ordinal (0.46 frente a 0.50) y el mismo kappa cuadrático (0.826 frente a
# 0.823). La diferencia de F1 con B (+0.09) tiene **IC [−0.06, 0.13]: no es
# significativa**. La logística es mejor en las clases minoritarias (F1 macro,
# balanced accuracy, AUC, log loss), pero en error ordinal típico una regla
# puramente geográfica la iguala.
#
# **2. ¿Cuánto aporta la ubicación?** En la CV espacial, casi nada: F1 0.303
# (A, solo físicas) frente a 0.314 (B). En test, B supera a A en 0.09 de F1
# (IC [−0.07, 0.14], no significativo), en accuracy (0.57 frente a 0.41) y,
# sobre todo, en error ordinal (MAE 0.50 frente a 0.83; kappa 0.82 frente a
# 0.65). Las coordenadas como tendencia lineal ayudan a no equivocarse "por
# mucho", pero no resuelven las clases intermedias.
#
# **3. ¿Dónde se equivoca?** Los errores son casi siempre **entre estratos
# adyacentes** (accuracy ±1 = 0.937) y con sesgo hacia arriba. El estrato 1
# se reconoce bien (F1 0.89). El **estrato 2 prácticamente desaparece**
# (recall 0.03): el 85 % de las unidades de estrato 2 se predicen como 3. Es
# una clase "intermedia" sin rasgos físicos propios, y el modelo lineal la
# absorbe en sus vecinas. Lo mismo, en menor medida, ocurre con el 4 (38 % se
# predice como 5) y el 6 (recall 0.14; la mitad se predice como 5).
#
# **4. ¿Confianza en las probabilidades?** Baja: ECE = 0.157 y Brier 0.59. El
# reponderado `balanced` mejora el recall de las clases raras a costa de
# distorsionar las probabilidades; si se van a usar como probabilidades, hay
# que recalibrarlas (entregable 2).
#
# **5. ¿Queda estructura?** Sí: Moran de residuos 0.67, errores agrupados por
# barrios (sección 10.5).
#
# **6. ¿Más datos ayudarían?** No: la curva de aprendizaje está plana
# (0.32–0.34). El límite es el modelo lineal y la falta de variables del
# entorno.
#
# **7. Coeficientes e importancia.** Hacia estratos altos empujan, sobre
# todo, estar **más al norte** (`y_km`: β6−β1 = 6.8 por DE, la señal
# dominante), el **área construida** (3.1), ser apartamento en PH (1.3) y
# tener más **baños** (1.3). Con el área fija, **más habitaciones** empuja
# hacia estratos bajos (−1.3: habitaciones más pequeñas), igual que ser
# `Informal` (−1.2) o vivienda de hasta 3 pisos (−1.1). La importancia por
# permutación (en un fold de validación) confirma que lo que generaliza a
# zonas nuevas es la ubicación (`y_km`, `x_km`) y la tipología (`uso`,
# `condicion_predio`). Terreno, baños, tipo de vivienda y (levemente) piso de
# ubicación tienen importancia **negativa**: su relación con el estrato cambia entre zonas (cap. 8.5), y
# lo que el modelo aprende de ellas en unas zonas perjudica en otras. Es una
# pista concreta para el entregable 2.
#
# **8. ¿Desempeño sospechoso?** No. El accuracy (0.568) está lejos del umbral
# de alerta de 0.80, la auditoría de fuga no encontró proxies (cap. 9) y la
# validación respeta edificios y bloques. La cifra más honesta del desempeño
# esperado en zonas nuevas es la de la CV espacial, **F1 macro ≈ 0.31**.
# ```
