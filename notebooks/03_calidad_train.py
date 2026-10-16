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
# # 3. Calidad de datos en entrenamiento: faltantes, outliers y sesgo
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# ```{admonition} Qué hace este capítulo
# :class: tip
# **Pregunta que responde:** ¿cómo tratar los datos faltantes y los valores
# extremos, y a quién deja fuera el modelo?
#
# **Datos:** las **218 870 viviendas de entrenamiento** (capítulo 2). A
# partir de aquí y hasta el capítulo 9 no se mira el conjunto de prueba.
#
# **Conceptos nuevos:** mecanismos de faltantes (MCAR, MAR, MNAR), diferencia
# de medias estandarizada (SMD), AUC, regla de Tukey y winsorización.
#
# **Sale de aquí:** cómo imputar los faltantes, cómo limitar los extremos y
# la medida del sesgo de cobertura.
# ```
#
# El capítulo 1 ya contó los faltantes y anuló los valores imposibles con
# reglas fijas. Lo que falta de la parte de **calidad de datos** de la
# Sección 1 de las instrucciones del proyecto son las preguntas que **sí**
# llevan a decisiones de modelado (con qué valor rellenar un faltante, hasta
# dónde recortar un extremo). Esas decisiones aprenden de los datos, y por eso
# se toman **solo con entrenamiento**:
#
# - Valores faltantes: **patrón** (matriz de nulos) y **mecanismo**
#   (MCAR / MAR / MNAR) con evidencia (3.1).
# - Outliers univariados: detección (regla de Tukey) y tratamiento (3.2).
# - Sesgo de muestreo y representatividad: quién queda fuera del modelo (3.3).

# %% tags=["hide-input"]
# Configuración y carga de SOLO entrenamiento. Desde este capítulo todo lo que
# pueda convertirse en una decisión de modelado (imputación, outliers) se calcula
# sin mirar test, para evitar fuga de información (orden de trabajo de las instrucciones del proyecto).
import sys
import warnings
from pathlib import Path

# Raíz del libro en sys.path para importar src/ desde notebooks/ o desde la raíz.
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
# Regresión logística + CV: se usan para intentar PREDECIR el indicador de nulo
# (diagnóstico MCAR vs. MAR), no para predecir el estrato.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C       # listas de variables, SEED, guardar_decision
from src import estadistica as E  # V de Cramér y demás pruebas del EDA
from src import graficos as G
from src import particion as P    # cargar_particion lee el dataset del cap. 2

G.estilo()
C.aviso_sintetico()
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

# solo="train" filtra particion == "train": el conjunto de prueba no se abre aquí.
train = P.cargar_particion(solo="train")
print(f"Filas de entrenamiento: {len(train):,}")

# %% [markdown]
# ## 3.1 Valores faltantes
#
# ### Patrón

# %%
# Patrón de faltantes en train: conteo y % de nulos por variable, más una columna
# que indica si el nulo proviene sobre todo de una regla del cap. 1 (valor
# imposible: año 1512, piso 99...; o incoherencia dentro de la fila) o si ya venía
# vacío en el origen.
vars_nulos = C.NUMERICAS + C.CATEGORICAS
tabla_nulos = pd.DataFrame({
    "n nulos": train[vars_nulos].isna().sum(),
    "% nulos": 100 * train[vars_nulos].isna().mean(),
}).sort_values("% nulos", ascending=False)
# Criterio: si al menos la mitad de los nulos de la variable tiene su bandera
# flag_imposible_* (rango de dominio, paso 4) o flag_incoherente_* (coherencia
# dentro de la fila, paso 5) activa, ese es el origen principal. 'antiguedad'
# no tiene bandera propia: hereda la de 'anio_construccion', de donde se deriva.
# max(1, n) evita que una variable sin nulos (n = 0) cuente como "valor imposible".
tabla_nulos["origen principal"] = [
    "valor imposible -> NaN" if f"flag_imposible_{v}" in train and train[f"flag_imposible_{v}"].sum() >= 0.5 * max(1, n)
    else ("incoherencia en la fila -> NaN" if f"flag_incoherente_{v}" in train
          and train[f"flag_incoherente_{v}"].sum() >= 0.5 * max(1, n)
          else ("año imposible -> NaN" if v == "antiguedad" and train["flag_imposible_anio_construccion"].sum() >= 0.5 * max(1, n)
                else "faltante en origen"))
    for v, n in zip(tabla_nulos.index, tabla_nulos["n nulos"])]
tabla_nulos

# %% tags=["hide-input"]
# Figura del patrón de nulos. Izquierda: matriz de nulos (oscuro = falta el dato)
# en 4 000 filas ordenadas por estrato; si los nulos se agrupan en una franja de
# estrato, el faltante no es al azar. Derecha: correlación entre indicadores de
# nulo (valores altos = variables que faltan juntas, en los mismos registros).
con_nulos = [c for c in vars_nulos if train[c].isna().any()]   # solo las que tienen nulos
# Submuestra con semilla fija solo para que la matriz sea legible; ordenada por estrato.
muestra = train.sample(min(4000, len(train)), random_state=C.SEED).sort_values(C.OBJETIVO)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [3, 2]})
if con_nulos:
    sns.heatmap(muestra[con_nulos].isna().T, cbar=False, cmap=["#e8eef4", "#1f3b57"], ax=axes[0],
                xticklabels=False)
    axes[0].set_title("Matriz de nulos (4 000 filas, ordenadas por estrato)")
    axes[0].set_xlabel("filas (estrato 1 → 6)")
    # Indicadores 0/1 de nulo en TODO train (no en la submuestra) para la correlación.
    co = train[con_nulos].isna().astype(float)
    sns.heatmap(co.corr() if len(con_nulos) > 1 else co.corr(), annot=True, fmt=".2f", cmap="rocket_r",
                vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("Correlación entre indicadores de nulo")
G.guardar(fig, "03_matriz_nulos")
plt.show()

# %% [markdown]
# ### Mecanismo (MCAR / MAR / MNAR)
#
# No basta con saber **cuántos** datos faltan; importa **por qué** faltan,
# porque de eso depende si rellenarlos introduce sesgo. Hay tres casos
# clásicos (Rubin, 1976), que se entienden mejor con un ejemplo sobre el año
# de construcción:
#
# | Mecanismo | Significado | Ejemplo |
# |---|---|---|
# | **MCAR** (completamente al azar) | faltar no depende de nada | un funcionario olvida digitar el año en fichas elegidas al azar |
# | **MAR** (al azar, dado lo observado) | faltar depende de otras variables que sí vemos | el año falta más en cierto sector de la ciudad, y el sector sí está en los datos |
# | **MNAR** (no al azar) | faltar depende del propio valor que falta | el año falta más justo en las casas muy antiguas, porque nadie lo sabe |
#
# Con MCAR, rellenar con la mediana no sesga nada. Con MAR, las otras
# variables ayudan a corregir. MNAR no se puede detectar con los datos (habría
# que conocer el valor que falta) y solo se argumenta con conocimiento del
# dominio.
#
# Con cientos de miles de filas la prueba de Little rechaza MCAR ante
# cualquier diferencia mínima, así que se usan tres evidencias complementarias
# y se interpreta la **magnitud**:
#
# 1. **Faltante vs. estrato:** chi² y V de Cramér entre el indicador de nulo y
#    el objetivo. V ≈ 0 es compatible con MCAR respecto al objetivo.
# 2. **Registros completos vs. incompletos:** diferencia de medias estandarizada
#    (SMD) en las demás variables: la diferencia de medias entre las filas con
#    y sin el dato, dividida por la desviación estándar. *Ejemplo:* si las
#    viviendas sin año miden en promedio 90 m² y las demás 80 m², con una
#    desviación de 50 m², la SMD es (90 − 80) ÷ 50 = 0.2. |SMD| < 0.1 se
#    considera despreciable.
# 3. **¿Se puede predecir el faltante?** Se entrena una regresión logística
#    que intenta adivinar, con las variables observadas, si a una fila le
#    falta el dato. Su acierto se mide con el **AUC**: la probabilidad de que,
#    al tomar una fila con faltante y otra sin él, el modelo le dé más
#    probabilidad de faltante a la primera. AUC ≈ 0.5 es adivinar a ciegas
#    (MCAR plausible); AUC claramente > 0.5 indica que el faltante depende de
#    variables observadas (**MAR**). MNAR no se puede probar con los datos: se
#    argumenta con conocimiento del dominio.

# %%
# Diagnóstico del mecanismo de faltantes con tres evidencias por variable:
# (1) V de Cramér entre el indicador de nulo y el estrato, (2) diferencia de medias
# estandarizada (SMD) entre registros con y sin nulo, y (3) AUC de una regresión
# logística que intenta predecir el nulo a partir de variables observadas.
# Variables observadas que se usan como "explicativas" del faltante.
observadas = ["area_construida", "total_habitaciones", "total_banios", "total_plantas", "x_km", "y_km"]
# Régimen del predio en one-hot (con "NA" como categoría) para añadirlo a la logística.
dummies_cond = pd.get_dummies(train["condicion_predio"].fillna("NA"), prefix="cond", dtype=float)
filas = []
for v in con_nulos:
    ind = train[v].isna()          # indicador de nulo de la variable v (True = falta)
    if ind.sum() < 30:
        continue                   # con menos de 30 nulos no hay base para ningún diagnóstico
    # (1) Asociación nulo ~ estrato: V ≈ 0 es compatible con MCAR respecto al objetivo.
    cv = E.cramers_v(ind, train[C.OBJETIVO])
    # (2) SMD de cada variable observada entre filas con nulo (a) y sin nulo (b),
    # usando la desviación estándar combinada; |SMD| < 0.1 se considera despreciable.
    smd = {}
    for o in observadas:
        if o == v:
            continue               # no se compara la variable consigo misma
        a, b = train.loc[ind, o].dropna(), train.loc[~ind, o].dropna()
        sd = np.sqrt((a.var() + b.var()) / 2)
        smd[o] = (a.mean() - b.mean()) / sd if sd > 0 else np.nan
    # (3) Matriz de predictores del nulo: observadas (sin v) imputadas con su mediana
    # de train, solo para este diagnóstico, más las dummies de condición del predio.
    X = pd.concat([train[[o for o in observadas if o != v]].fillna(train[[o for o in observadas if o != v]].median()),
                   dummies_cond], axis=1)
    # Submuestra de hasta 60 000 filas (semilla fija) para acotar el tiempo de cómputo.
    sub = np.random.default_rng(C.SEED).choice(len(X), min(60_000, len(X)), replace=False)
    # AUC promedio en CV de 3 folds; el escalado va dentro del pipeline para que se
    # ajuste en cada fold. AUC ≈ 0.5 -> MCAR plausible; AUC alto -> MAR.
    auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
                          X.iloc[sub], ind.iloc[sub], cv=3, scoring="roc_auc").mean()
    # Variable con la mayor |SMD| (los NaN se ignoran asignándoles -1 en la comparación).
    smd_max = max(smd.items(), key=lambda t: abs(t[1]) if pd.notna(t[1]) else -1)
    # Se guarda también la tasa de nulo en cada estrato (1/2/3/4/5/6) para leerla a ojo.
    filas.append({"variable": v, "n nulos": int(ind.sum()), "% nulos": 100 * ind.mean(), "V (nulo~estrato)": cv["V"],
                  "|SMD| máx.": abs(smd_max[1]), "en variable": smd_max[0], "AUC predecir nulo": auc,
                  "tasa de nulo por estrato (%)": " / ".join(f"{100 * r:.1f}" for r in
                                                              ind.groupby(train[C.OBJETIVO]).mean())})
mecanismo = pd.DataFrame(filas).set_index("variable") if filas else pd.DataFrame()


# Regla de lectura que combina las tres evidencias. MCAR solo si las tres son
# débiles a la vez; en otro caso el faltante depende de lo observado (MAR). MNAR no
# se puede probar con los datos y se argumenta con conocimiento del dominio.
def lectura_mecanismo(r):
    if r["n nulos"] < 100:
        return "muy pocos casos (<100): no evaluable; proviene de valores imposibles"
    if r["AUC predecir nulo"] < 0.6 and r["V (nulo~estrato)"] < 0.05 and r["|SMD| máx."] < 0.1:
        return "compatible con MCAR"
    return "MAR (depende de variables observadas)"


if len(mecanismo):
    mecanismo["lectura"] = mecanismo.apply(lectura_mecanismo, axis=1)
mecanismo

# %% [markdown]
# ```{admonition} Interpretación y decisión
# :class: note
# - **Los faltantes son mínimos.** En train, la variable con más nulos es
#   `total_habitaciones` con **271 (0.12 %)**; luego `planta_ubicacion` con
#   237 (0.11 %), `antiguedad` con 129 (0.06 %), baños con 28 (0.01 %) y
#   plantas con 1. Todos provienen de las reglas del capítulo 1, no de celdas
#   vacías en origen: pisos codificados (97–99) y años centinela (paso 4), e
#   incoherencias dentro de la fila (paso 5). La columna "origen principal"
#   lo confirma: planta y plantas salen como "valor imposible", antigüedad
#   como "año imposible", y habitaciones y baños como "incoherencia en la
#   fila" (bandera `flag_incoherente_*`), igual que en el dataset completo,
#   donde todos los nulos de habitaciones (0.107 %) y de baños (0.012 %) son
#   incoherencias (cap. 1, columna "de ellos, por incoherencia"). Las
#   variables sin nulos aparecen como "faltante en origen" solo porque es la
#   etiqueta por defecto de la tabla.
# - **Baños** tiene menos de 30 nulos y queda fuera del diagnóstico de
#   mecanismo: no hay base para evaluarlo.
# - **Habitaciones** es **MAR**, pero por construcción: su nulo se predice casi
#   perfectamente (AUC = 0.997) y la mayor diferencia está en
#   `area_construida` (|SMD| = 1.63), porque la regla anula las habitaciones
#   cuando el área por habitación es menor de 6 m². El faltante depende de
#   una variable observada (el área), no del azar.
# - **Piso de ubicación** es **MAR**: su nulo se predice casi perfectamente
#   (AUC = 0.963) y la mayor diferencia está en `total_habitaciones`
#   (|SMD| = 1.65); el 0.3 % de las unidades de estrato 3 tiene un código de
#   piso, frente a ≤ 0.2 % en el resto. El código se concentra en ciertos
#   edificios o sectores, no al azar.
# - **Antigüedad** es **MAR**: su nulo se predice casi perfectamente con las
#   variables observadas (AUC = 0.971) y la mayor diferencia estandarizada
#   está en `y_km` (|SMD| = 1.96), es decir, los años faltantes se concentran
#   en una zona concreta de la ciudad (0.3 % de nulos en estrato 4 frente a
#   ~0 % en el resto). Los 129 años imposibles del dataset completo (cap. 1)
#   caen todos en train, lo que es coherente con esa concentración espacial.
#   No es un faltante al azar: parece un proyecto o un sector cuyo año no se
#   registró.
# - **Decisión:** mediana + indicador de faltante, dentro del Pipeline (el
#   indicador deja que el modelo use "no tener el dato" si eso informa el
#   estrato). Con tan pocos nulos (< 1 %), la elección del imputador apenas afecta
#   al modelo; lo importante es no aprenderlo con test. Ninguna variable
#   supera el 30 % de faltantes, así que, siguiendo las notas del curso
#   (9.10.4.1.4), **no se elimina ninguna variable** por faltantes.
# ```

# %%
# Registro de decisiones de imputación en decisiones_eda.json, con su motivo. El
# modelo base (cap. 10) lee este archivo, de modo que cada paso del Pipeline queda
# rastreado al hallazgo de este capítulo. La imputación se ajusta dentro del
# Pipeline, es decir, solo con los datos de entrenamiento de cada fold.
C.guardar_decision(
    "imputacion_numericas", "mediana + indicador de faltante (SimpleImputer(add_indicator=True)), dentro del Pipeline",
    "Faltantes < 30 % en todas las variables; mecanismo MAR/informativo según AUC y SMD (cap. 3.1); "
    "la mediana es robusta a la asimetría fuerte de áreas y conteos.")
C.guardar_decision(
    "imputacion_categoricas", "categoría explícita 'faltante'",
    "Conserva la información de que el dato no existía, sin inventar una categoría real.")

# %% [markdown]
# ## 3.2 Outliers univariados (IQR de Tukey)
#
# Los valores **imposibles** ya se convirtieron en NaN (capítulo 1). Lo que
# queda son valores **extremos pero plausibles**: una casa de 8 baños, un lote
# muy grande. Se cuantifican con la **regla de Tukey**, que se basa en los
# cuartiles: $Q_1$ deja por debajo el 25 % de los datos, $Q_3$ el 75 %, y el
# rango intercuartílico es $IQR = Q_3 - Q_1$. Es outlier todo valor fuera de
# $[Q_1 - 1.5\,IQR,\ Q_3 + 1.5\,IQR]$, y "outlier lejano" todo valor fuera
# de $[Q_1 - 3\,IQR,\ Q_3 + 3\,IQR]$.
#
# *Ejemplo:* si el área construida tiene $Q_1 = 50$ m² y $Q_3 = 100$ m², el
# $IQR$ es 50 m² y el límite superior de Tukey es 100 + 1.5 × 50 = 175 m².
# Una casa de 200 m² sería "outlier", aunque en el norte de la ciudad es
# perfectamente normal. Por eso la regla aquí **describe**, no limpia.

# %%
# Outliers univariados con la regla de Tukey (1.5·IQR) y con el criterio de
# "outliers lejanos" (3·IQR), calculados solo con train. Se reportan también p99.9
# y máximo porque el p99.9 es el umbral de la winsorización que se decide después.
nums = [c for c in C.NUMERICAS if c in train]
filas = []
for c in nums:
    v = train[c].dropna()
    q1, q3 = v.quantile([.25, .75])
    iqr = q3 - q1
    # Fracción de valores fuera de las vallas de Tukey (1.5·IQR) y de las lejanas (3·IQR).
    f15 = ((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).mean()
    f30 = ((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).mean()
    filas.append({"variable": c, "Q1": q1, "Q3": q3, "IQR": iqr, "lím. sup. 1.5·IQR": q3 + 1.5 * iqr,
                  "% fuera 1.5·IQR": 100 * f15, "% fuera 3·IQR": 100 * f30,
                  "p99.9": v.quantile(.999), "máx": v.max()})
outliers = pd.DataFrame(filas).set_index("variable")
outliers

# %% tags=["hide-input"]
# Boxplots de cada variable numérica de train en escala log(1 + x). El logaritmo
# comprime las colas largas para poder ver el cuerpo de la distribución y los
# extremos en la misma figura; los puntos sueltos son los outliers de Tukey.
fig, axes = plt.subplots(2, (len(nums) + 1) // 2, figsize=(16, 8))
for ax, c in zip(axes.ravel(), nums):
    v = train[c].dropna()
    # clip(lower=0) evita log de negativos; los puntos atípicos se dibujan pequeños.
    ax.boxplot(np.log1p(v.clip(lower=0)), orientation="horizontal", widths=0.6,
               flierprops={"markersize": 2, "alpha": 0.3})
    ax.set_title(c)
    ax.set_xlabel("log(1 + valor)")
    ax.set_yticks([])
# Se ocultan los paneles sobrantes de la grilla si el número de variables es impar.
for ax in axes.ravel()[len(nums):]:
    ax.axis("off")
fig.suptitle("Boxplots en escala log1p (entrenamiento)")
fig.tight_layout()
G.guardar(fig, "03_boxplots_outliers")
plt.show()

# %% [markdown]
# ```{admonition} Interpretación y decisión
# :class: note
# - La regla de Tukey marca entre **3 % y 6 %** de las filas en áreas,
#   habitaciones y baños (8 % en piso de ubicación), pero muy pocas son "lejanas" (3·IQR): 2.1 % en
#   área construida, 1.3 % en terreno, < 1 % en habitaciones y baños. Las
#   distribuciones son muy **asimétricas a la derecha**; esos puntos son
#   viviendas grandes, no errores.
# - En `total_plantas` y `altura` la regla es **inservible**: su IQR es 0 (Q1 =
#   Q3 = 1 planta; Q1 = Q3 = 3 m). En plantas, *cualquier* casa de dos pisos
#   queda marcada como outlier (11.1 % de train); en altura, cualquier valor
#   distinto de 3 (0.3 %). Es un ejemplo de por qué no se debe eliminar filas
#   con una regla automática sin mirarla.
# - Los máximos que quedan son grandes pero posibles: 1 971 m² construidos,
#   36 022 m² de terreno, 20 habitaciones, 15 baños y 35 plantas. Los de
#   área construida, habitaciones y baños quedan dentro de los topes de la
#   regla de registros agregados del cap. 1 (paso 2c: más de 2 000 m², 20
#   habitaciones o 15 baños es un edificio entero en una fila). El p99.9 (los
#   umbrales que usará la winsorización) está en 692 m² de área construida,
#   5 193 m² de terreno, 12 habitaciones, 6 baños y 12 plantas.
# - En `planta_ubicacion`, los códigos 97–99 ya se trataron como faltantes
#   (cap. 1, techo del rango en 60): el p99.9 queda en 20 y el máximo en 41,
#   un piso real de las torres más altas, no un código.
# - **Decisión:** no se elimina ninguna fila por ser outlier estadístico,
#   porque el área construida crece con el estrato (el capítulo 6 lo mide: la
#   mediana pasa de 54 m² en estrato 1 a 152 m² en estrato 6). Las viviendas
#   más grandes pesan más en las clases altas, que ya son minoritarias, y
#   eliminarlas les quitaría ejemplos. Su influencia en el modelo lineal se
#   reduce de dos formas: (i) con `log1p` en las variables muy asimétricas
#   (capítulo 5) y (ii) con **winsorización** a los percentiles 0.1 y 99.9
#   **aprendidos en train** dentro del Pipeline. Winsorizar es recortar sin
#   borrar: un valor por encima del percentil 99.9 (por ejemplo, 692 m² de
#   área construida) se reemplaza por ese percentil, y la fila se conserva.
# - Los outliers **multivariados** (combinaciones raras) se revisan en el
#   capítulo 7.
# ```

# %%
# Se registra la decisión de winsorizar a los percentiles 0.1 y 99.9 en lugar de
# eliminar filas. Los cuantiles se aprenden en train dentro del Pipeline (clase de
# recorte de modelo.py), así que test nunca influye en los umbrales.
C.guardar_decision("winsorizacion_cuantiles", [0.001, 0.999],
                   "Colas largas plausibles (cap. 3.2): se recortan a p0.1/p99.9 de train dentro del "
                   "Pipeline en lugar de eliminar filas, para no sesgar las clases altas.")

# %% [markdown]
# ## 3.3 Sesgo de muestreo y representatividad
#
# ¿A quién deja fuera el modelo? En la sección 1.7 se decidió modelar solo
# con las viviendas que tienen coordenadas. Aquí se mide el costo de esa
# decisión: se comparan las filas **con** coordenadas (población de
# modelado) contra las **53 894 sin coordenadas**, casi todas mejoras en
# terreno ajeno. Para no mirar el conjunto de prueba, la comparación usa solo
# train frente a las filas no ubicables.
#
# La fuerza de la diferencia se mide con la **V de Cramér**, una medida de
# asociación entre dos variables categóricas que va de 0 (las proporciones de
# estrato son iguales en los dos grupos) a 1 (conocer el grupo determina el
# estrato). Como guía: menos de 0.1 es despreciable, 0.1–0.3 débil, 0.3–0.5
# moderada y más de 0.5 fuerte.

# %%
# Sesgo de cobertura: compara la distribución del estrato en la población modelada
# (solo train, para no mirar test) con la de las filas sin coordenadas, que quedan
# fuera del modelo. Si difieren mucho, el modelo no representa a toda la ciudad.
ruta_sin = C.CARPETA_PROCESADOS / "sin_coordenadas.parquet"
# Se lee el formato con que se guardó en el cap. 2 (parquet o, si no, CSV).
sin = pd.read_parquet(ruta_sin) if ruta_sin.exists() else pd.read_csv(ruta_sin.with_suffix(".csv"))
# % de cada estrato dentro de cada grupo; fillna(0) por si a un grupo le falta un estrato.
comp = pd.concat({
    "modelado (train)": train[C.OBJETIVO].value_counts(normalize=True),
    "sin coordenadas": sin[C.OBJETIVO].value_counts(normalize=True),
}, axis=1).sort_index().fillna(0).mul(100)
comp.index = [C.NOMBRES_CLASES[i] for i in comp.index]
# V de Cramér entre "grupo" (modelado / sin coordenadas) y estrato: se apilan ambos
# grupos en una sola tabla para medir la fuerza de la asociación (0 = sin sesgo).
cv_sesgo = E.cramers_v(pd.concat([pd.Series("modelado", index=train.index), pd.Series("sin", index=sin.index)],
                                 ignore_index=True),
                       pd.concat([train[C.OBJETIVO], sin[C.OBJETIVO]], ignore_index=True))
print(f"V de Cramér (grupo ~ estrato) = {cv_sesgo['V']:.3f}")
comp

# %%
# Misma comparación para el régimen del predio (condicion_predio): sirve para
# explicar POR QUÉ faltan coordenadas (se espera que dominen los predios informales).
cond = pd.concat({
    "modelado (train)": train["condicion_predio"].value_counts(normalize=True),
    "sin coordenadas": sin["condicion_predio"].value_counts(normalize=True),
}, axis=1).fillna(0).mul(100)
cond

# %% tags=["hide-input"]
# Barras pareadas del % de cada estrato en la población modelada (azul) y en las
# filas sin coordenadas (naranja): visualiza el sesgo de cobertura hacia estratos bajos.
fig, ax = plt.subplots(figsize=(11, 4.5))
comp.plot.barh(ax=ax, color=["#4c72b0", "#dd8452"])
ax.invert_yaxis()   # estrato 1 arriba, 6 abajo
ax.set_xlabel("% de filas del grupo")
ax.set_title("Distribución de estrato: población modelada vs. filas sin coordenadas")
G.guardar(fig, "03_sesgo_cobertura")
plt.show()

# %% [markdown]
# ```{admonition} Interpretación
# :class: important
# - El sesgo de cobertura es **fuerte**: de las filas sin coordenadas, el
#   **66.3 % son estrato 1 y el 30.2 % estrato 2** (96.5 % en los dos estratos
#   más bajos), frente a 25.5 % y 19.9 % en la población modelada. La V de
#   Cramér entre "tener o no coordenadas" y el estrato es **0.42**: una
#   asociación moderada en la escala de arriba, cerca del límite de "fuerte"
#   (0.5). Para comparar, ninguna predictora categórica del capítulo 6 supera
#   0.33 con el estrato.
# - La razón es estructural: el **99.5 %** de las filas sin coordenadas son
#   predios `Informal` (sin lote propio ni edificio matriz de donde tomar un
#   centroide). En train, lo informal pesa solo 7.6 %.
# - Consecuencia: el modelo representa la **ciudad formal**. Aprenderá del
#   estrato 1 sobre todo su versión formal, y sus métricas no deben
#   extrapolarse a asentamientos informales. Se declara como limitación en las
#   conclusiones (11.2).
# - Mejora evaluada en la sección 1.7.3: el 92.7 % de estas viviendas comparte
#   manzana (primeros 17 dígitos del NPN) con viviendas ubicadas, así que se
#   podrían ubicar con el centroide de su manzana. Queda como primer paso del
#   Entregable 2 (11.3).
# ```
