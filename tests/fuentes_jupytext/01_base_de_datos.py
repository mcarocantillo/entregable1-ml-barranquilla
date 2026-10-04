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
# # 1. Base de datos
#
# Este capítulo cubre la **Sección 1 de las instrucciones del proyecto**: problema de investigación,
# justificación y fuente del dataset, diccionario de variables, estructura de
# los datos, tamaño de muestra y calidad de datos (duplicados, valores
# imposibles, categorías mal escritas), más las consideraciones éticas.
#
# ```{important}
# Todo lo que se hace en este capítulo son **reglas fila a fila** que no
# aprenden nada de otras filas (por ejemplo "un año de construcción de 1512 es
# imposible"). Ninguna de ellas usa estadísticas del conjunto de datos, por eso
# pueden aplicarse **antes** de reservar el conjunto de prueba (capítulo 2) sin
# generar fuga de información. Todo lo que sí aprende de los datos (medianas
# para imputar, umbrales de outliers, escalado) se calcula después, solo con
# entrenamiento.
# ```

# %% tags=["hide-input"]
# Celda de configuración: importa librerías y los módulos propios de src/.
# Toda la lógica reutilizable (rutas, semillas, reglas de limpieza) vive en src/
# para que cada capítulo use exactamente las mismas definiciones (reproducibilidad).
import sys
import warnings
from pathlib import Path

# La raíz del libro es la carpeta padre de notebooks/; se agrega a sys.path para
# poder importar el paquete src tanto al ejecutar desde notebooks/ como desde la raíz.
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))
# Se silencian solo avisos de obsolescencia de pandas/seaborn (no afectan resultados).
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config as C      # rutas, SEED, constantes y listas de variables
from src import graficos as G    # estilo común de figuras y guardado en figuras/
from src import limpieza as L    # carga del CSV y reglas deterministas fila a fila

G.estilo()             # fuentes grandes y tema uniforme para todas las figuras
C.aviso_sintetico()    # imprime una alerta si se está usando el CSV sintético de prueba
# Formato de tablas: 3 decimales con separador de miles y hasta 40 columnas visibles.
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
pd.set_option("display.max_columns", 40)

# %% [markdown]
# ## 1.1 Problema de investigación
#
# **Pregunta.** ¿En qué medida las **características físicas y catastrales** de
# una unidad de vivienda (área construida, número de baños y habitaciones,
# pisos, antigüedad, tipo de vivienda, régimen de propiedad) y su
# **localización** permiten predecir su **estrato socioeconómico** (1 a 6) en
# Barranquilla?
#
# **Tipo de tarea.** Clasificación supervisada multiclase (6 clases,
# **ordinales**) con datos que tienen coordenadas pero no tiempo: según la
# Figura 1 de las instrucciones del proyecto se sigue la **ruta A (clasificación)** añadiendo la
# sección 2.7 (componente espacial) y una partición por **bloques espaciales**.
#
# **Por qué importa.** En Colombia el estrato (Ley 142 de 1994) determina las
# tarifas y subsidios cruzados de los servicios públicos domiciliarios y se usa
# para focalizar inversión social. Según el DANE, la estratificación clasifica
# los inmuebles residenciales a partir de la base catastral y de las
# características de las viviendas y su entorno. Un modelo que relacione la
# información catastral con el estrato sirve para (i) entender qué atributos
# físicos se asocian con cada estrato, (ii) detectar predios cuyo estrato
# asignado es atípico frente a sus características (posibles casos a revisar) y
# (iii) proponer un estrato de referencia para predios que hoy aparecen como
# `No_Aplica`/`Otro`.
#
# **Relación con la tesis.** La tesis (tipologías constructivas y demanda de
# enfriamiento) usa estos mismos datos con un enfoque **no supervisado**; este
# entregable es el ejercicio **supervisado** del curso y le aporta a la tesis
# una base de datos limpia, auditada y con un EDA espacial completo.

# %% [markdown]
# ## 1.2 Justificación, fuente y licencia
#
# | Aspecto | Detalle |
# |---|---|
# | Fuente | Servicio ArcGIS REST de **datos abiertos de catastro** de la Alcaldía de Barranquilla (Gerencia de Gestión Catastral) |
# | Enlace | `https://miciudad.barranquilla.gov.co/gis/rest/services/catastro/datosabiertos/MapServer` |
# | Capas usadas | 505 Características, 305 Unidad de construcción, 602 relación UC–Predio, 500 Predio, 315 Terreno (geometría), 601 relación Terreno–Predio |
# | Descarga | `01_descarga_predios.py` (paginación de 2 000 registros, caché en parquet, uniones locales con pandas) |
# | Fecha de descarga | 15 de septiembre de 2026 |
# | Licencia | Publicado como **dato abierto** por la Alcaldía de Barranquilla (Gerencia de Gestión Catastral) en su portal de datos geográficos. El servicio no declara una licencia explícita en sus metadatos; se usa como información pública con fines académicos y se cita la fuente. No se redistribuyen datos de personas: el NPN y las coordenadas se tratan como se explica en la sección 1.8. |
#
# **Por qué este dataset.** (i) Es información oficial, completa para la zona
# urbana formal (no es una encuesta), (ii) supera ampliamente el mínimo de
# 20 000 observaciones, (iii) combina variables físicas, categóricas y
# espaciales, lo que permite un EDA completo (uni, bi, multivariado y
# espacial) y (iv) es el mismo insumo de la tesis.

# %% [markdown]
# ## 1.3 Carga y embudo de filas
#
# El embudo muestra cuántas filas quedan tras cada regla y por qué. Esto
# resuelve la tabla pendiente de la bitácora (sección 3.3): los números salen
# directamente de `src/limpieza.py`.

# %%
# Carga del CSV crudo y aplicación de las reglas deterministas de limpieza.
# Son reglas fila a fila (no aprenden de otras filas), por eso pueden ir ANTES de
# reservar el conjunto de prueba sin generar fuga de información.
# cargar_crudo lee el CSV (NPN como texto para no perder ceros) y añade 'id_fila'.
crudo = L.cargar_crudo()
# Se resta 1 a las columnas porque 'id_fila' la agrega el pipeline, no viene del CSV.
print(f"Archivo: {C.ruta_csv().name}  |  filas: {len(crudo):,}  |  columnas: {crudo.shape[1] - 1}")
# limpiar devuelve: el dataset de trabajo, el embudo (filas que quedan tras cada
# regla y su motivo) y la tabla de valores físicamente imposibles convertidos en NaN.
df, embudo, tabla_imposibles = L.limpiar(crudo, verbose=False)
embudo

# %% tags=["hide-input"]
# Figura del embudo: barras horizontales con las filas que sobreviven a cada regla.
# Permite ver de un vistazo qué decisión de limpieza elimina más filas.
fig, ax = plt.subplots(figsize=(11, 4.5))
# Se invierte el orden ([::-1]) para que el paso 0 quede arriba y se lea de arriba abajo.
ax.barh(embudo["paso"][::-1], embudo["filas"][::-1], color=sns.color_palette("crest", len(embudo)))
# Etiqueta con el número de filas al final de cada barra.
for i, v in enumerate(embudo["filas"][::-1]):
    ax.text(v, i, f" {v:,}", va="center", fontsize=11)
ax.set_xlabel("filas")
ax.set_title("Embudo de filas por decisión de limpieza")
# Margen del 18 % a la derecha para que las etiquetas no se corten.
ax.set_xlim(0, embudo["filas"].max() * 1.18)
G.guardar(fig, "01_embudo")  # se guarda en figuras/ para el informe
plt.show()

# %% [markdown]
# Vista de las primeras filas. Por la **consideración ética** de la sección 1.8
# el número predial se muestra truncado (identifica un inmueble concreto).

# %%
# Vista de las primeras filas con el número predial enmascarado (consideración
# ética 1.8: el NPN completo identifica un inmueble concreto y permite reidentificar).
def enmascarar_npn(s):
    # Conserva solo los 12 primeros dígitos (depto, municipio, zona, sector...),
    # que ubican la zona pero no el predio individual.
    return s.astype("string").str[:12] + "…"

# Se omiten las columnas de bandera (flag_*) para que la vista sea legible.
vista = df.drop(columns=df.filter(like="flag_").columns).head(5).copy()
vista["numero_predial_nacional"] = enmascarar_npn(vista["numero_predial_nacional"])
vista["npn_edificio"] = enmascarar_npn(vista["npn_edificio"])
vista.T  # transpuesta: una fila por variable, más fácil de leer con muchas columnas

# %% [markdown]
# ## 1.4 Diccionario de variables
#
# El rol de cada variable (objetivo, predictora, identificador, metadato) se
# justifica en la auditoría de fuga (capítulo 9). Si el servidor responde, se
# descargan también los alias y dominios oficiales de ArcGIS (por ejemplo, el
# significado de los códigos de `tipo_vivienda`).

# %%
# Diccionario de variables: nombre, tipo, unidad, significado y ROL en el modelo.
# El rol (objetivo, predictora, identificador, metadato) es lo que luego se audita
# en el capítulo 9 para asegurar que ninguna columna con fuga entre al modelo.
dicc = pd.DataFrame([
    ("numero_predial_nacional", "texto (30 dígitos)", "—", "Número predial nacional (IGAC): depto, municipio, zona, sector, comuna, barrio, manzana, terreno, condición, edificio, piso, unidad", "identificador"),
    ("npn_edificio", "texto (22 dígitos)", "—", "Prefijo del NPN que identifica el edificio/lote (derivada)", "identificador / grupo"),
    ("estrato", "categórica ordinal (texto)", "—", "Estrato socioeconómico tal como llega de ArcGIS (Bajo_Bajo_1 … Alto_6, No_Aplica, Otro)", "origen del objetivo"),
    ("estrato_num", "categórica ordinal", "1–6", "Estrato numérico extraído del texto", "OBJETIVO"),
    ("area_construida", "numérica continua", "m²", "Área construida de la unidad", "predictora"),
    ("area_catastral_terreno", "numérica continua", "m²", "Área del terreno del predio (0 en unidades PH sin lote propio)", "predictora"),
    ("total_habitaciones", "numérica discreta", "conteo", "Número de habitaciones", "predictora"),
    ("total_banios", "numérica discreta", "conteo", "Número de baños", "predictora"),
    ("total_plantas", "numérica discreta", "conteo", "Número de plantas/pisos de la construcción", "predictora"),
    ("anio_construccion", "numérica discreta", "año", "Año de construcción", "origen de antigüedad"),
    ("antiguedad", "numérica discreta", "años", f"{C.ANIO_ACTUAL} − año de construcción (derivada)", "predictora"),
    ("planta_ubicacion", "numérica discreta", "piso", "Piso en que está la unidad", "predictora"),
    ("altura", "numérica continua", "m (verificar)", "Altura de la unidad de construcción", "predictora"),
    ("tipo_vivienda", "categórica nominal (código)", "—", "Código de dominio ArcGIS del tipo de vivienda", "predictora"),
    ("uso", "categórica nominal", "—", "Uso catastral de la unidad (casa, apartamento en PH, barraca…)", "predictora"),
    ("condicion_predio", "categórica nominal", "—", "Régimen: NPH (no propiedad horizontal), PH_Matriz, PH_Unidad_Predial, Informal…", "predictora"),
    ("destinacion_economica", "categórica nominal", "—", "Destinación económica del predio", "predictora"),
    ("tipo_planta", "categórica nominal", "—", "Tipo de planta (Piso, Sótano, Mezanine…)", "predictora"),
    ("centroide_lat / centroide_lon", "numérica continua", "grados (WGS84, EPSG:4326)", "Centroide del terreno del predio (o de su edificio matriz)", "espacial"),
    ("x_km / y_km", "numérica continua", "km", "Coordenadas proyectadas localmente, origen en el centro histórico (derivada)", "predictora espacial"),
    ("dist_centro_km", "numérica continua", "km", "Distancia al centro histórico (derivada)", "predictora espacial"),
    ("centroide_fuente", "categórica nominal", "—", "Cómo se obtuvo el centroide: directo / propagado_matriz / sin_dato (metadato del pipeline)", "metadato (excluida)"),
    ("flag_imposible_*", "booleana", "—", "Marca si el valor original era físicamente imposible y se convirtió en NaN", "auditoría"),
], columns=["variable", "tipo", "unidad", "significado", "rol"])

# Intento opcional de enriquecer el diccionario con los alias y dominios oficiales
# del servicio ArcGIS (se descargan una vez y se guardan en datos/dominios_arcgis.json).
try:
    from src.dominios import descargar_dominios
    dominios = descargar_dominios()
    # Alias oficial de cada campo; queda vacío para las variables derivadas.
    dicc["alias ArcGIS"] = dicc["variable"].map(lambda v: (dominios.get(v) or {}).get("alias"))
    # Tabla código -> significado de tipo_vivienda (llega como código numérico).
    tv = (dominios.get("tipo_vivienda") or {}).get("dominio")
    if tv:
        print("Dominio oficial de tipo_vivienda:", tv)
except Exception as e:  # sin internet o bloqueado: el libro sigue
    print(f"(No se pudieron descargar los dominios de ArcGIS: {type(e).__name__}. "
          "El diccionario se muestra sin alias oficiales.)")
dicc

# %% [markdown]
# ## 1.5 Estructura de los datos
#
# - **Unidad de observación:** la *unidad residencial física* (una fila de la
#   capa 505 "Características" = una casa o un apartamento). Cada fila se une a
#   su unidad de construcción, a su predio y al centroide del terreno.
# - **Nivel de agregación:** unidad de vivienda (no manzana ni barrio). Varias
#   unidades pueden pertenecer al mismo predio y al mismo edificio.
# - **Tipo de datos:** **transversal con componente espacial** (una foto del
#   catastro en la fecha de descarga). `anio_construccion` es un atributo, no una
#   marca temporal de la observación, así que no aplica la sección 2.6
#   (componente temporal) ni la 2.8 (espacio-temporal).
# - **Jerarquía:** unidad ⊂ predio (NPN de 30 dígitos) ⊂ edificio/lote (prefijo
#   de 22 dígitos) ⊂ manzana ⊂ barrio. Esta jerarquía es la razón para partir
#   por grupos/bloques y no por filas.

# %% [markdown]
# ## 1.6 Tamaño de la muestra, relación n/p y entidades independientes

# %%
# Tamaño de muestra y relación n/p: verifica el mínimo de 20 000 observaciones de
# las instrucciones del proyecto y que haya suficientes filas por parámetro. También cuenta entidades
# (predios, edificios), porque las filas de un mismo edificio no son independientes.
n = len(df)
p_num = len(C.NUMERICAS) + len(C.ESPACIALES)   # numéricas + coordenadas/distancia
p_cat = len(C.CATEGORICAS)
# Tras one-hot cada nivel de una categórica se vuelve una columna: se suman niveles.
n_niveles = sum(df[c].nunique() for c in C.CATEGORICAS if c in df)
p_efectivo = p_num + n_niveles
predios = df["numero_predial_nacional"].nunique()   # NPN de 30 dígitos
edificios = df["npn_edificio"].nunique()            # prefijo de 22 dígitos (edificio/lote)
tam = pd.DataFrame({
    "valor": [n, p_num + p_cat, p_efectivo, n / p_efectivo, predios, edificios,
              int(df["tiene_coordenadas"].sum())],
}, index=["filas (unidades de vivienda)", "variables predictoras candidatas",
          "columnas tras one-hot (aprox.)", "n / p (con one-hot)",
          "predios únicos (NPN)", "edificios/lotes únicos (NPN 22 díg.)",
          "filas con coordenadas"])
tam

# %%
# Casos por clase del objetivo (estrato 1-6), en conteo y porcentaje, para dimensionar
# el desbalance antes de decidir la partición estratificada y las métricas.
casos = (df[C.OBJETIVO].value_counts().sort_index().rename("casos").to_frame()
         .assign(**{"%": lambda t: 100 * t["casos"] / t["casos"].sum()}))
casos.index = [C.NOMBRES_CLASES[i] for i in casos.index]  # etiquetas legibles ("1 Bajo-bajo"...)
casos

# %% [markdown]
# **Tamaño efectivo (descriptivo).** Las unidades de un mismo edificio
# comparten estrato casi siempre, así que no son observaciones independientes.
# Se estima la correlación intraclase (ICC(1), ANOVA de una vía con el edificio
# como grupo, tratando el estrato como numérico) y el **efecto de diseño de
# Kish**: $\text{deff} = 1 + (\tilde m - 1)\,\text{ICC}$, con
# $\tilde m = \sum_j m_j^2 / N$ (tamaño medio de grupo ponderado, adecuado
# cuando los edificios tienen tamaños muy desiguales) y
# $n_\text{ef} = n/\text{deff}$. Es una descripción del dataset completo: no
# alimenta ninguna decisión de modelado.

# %%
# ICC(1) y efecto de diseño de Kish: miden cuánto se parecen en estrato las unidades
# de un mismo edificio y cuántas observaciones "independientes" equivalen a las filas.
# Es solo descriptivo (dataset completo); no alimenta ninguna decisión aprendida.
g = df.groupby("npn_edificio")[C.OBJETIVO]   # el edificio es el grupo del ANOVA
m = g.size()                                 # m_j: unidades por edificio
k, N = len(m), m.sum()                       # k grupos, N observaciones
media_total = df[C.OBJETIVO].mean()
# Sumas de cuadrados entre grupos (SSB) y dentro de grupos (SSW) del ANOVA de una vía.
ssb = (m * (g.mean() - media_total) ** 2).sum()
ssw = ((df[C.OBJETIVO] - g.transform("mean")) ** 2).sum()
msb, msw = ssb / (k - 1), ssw / (N - k)      # cuadrados medios con sus grados de libertad
# m0: tamaño de grupo "ajustado" para grupos desiguales (fórmula clásica del ICC(1)).
m0 = (N - (m ** 2).sum() / N) / (k - 1)
icc = (msb - msw) / (msb + (m0 - 1) * msw)
# Tamaño medio ponderado Σm²/N: da más peso a las torres grandes, que son las que
# más inflan la varianza cuando los edificios tienen tamaños muy desiguales.
m_pond = (m ** 2).sum() / N
deff = 1 + (m_pond - 1) * icc                # efecto de diseño de Kish
print(f"Unidades por edificio: media {m.mean():.2f}, mediana {m.median():.0f}, máximo {m.max()}, "
      f"media ponderada Σm²/N {m_pond:.2f}")
print(f"ICC(1) del estrato dentro de edificio: {icc:.3f}")
# Tamaño efectivo n_ef = N / deff: filas equivalentes si fueran independientes.
print(f"Efecto de diseño: {deff:.2f}  ->  tamaño efectivo aprox. {N / deff:,.0f} "
      f"(frente a {N:,} filas)")

# %% [markdown]
# ```{admonition} Interpretación
# :class: note
# - **Tamaño bruto.** Tras el embudo quedan **333 871 unidades de vivienda**
#   (87.3 % de las 382 597 filas descargadas), en **321 156 predios** y
#   **168 224 edificios/lotes**; 279 739 filas (83.8 %) tienen coordenadas. La
#   relación n/p es de ~9 500 filas por columna tras one-hot: el problema no es
#   de pocos datos respecto al número de variables, y el mínimo de 20 000
#   observaciones de las instrucciones del proyecto se cumple con holgura.
# - **Clases.** Estrato 1: 32.7 %, 2: 21.8 %, 3: 21.7 %, 4: 14.8 %, 5: 5.1 %,
#   6: 4.0 %. Las dos clases altas suman menos del 10 %, pero aún tienen más de
#   13 000 casos cada una.
# - **Dependencia dentro del edificio.** El ICC(1) es **0.991**: dentro de un
#   edificio el estrato prácticamente no varía, así que cien apartamentos de
#   una misma torre aportan casi la misma información sobre el estrato que uno
#   solo. La mediana es 1 unidad por edificio (la mayoría son casas), pero hay
#   torres de hasta **3 048 unidades**, y como $\tilde m$ pondera por tamaño
#   ($\tilde m = 179$), el efecto de diseño es **177.6** y el tamaño
#   efectivo "tipo muestra independiente" baja a **≈ 1 900**.
# - **Cómo leer ese 1 900.** No significa que sobren filas (hay 168 224
#   edificios distintos, muy por encima de 20 000), sino que *la precisión* de
#   cualquier estimación sobre el estrato está gobernada por los edificios y
#   las zonas, no por las filas. Dos consecuencias prácticas: (1) la
#   partición debe mantener juntas las unidades de un mismo edificio y de una
#   misma zona (cap. 2), y (2) los intervalos de confianza deben remuestrear
#   bloques, no filas (cap. 10). El cálculo usa el dataset completo solo como
#   descripción; no alimenta ninguna decisión aprendida.
# ```

# %% [markdown]
# ## 1.7 Calidad de datos
#
# ### 1.7.1 Valores faltantes (panorama)
#
# Aquí solo se cuentan. El **patrón** (matriz de nulos) y el **mecanismo**
# (MCAR/MAR/MNAR) se analizan en el capítulo 3 usando solo entrenamiento.

# %%
# Panorama de faltantes: solo se CUENTAN aquí (sobre todo el dataset, sin aprender
# nada). El patrón y el mecanismo MCAR/MAR/MNAR se estudian en el cap. 3 solo con train.
cols_revisar = C.NUMERICAS + C.CATEGORICAS + ["anio_construccion", "centroide_lat", "numero_predial_nacional"]
# % de nulos por columna tras las reglas de limpieza, ordenado de mayor a menor.
nulos = (df[cols_revisar].isna().mean().mul(100).rename("% nulos (tras reglas)")
         .to_frame().sort_values("% nulos (tras reglas)", ascending=False))
# Parte de esos nulos que se generó al convertir valores imposibles en NaN (según la
# bandera flag_imposible_*); 0 si la variable no tiene regla de dominio.
nulos["de ellos, por valor imposible"] = [
    100 * df[f"flag_imposible_{c}"].mean() if f"flag_imposible_{c}" in df else 0.0 for c in nulos.index]
nulos

# %% [markdown]
# ### 1.7.2 Duplicados exactos y casi-duplicados
#
# Tres niveles, de más a menos estricto:
# 1. **Duplicado exacto** en todas las columnas originales (incluido el NPN).
# 2. **Mismo NPN** (varias unidades físicas registradas bajo un mismo número
#    predial).
# 3. **Casi-duplicado:** mismas características físicas y mismo edificio, pero
#    distinta fila (apartamentos idénticos en serie).

# %%
# Duplicados en tres niveles de exigencia. Sirve para distinguir errores de carga
# (duplicado exacto) de unidades reales repetidas que hay que agrupar al partir.
# Columnas originales del CSV: se excluye 'id_fila' porque es única por construcción.
cols_orig = [c for c in crudo.columns if c != "id_fila"]
cols_fisicas = ["area_construida", "total_habitaciones", "total_banios", "total_plantas",
                "anio_construccion", "uso", "tipo_vivienda"]
dup = pd.DataFrame({
    "filas": [
        # Nivel 1: filas idénticas en todas las columnas originales (se cuenta la copia).
        int(df.duplicated(subset=[c for c in cols_orig if c in df]).sum()),
        # Nivel 2: keep=False marca TODAS las filas que comparten NPN, no solo la segunda.
        int(df["numero_predial_nacional"].duplicated(keep=False).sum()),
        # Nivel 3: mismo edificio y mismas características físicas (apartamentos en serie).
        int(df.duplicated(subset=["npn_edificio"] + cols_fisicas, keep=False).sum()),
    ]}, index=["1. duplicado exacto (todas las columnas)",
               "2. NPN repetido (varias unidades por predio)",
               "3. casi-duplicado (mismo edificio + mismas características)"])
dup["% del total"] = 100 * dup["filas"] / len(df)
dup

# %% [markdown]
# ```{admonition} Interpretación y decisión
# :class: note
# - El bug de uniones muchos-a-muchos que producía ~29 % de filas repetidas ya
#   se corrigió en `01_descarga_predios.py` (bitácora 3.1): con los datos
#   reales el nivel 1 es **0 filas**.
# - Nivel 2: **24 180 filas (7.2 %)** comparten NPN con otra fila (varias
#   construcciones registradas bajo un mismo predio). Nivel 3: **127 815 filas
#   (38.3 %)** son casi-duplicados: apartamentos en serie con idénticas
#   características dentro del mismo edificio.
# - Los niveles 2 y 3 **no son errores**: son unidades reales distintas
#   (apartamentos idénticos de un mismo proyecto, dos construcciones en un mismo
#   lote). Eliminarlas borraría viviendas reales. **Decisión:** se conservan,
#   pero se tratan como una misma *entidad* al partir los datos (bloques
#   espaciales, capítulo 2), para que un apartamento no quede en
#   entrenamiento y su "gemelo" en prueba.
# ```

# %% [markdown]
# ### 1.7.3 Valores imposibles o inconsistentes
#
# Rangos de dominio (no aprendidos de los datos). Los valores fuera de rango se
# convierten en `NaN` y se marcan con una bandera; **no se elimina la fila**.

# %%
# Tabla generada por L.limpiar: por variable, el rango plausible de dominio
# (definido en config, no aprendido de los datos) y cuántos valores cayeron fuera.
tabla_imposibles

# %%
# Años de construcción fuera de rango en el CSV CRUDO (antes del embudo), con su
# frecuencia. Si un valor imposible se repite mucho, es un código centinela, no ruido.
anio_crudo = pd.to_numeric(crudo["anio_construccion"], errors="coerce")
top_raros = anio_crudo[(anio_crudo < C.ANIO_MIN_PLAUSIBLE) | (anio_crudo > C.ANIO_MAX_PLAUSIBLE)]
print("Años fuera de rango más frecuentes (crudo):")
print(top_raros.value_counts().head(10).to_string())

# %% [markdown]
# ```{admonition} Hallazgo: códigos centinela en el año y en el piso
# :class: important
# En el CSV crudo, el año `2500` aparece **809 veces**, `1512` **200 veces** y
# `20` una vez. Un año imposible repetido de forma idéntica no es un error de
# digitación aleatorio: es un **código centinela** del sistema de origen para
# "año desconocido". No hay ningún dato real entre el año 20 y 1899; el primer
# año real es 1900, de ahí el límite inferior (bitácora 3.2). La mayoría de
# esas filas pertenecen a garajes, depósitos o predios sin estrato y salen en
# los pasos 1–3 del embudo; dentro del dataset de trabajo quedan **129 años
# imposibles (0.04 %)**, que se tratan como **faltantes**, no como años.
#
# `planta_ubicacion` muestra el mismo patrón: en la primera ejecución su
# máximo era exactamente **99** y coincidía con el percentil 99.9, mientras
# que el percentil 99 era 15. Un piso 99 no existe en Barranquilla: se trata
# como código y se acota al rango [1, 40] (el mismo techo que
# `total_plantas`). Afecta a **434 filas (0.13 %)**; tras la regla, el piso
# máximo es 40 y el p99.9 es 20 (cap. 3).
#
# En habitaciones (61 casos), baños (48), plantas (5) y área construida (3),
# los extremos se concentran en registros `NPH` con uso de apartamentos: lo
# más probable es que representen **edificios completos no subdivididos en
# propiedad horizontal**, no un apartamento. Son muy pocos (≤ 0.02 % por
# variable), así que se convierten en NaN y se imputan dentro del Pipeline,
# siguiendo el precedente de De Cock (2011) de documentar y tratar de forma
# simple los extremos marginales. Ninguna fila se elimina en este paso: en
# total, **646 filas** tienen al menos un valor imposible (paso 4 del
# embudo), dos tercios de ellas por el piso 99.
# ```

# %% [markdown]
# **Categorías mal escritas / inconsistentes.** `src/limpieza.py` normaliza
# espacios y guiones bajos (`"No Aplica"` → `"No_Aplica"`). Con los datos
# reales, solo `estrato` tenía variantes (9 categorías crudas → 8
# normalizadas). `destinacion_economica` tiene 18 categorías en el crudo, pero
# en el dataset de trabajo es casi constante ("Habitacional", cap. 5). Conteo de
# variantes en el crudo:

# %%
# Categorías mal escritas: compara cuántas categorías hay antes y después de la
# normalización de texto de limpieza.py. Si el número baja, había variantes
# ("No Aplica" vs "No_Aplica") que se habrían tratado como niveles distintos.
for col in ["estrato", "condicion_predio", "tipo_planta", "destinacion_economica"]:
    crudas = crudo[col].astype("string").str.strip()               # sin espacios extremos
    normalizadas = crudas.str.replace(r"\s+", "_", regex=True)     # espacios internos -> "_"
    print(f"{col:<24} categorías crudas: {crudas.nunique():>3}  ->  normalizadas: {normalizadas.nunique():>3}")

# %% [markdown]
# **Unidades.** Las áreas están en m² (servicio catastral); la `altura` debe
# verificarse (metros o número de pisos) mirando su distribución en el
# capítulo 5.

# %% [markdown]
# ### 1.7.4 Sesgos de muestreo y representatividad
#
# El catastro no es una muestra sino (casi) un censo de los predios formales,
# pero tiene dos sesgos conocidos que se cuantifican en el capítulo 3:
#
# 1. **Predios informales sin coordenadas** (16.2 % de las filas de trabajo): no
#    tienen terreno propio ni edificio matriz, así que no se pueden ubicar ni
#    asignar a un bloque espacial. Quedan fuera del modelo, y como son
#    mayoritariamente de estratos bajos, el modelo representa sobre todo la
#    ciudad **formal**.
# 2. **Predios sin estrato residencial** (`No_Aplica`/`Otro`/nulo: 12 218
#    filas, 3.2 % del crudo): excluidos por ser la variable objetivo.
#
# Además, la información catastral puede estar desactualizada respecto a la
# realidad física (reformas no declaradas), lo que añade ruido en las
# predictoras.

# %% [markdown]
# ## 1.8 Consideraciones éticas
#
# - **Datos personales:** el dataset no trae nombres ni cédulas de
#   propietarios. Sin embargo, el **número predial** y las **coordenadas**
#   identifican un inmueble concreto, y con ellos se puede consultar
#   información registral: hay **riesgo de reidentificación** indirecta.
# - **Mitigación aplicada en el libro:** el NPN se muestra truncado; los mapas
#   se presentan agregados (hexbin, grilla) o con puntos pequeños sin
#   etiquetas; no se publica la tabla fila a fila con coordenadas.
# - **Uso del estrato:** es una etiqueta socioeconómica. Un modelo que la
#   predice podría usarse para discriminar (p. ej. en crédito o seguros). El
#   propósito aquí es analítico y de apoyo a la política pública; se declara
#   explícitamente que el modelo **no** debe usarse para decisiones sobre
#   personas.
# - **Sesgo de cobertura:** los asentamientos informales quedan
#   sub-representados; cualquier conclusión aplica a la ciudad formal.

# %% [markdown]
# ## 1.9 Resumen del capítulo
#
# ```{admonition} Resumen
# :class: note
# - De 382 597 filas descargadas quedan **333 871 unidades de vivienda**
#   (−36 497 garajes/depósitos, −11 sin área, −12 218 sin estrato
#   residencial), muy por encima del mínimo de 20 000. La precisión, sin
#   embargo, está gobernada por los edificios (ICC = 0.991).
# - No hay duplicados exactos; las unidades repetidas (7.2 % por NPN, 38.3 %
#   casi-duplicados) son viviendas reales y se manejan agrupando por
#   edificio/bloque al partir.
# - Los valores imposibles son muy pocos (≤ 0.04 % por variable, salvo el
#   código 99 del piso, que afecta al 0.13 %), incluyen códigos centinela
#   (año 2500/1512, piso 99) y se tratan como faltantes a imputar dentro del
#   Pipeline.
# - Riesgos éticos: reidentificación por NPN + coordenadas y uso
#   discriminatorio del estrato; se mitigan con agregación y declarando el
#   alcance.
# ```
