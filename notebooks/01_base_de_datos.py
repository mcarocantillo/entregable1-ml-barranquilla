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
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# ```{admonition} Qué hace este capítulo
# :class: tip
# **Pregunta que responde:** ¿qué datos tenemos, cuántos son, qué problemas
# de calidad traen y con cuáles filas se puede trabajar?
#
# **Recorrido:** primero se plantea el problema (1.1) y la fuente (1.2).
# Luego se explica cómo está organizada la información del catastro, que es
# lo que hace falta para entender todo lo demás (1.3–1.4). Con eso claro se
# limpian los datos paso a paso (1.5) y se revisa la calidad que queda
# (1.6). Después se toma la decisión más importante del capítulo, qué hacer
# con las filas sin coordenadas (1.7), y se mide el tamaño de la muestra
# (1.8) y cuánta información independiente contiene (1.9). Se cierra con
# los sesgos (1.10), la ética (1.11) y un resumen (1.12).
#
# **Cubre:** la Sección 1 de las instrucciones del proyecto (problema,
# fuente, diccionario, estructura, tamaño de muestra, calidad de datos y
# ética).
# ```
#
# ```{important}
# Todo lo que se hace en este capítulo son **reglas fila a fila**: cada regla
# mira una sola fila y decide sin consultar las demás. Por ejemplo, "un año de
# construcción de 1512 es imposible" vale igual para cualquier vivienda, sin
# importar cómo sean las otras. Como estas reglas no **aprenden** nada del
# conjunto de datos, pueden aplicarse **antes** de separar el conjunto de
# prueba (capítulo 2) sin hacer trampa (sin *fuga de información*, ver la
# Guía de lectura). Lo que sí aprende de los datos (medianas para imputar,
# umbrales de outliers, escalado) se calcula después y solo con el conjunto de
# entrenamiento.
#
# Hay **una excepción deliberada**: las viviendas sin coordenadas reciben el
# centroide de su **manzana**, calculado con las demás viviendas ubicadas de
# esa manzana (paso 6, sección 1.7.2). Ese paso sí mira otras filas, pero solo
# su **posición**, nunca su estrato: equivale a leer dónde está la manzana en
# la cartografía, un dato que existiría igual para un predio nuevo.
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
# *Un ejemplo para fijar la idea.* Imaginemos dos viviendas. La primera es una
# casa de 55 m², con 2 habitaciones y 1 baño, en el suroccidente de la ciudad.
# La segunda es un apartamento de 150 m², con 3 habitaciones y 3 baños, en el
# piso 12 de una torre del norte. Sin conocer su estrato, cualquier
# barranquillero apostaría a que la primera es de estrato bajo y la segunda de
# estrato alto. Este proyecto convierte esa intuición en un modelo
# estadístico: le damos al modelo la "ficha" de la vivienda y su ubicación, y
# él propone un estrato. Después medimos qué tan seguido acierta.
#
# **Tipo de tarea.** Clasificación supervisada multiclase: hay 6 clases
# (estratos 1 a 6) y son **ordinales**, es decir, tienen un orden (equivocarse
# de 5 a 6 es menos grave que de 1 a 6). Los datos tienen coordenadas pero no
# tiempo. Según la Figura 1 de las instrucciones del proyecto, se sigue la
# **ruta A (clasificación)**, añadiendo la sección 2.7 (componente espacial) y
# una partición por **bloques espaciales** (capítulo 2).
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
# | Licencia | Publicado como **dato abierto** por la Alcaldía de Barranquilla (Gerencia de Gestión Catastral) en su portal de datos geográficos. El servicio no declara una licencia explícita en sus metadatos; se usa como información pública con fines académicos y se cita la fuente. No se redistribuyen datos de personas: el NPN y las coordenadas se tratan como se explica en la sección 1.11. |
#
# **Por qué este dataset.** (i) Es información oficial, completa para la zona
# urbana formal (no es una encuesta), (ii) supera ampliamente el mínimo de
# 20 000 observaciones, (iii) combina variables físicas, categóricas y
# espaciales, lo que permite un EDA completo (uni, bi, multivariado y
# espacial) y (iv) es el mismo insumo de la tesis.

# %% [markdown]
# ## 1.3 Cómo está organizada la información del catastro
#
# Antes de contar filas o limpiar datos hay que entender **qué es una fila** y
# cómo se relacionan las filas entre sí. Casi todas las decisiones del libro
# (cómo partir los datos, cómo medir la incertidumbre, por qué las filas no son
# independientes) dependen de esta sección.
#
# ### Tres unidades de conteo: vivienda, predio y predio matriz
#
# | Unidad | Qué es | Cómo se identifica | Ejemplo |
# |---|---|---|---|
# | **Vivienda** (una fila) | una casa o un apartamento concreto | una fila de la capa 505 "Características" | el apartamento 802 de la torre B |
# | **Predio** | la unidad jurídica que registra el catastro | el **número predial nacional (NPN)** completo, de 30 dígitos | el apartamento 802 tiene su propio NPN |
# | **Predio matriz** | el lote con lo que está construido encima: una casa, un edificio o un conjunto residencial | los **primeros 22 dígitos** del NPN | el conjunto residencial completo, con todas sus torres |
#
# *Ejemplo.* Un conjunto residencial con 200 apartamentos son **200 viviendas**
# (200 filas) y **200 predios** (cada apartamento tiene su NPN de 30 dígitos),
# pero un **solo predio matriz**, porque todos comparten los primeros 22
# dígitos. Una casa sola, en cambio, es a la vez 1 vivienda, 1 predio y 1
# predio matriz. En el resto del libro, cuando se habla de "edificio" (por
# ejemplo, en la partición del capítulo 2 o en la columna `npn_edificio`) se
# hace referencia a este predio matriz.
#
# ### El número predial nacional (NPN), dígito por dígito
#
# El NPN es un código de 30 dígitos definido por el IGAC. Leído de izquierda a
# derecha, va de lo más general (el departamento) a lo más particular (la
# unidad dentro del edificio):
#
# | Dígitos | Campo | Qué indica |
# |---|---|---|
# | 1–2 | departamento | 08 = Atlántico |
# | 3–5 | municipio | 001 = Barranquilla |
# | 6–7 | zona | urbana o rural |
# | 8–9 | sector | |
# | 10–11 | comuna | |
# | 12–13 | barrio | |
# | 14–17 | manzana | **los primeros 17 dígitos identifican la manzana** |
# | 18–21 | terreno | el lote dentro de la manzana |
# | 22 | condición de propiedad | 9 = propiedad horizontal (PH); 5 = mejoras en terreno ajeno, etc. |
# | 23–24 | edificio o torre | |
# | 25–26 | piso | |
# | 27–30 | unidad predial | el apartamento o local dentro del piso |
#
# Por eso el **predio matriz** se define con los primeros 22 dígitos (hasta la
# condición de propiedad): todas las unidades de un mismo lote con la misma
# condición comparten ese prefijo.
#
# **Una excepción importante: la condición 5 (mejoras en terreno ajeno).** Una
# *mejora* es una construcción levantada sobre un terreno que pertenece a otra
# persona, por ejemplo una casa construida en un lote ajeno, algo frecuente en
# la vivienda informal. En el catastro de Barranquilla, muchas mejoras
# **distintas e independientes** quedan registradas bajo el mismo prefijo de 22
# dígitos, aunque no forman un edificio. Cuando haga falta agrupar viviendas
# "del mismo edificio" (sección 1.9), cada mejora se trata como su propio
# grupo. El código es corto:
#
# ```python
# es_mejora = npn.str[21] == "5"                  # dígito 22 (Python cuenta desde 0)
# grupo = npn.str[:22].where(~es_mejora, npn)     # predio matriz, salvo las mejoras
# ```
#
# ### Estructura de los datos
#
# - **Unidad de observación:** la *unidad residencial física* (una fila = una
#   casa o un apartamento). Cada fila se une a su unidad de construcción, a su
#   predio y al centroide (punto central) de su terreno.
# - **Jerarquía:** vivienda ⊂ predio (NPN de 30 dígitos) ⊂ predio matriz
#   (22 dígitos) ⊂ manzana (17 dígitos) ⊂ barrio. Como varias viviendas
#   comparten predio matriz, manzana y barrio, **las filas no son
#   independientes**: esta es la razón para partir los datos por grupos y
#   bloques y no por filas sueltas (capítulo 2).
# - **Tipo de datos:** **transversal con componente espacial**. Es una foto
#   del catastro en la fecha de descarga: cada vivienda aparece una sola vez y
#   no hay mediciones repetidas en el tiempo. `anio_construccion` es un
#   atributo de la vivienda, no la fecha de la observación, así que no aplican
#   las secciones 2.6 (componente temporal) ni 2.8 (espacio-temporal) de las
#   instrucciones.

# %% [markdown]
# Se cargan los datos crudos y se aplican las reglas de limpieza, que se
# explican una por una en la sección 1.5. Aquí solo se muestra el tamaño del
# archivo y una vista de las primeras filas. Por la **consideración ética** de
# la sección 1.11, el número predial se muestra truncado (el NPN completo
# identifica un inmueble concreto).

# %%
# Carga del CSV crudo y aplicación de las reglas deterministas de limpieza.
# cargar_crudo lee el CSV (NPN como texto para no perder ceros) y añade 'id_fila'.
crudo = L.cargar_crudo()
# Se resta 1 a las columnas porque 'id_fila' la agrega el pipeline, no viene del CSV.
print(f"Archivo: {C.ruta_csv().name}  |  filas: {len(crudo):,}  |  columnas: {crudo.shape[1] - 1}")
# limpiar devuelve: el dataset de trabajo, el embudo (filas que quedan tras cada
# regla y su motivo) y la tabla de valores físicamente imposibles o incoherentes
# convertidos en NaN (pasos 4 y 5). Las reglas se explican en la sección 1.5.
df, embudo, tabla_imposibles = L.limpiar(crudo, verbose=False)
print(f"Filas tras la limpieza: {len(df):,}")


# Vista de las primeras filas con el número predial enmascarado (consideración
# ética 1.11: el NPN completo identifica un inmueble concreto y permite reidentificar).
def enmascarar_npn(s):
    # Conserva solo los 12 primeros dígitos (depto, municipio, zona, sector...),
    # que ubican la zona pero no el predio individual.
    return s.astype("string").str[:12] + "…"


# Se omiten las columnas de bandera (flag_*) para que la vista sea legible.
vista = df.drop(columns=df.filter(like="flag_").columns).head(5).copy()
vista["numero_predial_nacional"] = enmascarar_npn(vista["numero_predial_nacional"])
vista["npn_edificio"] = enmascarar_npn(vista["npn_edificio"])
vista["npn_manzana"] = enmascarar_npn(vista["npn_manzana"])
vista.T  # transpuesta: una fila por variable, más fácil de leer con muchas columnas

# %% [markdown]
# ## 1.4 Diccionario de variables
#
# Cada variable tiene un **rol**: la **objetivo** es la que se quiere predecir
# (el estrato); las **predictoras** son las que el modelo usa para predecir;
# los **identificadores** sirven para saber a qué vivienda corresponde cada
# fila, pero nunca entran al modelo (el modelo podría memorizarlos en lugar
# de aprender); los **metadatos** describen cómo se construyó el dato. Que
# ninguna variable con rol de identificador o metadato se cuele en el modelo se
# verifica en la auditoría de fuga (capítulo 9). Si el servidor responde, se
# descargan también los alias y dominios oficiales de ArcGIS (por ejemplo, el
# significado de los códigos de `tipo_vivienda`).

# %%
# Diccionario de variables: nombre, tipo, unidad, significado y ROL en el modelo.
# El rol (objetivo, predictora, identificador, metadato) es lo que luego se audita
# en el capítulo 9 para asegurar que ninguna columna con fuga entre al modelo.
dicc = pd.DataFrame([
    ("numero_predial_nacional", "texto (30 dígitos)", "—", "Número predial nacional (IGAC): depto, municipio, zona, sector, comuna, barrio, manzana, terreno, condición, edificio, piso, unidad (sección 1.3)", "identificador"),
    ("npn_edificio", "texto (22 dígitos)", "—", "Prefijo del NPN que identifica el predio matriz: edificio, conjunto o lote (derivada)", "identificador / grupo"),
    ("npn_manzana", "texto (17 dígitos)", "—", "Prefijo del NPN que identifica la manzana (derivada; sirve para ubicar viviendas sin coordenadas)", "identificador"),
    ("estrato", "categórica ordinal (texto)", "—", "Estrato socioeconómico tal como llega de ArcGIS (Bajo_Bajo_1 … Alto_6, No_Aplica, Otro)", "origen del objetivo"),
    ("estrato_num", "categórica ordinal", "1–6", "Estrato numérico extraído del texto", "OBJETIVO"),
    ("area_construida", "numérica continua", "m²", "Área construida de la unidad", "predictora"),
    ("area_catastral_terreno", "numérica continua", "m²", "Área del terreno del predio (en unidades PH, su cuota del lote común)", "predictora"),
    ("total_habitaciones", "numérica discreta", "conteo", "Número de habitaciones", "predictora"),
    ("total_banios", "numérica discreta", "conteo", "Número de baños", "predictora"),
    ("total_plantas", "numérica discreta", "conteo", "Número de plantas/pisos de la construcción", "predictora"),
    ("anio_construccion", "numérica discreta", "año", "Año de construcción registrado en el catastro", "origen de antigüedad"),
    ("antiguedad", "numérica discreta", "años", f"{C.ANIO_ACTUAL} − año de construcción (derivada)", "predictora"),
    ("planta_ubicacion", "numérica discreta", "piso", "Piso en que está la unidad", "predictora"),
    ("altura", "numérica continua", "m (verificar)", "Altura de la unidad de construcción", "predictora"),
    ("tipo_vivienda", "categórica nominal (código)", "—", "Código de dominio ArcGIS del tipo de vivienda", "predictora"),
    ("uso", "categórica nominal", "—", "Uso catastral de la unidad (casa, apartamento en PH, barraca…)", "predictora"),
    ("condicion_predio", "categórica nominal", "—", "Régimen: NPH (no propiedad horizontal), PH_Matriz, PH_Unidad_Predial, Informal…", "predictora"),
    ("destinacion_economica", "categórica nominal", "—", "Destinación económica del predio", "predictora"),
    ("tipo_planta", "categórica nominal", "—", "Tipo de planta (Piso, Sótano, Mezanine…)", "predictora"),
    ("centroide_lat / centroide_lon", "numérica continua", "grados (WGS84, EPSG:4326)", "Centroide del terreno del predio (o de su predio matriz; o de su manzana si no tiene ninguno de los dos)", "espacial"),
    ("x_km / y_km", "numérica continua", "km", "Coordenadas proyectadas localmente, origen en el centro histórico (derivada)", "predictora espacial"),
    ("dist_centro_km", "numérica continua", "km", "Distancia al centro histórico (derivada)", "predictora espacial"),
    ("centroide_fuente", "categórica nominal", "—", "Cómo se obtuvo el centroide: directo / propagado_matriz / centroide_manzana / sin_dato (metadato del pipeline)", "metadato (excluida)"),
    ("ubicado_por_manzana", "booleana", "—", "True si la coordenada es el centroide de la manzana (paso 6 del embudo)", "metadato (excluida)"),
    ("flag_imposible_*", "booleana", "—", "Marca si el valor original era físicamente imposible y se convirtió en NaN", "auditoría"),
    ("flag_incoherente_*", "booleana", "—", "Marca si el valor era incoherente con el resto de la fila (baños o habitaciones) y se convirtió en NaN", "auditoría"),
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
# ## 1.5 Limpieza: las reglas y el embudo de filas
#
# El CSV descargado trae filas que no sirven para la pregunta (garajes,
# predios sin estrato) y valores que no pueden ser ciertos (un año 2500). La
# limpieza se hace con **reglas fijas, definidas de antemano** en
# `src/config.py`, que se aplican en orden. Hay dos tipos de regla:
#
# - las que **eliminan la fila**, porque la fila no describe una vivienda con
#   estrato (pasos 1 a 3), y
# - las que **conservan la fila** pero anulan un valor concreto (lo pasan a
#   `NaN`, es decir, "dato faltante") y dejan una bandera que recuerda que se
#   anuló (pasos 4 y 5). Así no se pierde el resto de la información de la
#   vivienda; el valor faltante se rellena después, dentro del modelo y solo
#   con el conjunto de entrenamiento (capítulos 3 y 10).
#
# Un último paso (6) tampoco elimina filas: ubica por el centroide de su
# manzana a las viviendas que llegan sin coordenadas (sección 1.7.2).
#
# ### 1.5.1 Las reglas, una por una
#
# | Paso | Regla | Qué hace con la fila | Por qué | Ejemplo |
# |---|---|---|---|---|
# | 1 | Excluir usos no habitables | elimina | un garaje o un depósito no es un hogar y no tiene estrato propio | `Residencial_Garajes_En_PH` |
# | 2 | Exigir área construida > 0 | elimina | sin área no hay vivienda que describir | área vacía o 0 |
# | 2b | Excluir unidades de menos de 10 m² | elimina | casi siempre son parqueaderos o depósitos registrados como vivienda | un "apartamento" de 6 m² sin baño |
# | 2c | Excluir registros agregados | elimina | la fila describe un edificio completo, no una vivienda | una fila con 684 habitaciones y 456 baños |
# | 3 | Excluir estrato no residencial o vacío | elimina | el estrato es la respuesta que se quiere predecir: no se puede inventar | `No_Aplica`, `Otro`, vacío |
# | 4 | Valores imposibles → `NaN` | conserva | el valor es un error o un código, no un dato | año de construcción 2500 |
# | 5 | Incoherencias dentro de la fila → `NaN` | conserva | cada valor por separado es posible, pero juntos no | 82 baños en un apartamento de 2 habitaciones y 82 m² |
# | 6 | Sin coordenadas → centroide de su manzana | conserva | la manzana está en el NPN y casi siempre tiene otras viviendas ubicadas | una mejora en terreno ajeno sin terreno propio |
#
# Los umbrales de los pasos 2b, 2c, 4 y 5 salieron de un **diagnóstico de
# rangos** (`diagnostico_rangos.py`, bitácora A.10; reporte en
# `diagnostico_rangos_reporte.txt`) que se hizo mirando **solo el conjunto de
# entrenamiento**, para no usar el conjunto de prueba al definir las reglas.
# Las cifras que aquí se citan "del diagnóstico previo" vienen de ese reporte;
# el efecto de las reglas sobre todo el dataset es el que muestra el embudo.
#
# ### 1.5.2 El embudo de filas
#
# El embudo muestra cuántas filas quedan después de cada regla. Sirve para
# juzgar si la limpieza deja fuera demasiado, o a un grupo concreto de
# viviendas.

# %%
# Embudo: filas que sobreviven a cada regla, cambio respecto al paso anterior y
# % del CSV original. Los pasos 4 y 5 no eliminan filas (cambio = 0).
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
# ```{admonition} Lectura del embudo
# :class: note
# - De las **382 597** filas descargadas quedan **332 718** (87.0 %).
# - La mayor pérdida es el paso 1: **36 497** garajes y depósitos (9.5 % del
#   crudo). Le sigue el paso 3: **12 093** filas sin estrato residencial
#   (3.2 %).
# - Las reglas de alcance retiran **1 044** unidades de menos de 10 m² (paso 2b)
#   y **234** registros agregados (paso 2c), el 0.3 % del crudo entre las dos.
#   El paso 2 solo retira 11 filas sin área.
# - Los pasos 4 y 5 **no eliminan filas** (cambio = 0): **484 filas** tienen al
#   menos un valor imposible y **396** una incoherencia. Esos valores pasan a
#   `NaN` y la fila se conserva.
# ```
#
# ### 1.5.3 Valores imposibles y códigos centinela (paso 4)
#
# Para cada variable numérica se fija un **rango plausible de dominio**, es
# decir, un rango que viene del conocimiento del problema y no de los datos
# (por ejemplo, ninguna vivienda de Barranquilla tiene más de 60 pisos). Un
# valor fuera de rango se convierte en `NaN` y se marca con una bandera
# `flag_imposible_*`. Las dos últimas filas de la tabla son las reglas de
# coherencia del paso 5, que se explican en 1.5.4.

# %%
# Tabla generada por L.limpiar: por variable, el rango plausible de dominio
# (definido en config, no aprendido de los datos) y cuántos valores cayeron fuera.
tabla_imposibles

# %% [markdown]
# Un caso especial son los **códigos centinela**: valores imposibles que se
# repiten muchas veces de forma idéntica. Un error de digitación al azar
# produciría valores dispersos; un mismo valor imposible repetido cientos de
# veces es un código que el sistema de origen usa para decir "no sé". La celda
# siguiente lista los años fuera de rango más frecuentes en el CSV crudo.

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
# los pasos previos del embudo; dentro del dataset de trabajo quedan **129
# años imposibles (0.04 %)**, que se tratan como **faltantes**, no como años.
#
# `planta_ubicacion` muestra el mismo patrón: los pisos 97, 98 y 99 no existen
# en Barranquilla y se tratan como código. El rango plausible es [1, 60]: el
# techo queda por encima de los pisos reales más altos y por debajo de los
# códigos. La regla anula **351 valores (0.11 %)** del dataset de trabajo; en
# entrenamiento, tras la regla, el piso máximo es 41 (capítulo 3), es decir,
# las torres altas reales se conservan.
#
# En la tabla de rangos, habitaciones, baños y área construida tienen **0
# valores imposibles**, y `total_plantas` solo 4: sus extremos ya no se anulan
# valor a valor, porque las filas que los tenían se excluyeron como registros
# agregados en el paso 2c (sección 1.5.4).
# ```
#
# ### 1.5.4 ¿Cada fila es una vivienda? Reglas de alcance (2b, 2c) e incoherencias (5)
#
# Un rango por variable no basta. El diagnóstico de rangos mostró que los
# valores extremos de habitaciones, baños y área no eran errores de digitación
# aislados, sino **filas que no describen una vivienda**:
#
# - **Registros agregados (paso 2c).** Filas `NPH` que describen un edificio
#   completo: en el diagnóstico previo, la más extrema tenía 684 habitaciones,
#   456 baños, 19 pisos y 52 168 m². Se excluyen las filas con más de 20
#   habitaciones, más de 15 baños o más de 2 000 m², igual que los garajes y
#   depósitos.
# - **Unidades diminutas (paso 2b).** En el diagnóstico previo había 215 filas
#   de menos de 5 m² y 656 de menos de 10 m², en todos los estratos y a menudo
#   con 0 baños o con el piso "99" (un código). Son, sobre todo, parqueaderos o
#   depósitos registrados con uso residencial. Se excluyen con un área mínima
#   de 10 m². No se usó un umbral mayor (15 o 20 m²) para no sacar viviendas
#   mínimas reales de uno o dos cuartos, que son las de los hogares más
#   pobres: un sesgo así afectaría justamente a los estratos 1 y 2.
# - **Piso de ubicación.** Los valores 97, 98 y 99 son códigos (299 filas en
#   el diagnóstico previo), pero hay pisos **reales** hasta el 41: 41
#   apartamentos en PH de estratos 5 y 6 entre los pisos 30 y 41. Por eso el
#   techo de `planta_ubicacion` se fija en 60 (y no en 40, como el de
#   `total_plantas`): deja pasar esas torres y sigue atrapando los códigos.
# - **Incoherencias dentro de la fila (paso 5).** Valores que por separado
#   están en rango pero son imposibles dado el resto de la fila: por ejemplo,
#   en el diagnóstico previo apareció un apartamento de 2 habitaciones y
#   82 m² con 82 baños. Las reglas son: más baños que habitaciones + 3, o
#   menos de 6 m² por habitación. Se anula el valor sospechoso (`NaN` +
#   bandera `flag_incoherente_*`) y la fila se conserva.
#
# La celda siguiente describe quiénes salen por las reglas 2b y 2c (por
# condición del predio, uso y estrato), para verificar que no afectan a un solo
# tipo de vivienda.

# %%
# Quiénes salen por las reglas de alcance 2b (área mínima) y 2c (registro
# agregado): se reproducen sobre el crudo tras los pasos 1 y 2 para describirlos por
# condición del predio, uso y estrato. Es una tabla descriptiva: los umbrales
# están fijados en config.py y no se ajustan aquí.
base = crudo.copy()
for col in ["uso", "condicion_predio", "estrato"]:
    base[col] = L._normalizar_texto(base[col])
base = base[~base["uso"].isin(C.USOS_NO_HABITABLES)]
area_b = pd.to_numeric(base["area_construida"], errors="coerce")
base = base[area_b > 0].copy()
base["estrato_num"] = L.extraer_estrato_numerico(base["estrato"])
for col in C.UMBRALES_REGISTRO_AGREGADO:
    base[col] = pd.to_numeric(base[col], errors="coerce")
regla_2b = base["area_construida"] < C.AREA_MIN_VIVIENDA
regla_2c = ~regla_2b & pd.concat([base[c] > u for c, u in C.UMBRALES_REGISTRO_AGREGADO.items()], axis=1).any(axis=1)
for nombre, m in [(f"2b: área < {C.AREA_MIN_VIVIENDA} m²", regla_2b), ("2c: registro agregado", regla_2c)]:
    sub = base[m]
    print(f"\n{nombre}: {int(m.sum()):,} filas")
    print("  condición del predio:", sub["condicion_predio"].value_counts().head(5).to_dict())
    print("  uso:", sub["uso"].value_counts().head(4).to_dict())
    print("  estrato (1-6; NaN = sin estrato residencial):",
          {(int(k) if pd.notna(k) else "NaN"): int(v)
           for k, v in sub["estrato_num"].value_counts(dropna=False).sort_index().items()})
    print("  área construida: mediana", round(sub["area_construida"].median(), 1),
          "| habitaciones: mediana", sub["total_habitaciones"].median())

# %% [markdown]
# ```{admonition} Lectura
# :class: note
# - **Regla 2b (área < 10 m²): 1 044 filas**, con una mediana de 6 m². Están
#   repartidas en todos los regímenes (400 `PH_Unidad_Predial`, 354 `NPH`, 278
#   `Informal`) y en todos los estratos (294 de estrato 1, 253 de 2, 270 de 3,
#   138 de 4, 36 de 5, 6 de 6 y 47 sin estrato residencial). No es una regla
#   que afecte solo a la vivienda informal ni a un solo estrato.
# - **Regla 2c (registros agregados): 234 filas**, sobre todo `NPH` (146) y con
#   uso de apartamentos de 4 y más pisos (112), con una mediana de 1 161 m² y
#   24 habitaciones. 78 de ellas tampoco tenían estrato residencial.
# - **Paso 5 (incoherencias): 396 filas (0.12 %)**: 355 con menos de 6 m² por
#   habitación y 41 con más baños que habitaciones + 3 (dos últimas filas de la
#   tabla de 1.5.3). Siguiendo el precedente de De Cock (2011) de documentar y
#   tratar de forma simple los casos marginales, esos valores se convierten en
#   `NaN` y se imputan dentro del Pipeline; ninguna fila se elimina en los
#   pasos 4 y 5.
# ```
#
# **¿Por qué no usar la regla de Tukey para limpiar?** La regla de Tukey marca
# como *outlier* todo valor que se aleje mucho del centro de la distribución
# (se explica en el capítulo 3). Las notas de clase (9.10.4.1.2) la proponen
# para **identificar** outliers, y así se usa en el capítulo 3. Pero en
# variables de cola larga marca miles de viviendas reales: en el diagnóstico
# previo señalaba 10 580 filas en habitaciones y 13 009 en área. Una casa de
# 400 m² en el norte es poco común, pero existe. Por eso aquí se separan dos
# conceptos:
#
# - **Imposible** (error, código o fila que no es vivienda): se trata con
#   reglas de dominio fijas, fila a fila y antes de la partición.
# - **Extremo pero posible** (una casa grande de estrato 6): se conserva, y su
#   influencia en el modelo se limita con una *winsorización* (recortar los
#   valores por encima del percentil 99.9) aprendida solo con entrenamiento
#   (capítulo 3).
#
# Dos hallazgos del diagnóstico **no** se convirtieron en reglas y quedan como
# limitaciones. (i) En el diagnóstico previo había 2 014 filas con 0
# habitaciones y 7 145 con 0 baños: pueden ser apartaestudios o viviendas con
# baño compartido, o un "dato no registrado"; sin el diccionario del catastro
# no se puede decidir, así que se conservan. (ii) El año de construcción está
# **amontonado** en ciertos valores (en ese mismo diagnóstico, 1986 aparecía
# en 27 280 filas, el 12 %; también 1932, 1940 y 1942), lo que sugiere años
# por defecto o aproximados del catastro. La antigüedad debe leerse con esa
# cautela (ver también 11.2).
#
# ### 1.5.5 Categorías mal escritas y unidades
#
# `src/limpieza.py` normaliza espacios y guiones bajos, para que
# `"No Aplica"` y `"No_Aplica"` cuenten como una misma categoría y no como
# dos. La celda compara cuántas categorías hay antes y después de normalizar:
# si el número baja, había variantes mal escritas.

# %%
# Categorías mal escritas: compara cuántas categorías hay antes y después de la
# normalización de texto de limpieza.py. Si el número baja, había variantes
# ("No Aplica" vs "No_Aplica") que se habrían tratado como niveles distintos.
for col in ["estrato", "condicion_predio", "tipo_planta", "destinacion_economica"]:
    crudas = crudo[col].astype("string").str.strip()               # sin espacios extremos
    normalizadas = crudas.str.replace(r"\s+", "_", regex=True)     # espacios internos -> "_"
    print(f"{col:<24} categorías crudas: {crudas.nunique():>3}  ->  normalizadas: {normalizadas.nunique():>3}")

# %% [markdown]
# Con los datos reales, solo `estrato` tenía variantes (9 categorías crudas →
# 8 normalizadas). `destinacion_economica` tiene 18 categorías en el crudo,
# pero en el dataset de trabajo es casi constante ("Habitacional"), como se
# verá en el capítulo 5.
#
# **Unidades.** Las áreas están en m² (servicio catastral); la unidad de
# `altura` (metros o número de pisos) se verifica en el capítulo 5 mirando su
# distribución.

# %% [markdown]
# ## 1.6 Calidad de los datos que quedan: duplicados y faltantes
#
# Después de limpiar se revisan dos problemas clásicos de calidad. Cada tabla
# se presenta para los **dos universos** de filas que se definen en la
# sección 1.7: todas las filas limpias y solo las que tienen coordenadas. Por
# ahora basta saber que el segundo es el subconjunto con el que se entrena el
# modelo.
#
# ### 1.6.1 Duplicados exactos y casi-duplicados
#
# Tres niveles, de más a menos estricto:
#
# 1. **Duplicado exacto:** dos filas idénticas en todas las columnas
#    originales, incluido el NPN. Sería un error de carga.
# 2. **Mismo NPN:** varias unidades físicas registradas bajo un mismo número
#    predial (por ejemplo, dos construcciones en un mismo lote).
# 3. **Casi-duplicado:** mismas características físicas y mismo predio matriz,
#    pero distinta fila. Es el caso típico de los apartamentos en serie: el
#    101, el 201 y el 301 de una torre suelen ser iguales.

# %%
# Duplicados en tres niveles de exigencia, en los dos universos (todas las filas
# limpias y solo las que tienen coordenadas). Distingue errores de carga
# (duplicado exacto) de unidades reales repetidas que hay que agrupar al partir.
# Columnas originales del CSV: se excluye 'id_fila' porque es única por construcción.
cols_orig = [c for c in crudo.columns if c != "id_fila"]
cols_fisicas = ["area_construida", "total_habitaciones", "total_banios", "total_plantas",
                "anio_construccion", "uso", "tipo_vivienda"]
con_coord = df[df["tiene_coordenadas"]]   # universo de modelado (sección 1.7)


def contar_duplicados(d):
    # Nivel 1: filas idénticas en todas las columnas originales (se cuenta la copia).
    # Nivel 2: keep=False marca TODAS las filas que comparten NPN, no solo la segunda.
    # Nivel 3: mismo predio matriz y mismas características físicas (apartamentos en serie).
    n = [int(d.duplicated(subset=[c for c in cols_orig if c in d]).sum()),
         int(d["numero_predial_nacional"].duplicated(keep=False).sum()),
         int(d.duplicated(subset=["npn_edificio"] + cols_fisicas, keep=False).sum())]
    return pd.Series(n) , pd.Series(n) * 100 / len(d)


dup = pd.DataFrame(index=["1. duplicado exacto (todas las columnas)",
                          "2. NPN repetido (varias unidades por predio)",
                          "3. casi-duplicado (mismo predio matriz + mismas características)"])
for etiqueta, d in [("todas las filas", df), ("con coordenadas", con_coord)]:
    filas_dup, pct_dup = contar_duplicados(d)
    dup[f"filas ({etiqueta})"] = filas_dup.to_numpy()
    dup[f"% ({etiqueta})"] = pct_dup.to_numpy()
dup

# %% [markdown]
# ```{admonition} Interpretación y decisión
# :class: note
# - El error de uniones muchos-a-muchos que producía ~29 % de filas repetidas
#   ya se corrigió en `01_descarga_predios.py` (bitácora 3.1): con los datos
#   reales el nivel 1 es **0 filas** en los dos universos.
# - Nivel 2: **22 707 filas (6.8 %)** comparten NPN con otra fila (varias
#   construcciones registradas bajo un mismo predio); en el universo con
#   coordenadas son 22 047 (6.8 %).
# - Nivel 3: **127 601 filas (38.4 %)** son casi-duplicados, apartamentos en
#   serie con idénticas características dentro del mismo predio matriz. En el
#   universo con coordenadas el porcentaje casi no cambia (38.6 %, 124 392
#   filas): tras ubicar por su manzana a la mayoría de las casas informales
#   sueltas (sección 1.7.2), los dos universos tienen una composición
#   parecida.
# - Los niveles 2 y 3 **no son errores**: son viviendas reales distintas.
#   Eliminarlas borraría viviendas que existen. **Decisión:** se conservan, pero
#   se tratan como una misma *entidad* al partir los datos (capítulo 2), para
#   que un apartamento no quede en entrenamiento y su "gemelo" en prueba.
# ```
#
# ### 1.6.2 Valores faltantes (panorama)
#
# Aquí solo se **cuentan** los faltantes. Su **patrón** (qué variables faltan
# juntas) y su **mecanismo** (si faltan al azar o no) se analizan en el
# capítulo 3, usando solo entrenamiento, porque de ese análisis sale una
# decisión de modelado (cómo rellenarlos). Dos columnas separan los nulos que
# crearon las propias reglas de limpieza: "de ellos, por valor imposible"
# (banderas `flag_imposible_*`, paso 4) y "de ellos, por incoherencia"
# (banderas `flag_incoherente_*`, paso 5).

# %%
# Panorama de faltantes: solo se CUENTAN aquí (sin aprender nada), en los dos
# universos. El patrón y el mecanismo MCAR/MAR/MNAR se estudian en el cap. 3 solo con train.
cols_revisar = C.NUMERICAS + C.CATEGORICAS + ["anio_construccion", "centroide_lat", "numero_predial_nacional"]
nulos = pd.DataFrame({
    "% nulos (todas las filas)": df[cols_revisar].isna().mean().mul(100),
    "% nulos (con coordenadas)": con_coord[cols_revisar].isna().mean().mul(100),
}).sort_values("% nulos (todas las filas)", ascending=False)
# Parte de esos nulos (en todas las filas) que generaron las reglas de limpieza:
# valores imposibles (paso 4) e incoherencias dentro de la fila (paso 5); 0 si la
# variable no tiene regla.
nulos["de ellos, por valor imposible"] = [
    100 * df[f"flag_imposible_{c}"].mean() if f"flag_imposible_{c}" in df else 0.0 for c in nulos.index]
nulos["de ellos, por incoherencia"] = [
    100 * df[f"flag_incoherente_{c}"].mean() if f"flag_incoherente_{c}" in df else 0.0 for c in nulos.index]
nulos

# %% [markdown]
# ```{admonition} Interpretación
# :class: note
# - **El único faltante importante es la coordenada:** el **16.2 %** de las
#   filas limpias llega sin ubicación propia; tras ubicar por su manzana a la
#   mayoría (paso 6, sección 1.7.2), sigue sin coordenadas el **3.1 %**. Todas
#   las demás variables tienen menos de 0.13 % de nulos, en los dos
#   universos.
# - Casi todos esos nulos pequeños los crearon las propias reglas de limpieza.
#   Los de `planta_ubicacion` (0.105 %), `anio_construccion` (0.039 %) y
#   `total_plantas` (0.001 %) son valores imposibles; los de
#   `total_habitaciones` (0.107 %) y `total_banios` (0.012 %) son
#   incoherencias (355 y 41 filas, sección 1.5.4). `antiguedad` aparece con 0
#   en ambas columnas porque no tiene bandera propia: sus nulos son los del año
#   de construcción, del que se deriva.
# - En el universo con coordenadas la coordenada ya no falta (por definición) y
#   el resto de porcentajes casi no cambia.
# - El faltante de coordenadas es tan grande, y tan importante para un estudio
#   espacial, que merece una sección propia: la siguiente.
# ```

# %% [markdown]
# ## 1.7 Las filas sin coordenadas: ¿ubicarlas, quitarlas o no?
#
# Esta es la decisión más importante del capítulo, porque define con qué filas
# se trabaja en el resto del libro. Se responde en cuatro pasos: (1) quiénes
# son las filas que llegan sin coordenadas, (2) si se pueden ubicar con la
# manzana a la que pertenecen, (3) si quitar las que quedan cambia las
# estadísticas y (4) qué cifras se reportan entonces.
#
# ### 1.7.1 ¿Quiénes son las filas que llegan sin coordenadas?
#
# La coordenada de cada vivienda es el centroide del terreno de su predio (o
# de su predio matriz, si la unidad no tiene terreno propio). La columna
# `centroide_fuente` dice de dónde salió: `directo` (del terreno del propio
# predio), `propagado_matriz` (copiado del predio matriz), `centroide_manzana`
# (ubicada por su manzana, sección 1.7.2) o `sin_dato`. Aquí se describen las
# filas que **no traen** coordenadas propias: las dos últimas categorías.

# %%
# Perfil de las filas que llegan sin coordenadas propias (antes de ubicarlas por
# su manzana): de dónde salió el centroide, régimen del predio y condición de
# propiedad (dígito 22 del NPN). Explica POR QUÉ faltan.
sin_propias = df[df["centroide_fuente"].isin(["sin_dato", "centroide_manzana"])]
con_propias = df.drop(index=sin_propias.index)
print(f"Filas limpias:                   {len(df):,}")
print(f"con coordenadas propias:         {len(con_propias):,} ({100 * len(con_propias) / len(df):.1f}%)")
print(f"sin coordenadas propias:         {len(sin_propias):,} ({100 * len(sin_propias) / len(df):.1f}%)")
perfil = pd.DataFrame({
    "sin coordenadas propias": sin_propias["condicion_predio"].value_counts(),
    "con coordenadas propias": con_propias["condicion_predio"].value_counts(),
}).fillna(0).astype(int)
perfil["% del régimen con coordenadas propias"] = 100 * perfil["con coordenadas propias"] / (
    perfil["con coordenadas propias"] + perfil["sin coordenadas propias"])
print("Filas sin coordenadas propias según la condición de propiedad (dígito 22 del NPN):")
print("  ", sin_propias["numero_predial_nacional"].str[21].value_counts().to_dict())
perfil.sort_values("sin coordenadas propias", ascending=False)

# %% [markdown]
# ```{admonition} Lectura
# :class: note
# - Son **53 894 filas (16.2 %)**.
# - El **99.5 %** (53 646) tiene régimen `Informal`, y casi todas (53 623)
#   tienen **condición de propiedad 5** en el dígito 22 del NPN: son **mejoras
#   en terreno ajeno** (sección 1.3). Las 248 restantes son unidades en
#   propiedad horizontal.
# - La razón es estructural: la coordenada sale del polígono del terreno, y una
#   mejora no tiene terreno propio registrado ni un predio matriz del cual
#   heredar el centroide.
# - Visto al revés: solo el **25.7 %** de las viviendas `Informal` trae
#   coordenadas propias, frente a prácticamente el 100 % de las `NPH` y de las
#   unidades en propiedad horizontal.
# ```
#
# ### 1.7.2 Ubicarlas por su manzana
#
# El NPN de esas viviendas sí dice en qué **manzana** están (primeros 17
# dígitos, sección 1.3), y casi todas esas manzanas tienen otras viviendas con
# coordenadas. El paso 6 del embudo (sección 1.5) le asigna a cada vivienda sin
# coordenadas el **centroide de su manzana**: el promedio de los puntos
# distintos (edificios o lotes) ya ubicados en ella. Un edificio de 300
# apartamentos cuenta como un solo punto, para no arrastrar el centro hacia él.
#
# El código de manzana no siempre describe un lugar compacto. La manzana
# `0000` es un comodín (zona rural o predio sin manzana asignada): hay
# prefijos con ese código que reúnen 820 edificios repartidos en 7 km. Por eso
# se aplican dos reglas fijas (`src/config.py`): (i) no se usa la manzana
# `0000` y (ii) no se usa una manzana si alguno de sus puntos ubicados está a
# más de **250 m** del centroide (una manzana de Barranquilla mide entre 80 y
# 120 m de lado). Las reglas solo miran coordenadas, nunca el estrato.
#
# **¿Por qué no hay fuga?** El centroide usa la posición de otras viviendas, no
# su estrato. Es un dato cartográfico (dónde está la manzana) que también
# existiría para un predio nuevo, y se calcula antes de la partición, igual
# que la grilla de bloques del capítulo 2.

# %%
# Destino de las filas que llegan sin coordenadas y precisión de la aproximación.
# Se reconstruye la tabla de manzanas con las coordenadas ANTES del paso 6 (las
# filas recuperadas vuelven a NaN) para describir cada manzana con sus puntos reales.
base = df.copy()
base.loc[base["ubicado_por_manzana"], ["centroide_lat", "centroide_lon"]] = np.nan
_, manz = L.recuperar_por_manzana(base)
mz = base.loc[base["centroide_lat"].isna(), "npn_manzana"]
tiene = mz.isin(manz.index).to_numpy()   # la manzana tiene viviendas ubicadas
valida = mz.isin(manz.index[manz["valida"]]).to_numpy()
generica = mz.isin(manz.index[manz["generica"]]).to_numpy()
destino = np.select(
    [valida, generica, tiene],
    ["recuperada: centroide de su manzana", "sin ubicar: manzana genérica (0000)",
     f"sin ubicar: manzana dispersa (> {C.DISPERSION_MAX_MANZANA_M} m)"],
    default="sin ubicar: manzana sin viviendas ubicadas")
tabla_destino = pd.Series(destino).value_counts().rename("filas").to_frame()
tabla_destino["% de las filas sin coordenadas propias"] = 100 * tabla_destino["filas"] / len(mz)
tabla_destino["% del dataset"] = 100 * tabla_destino["filas"] / len(df)
print(f"Filas con coordenadas tras el paso 6: {int(df['tiene_coordenadas'].sum()):,} "
      f"({100 * df['tiene_coordenadas'].mean():.1f} %)")
tabla_destino

# %%
# ¿Qué error introduce poner una vivienda en el centroide de su manzana? Se prueba
# con los puntos que SÍ están ubicados: a cada edificio de una manzana válida (con
# al menos 2 puntos) se le quita su coordenada y se le asigna el centroide de los
# DEMÁS puntos de la manzana (dejar uno fuera). La distancia entre su posición real
# y ese centroide es el error que tendría si hubiera llegado sin coordenadas.
pts = (base.loc[base["centroide_lat"].notna(), ["npn_manzana", "x_km", "y_km"]]
       .drop_duplicates())
pts = pts[pts["npn_manzana"].isin(manz.index[manz["valida"]])]
n_pts = pts.groupby("npn_manzana")["x_km"].transform("size")
pts, n_pts = pts[n_pts >= 2], n_pts[n_pts >= 2]
# Centroide de los demás puntos: (suma de la manzana - el punto) / (n - 1)
cx = (pts.groupby("npn_manzana")["x_km"].transform("sum") - pts["x_km"]) / (n_pts - 1)
cy = (pts.groupby("npn_manzana")["y_km"].transform("sum") - pts["y_km"]) / (n_pts - 1)
error_m = 1000 * np.hypot(pts["x_km"] - cx, pts["y_km"] - cy)
print(f"Puntos evaluados: {len(error_m):,} en {pts['npn_manzana'].nunique():,} manzanas válidas")
print(f"Error de posición (m): mediana {error_m.median():.0f}, p90 {error_m.quantile(0.9):.0f}, "
      f"p99 {error_m.quantile(0.99):.0f}, máximo {error_m.max():.0f}")
print(f"Puntos a menos de 100 m de su centroide: {100 * (error_m < 100).mean():.1f} %")
print(f"Comparación: bloque de la partición = {1000 * C.TAM_BLOQUE_KM:.0f} m de lado")
# Cuántas viviendas recuperadas comparten un mismo punto (todas las de su manzana)
por_punto = df[df["ubicado_por_manzana"]].groupby("npn_manzana").size()
print(f"Viviendas recuperadas por manzana: mediana {por_punto.median():.0f}, "
      f"p90 {por_punto.quantile(0.9):.0f}, máximo {por_punto.max()} ({len(por_punto):,} manzanas)")

# %% [markdown]
# ```{admonition} Lectura
# :class: important
# - De las **53 894 filas** sin coordenadas propias, el **92.7 %** comparte
#   manzana con viviendas ubicadas, pero no todas esas manzanas son lugares
#   compactos. Con las dos reglas, el paso 6 ubica **43 594 (80.9 %)**. Con
#   ello el **96.9 %** de las viviendas tiene coordenadas, frente al 83.8 %
#   sin este paso. Quedan **10 300 filas (3.1 %)** sin ubicar: 3 913 en
#   manzanas sin ninguna vivienda ubicada, 3 502 en manzanas dispersas y
#   2 885 con la manzana genérica `0000`.
# - **La aproximación es buena.** Dejando fuera cada uno de los 128 314
#   edificios o lotes ubicados de las manzanas válidas y poniéndolo en el
#   centroide de los demás puntos de su manzana, el error mediano es de
#   **31 m**, el p90 de 68 m y el p99 de 122 m; el 97.5 % queda a menos de
#   100 m. Es muy poco frente a los bloques de 2 km de la partición
#   (capítulo 2) y al buffer de 1 km (capítulo 8): una vivienda recuperada cae
#   casi siempre en el mismo bloque y del mismo lado del buffer que si se
#   conociera su terreno.
# - **El costo: varias viviendas en un mismo punto.** Todas las viviendas
#   recuperadas de una manzana comparten coordenada (3 483 manzanas; mediana
#   de 7 viviendas por manzana, máximo 329). Es el mismo caso de un edificio de
#   apartamentos, y se trata igual: en los análisis espaciales (Moran, LISA,
#   correlograma) las viviendas recuperadas de una manzana cuentan como **un
#   solo punto** (`espacial.deduplicar_por_edificio`), para no fabricar
#   vecinos a distancia 0 con el mismo estrato.
# ```
#
# ### 1.7.3 ¿Quitar las que quedan cambia las estadísticas?
#
# Si las coordenadas faltaran **al azar** (por ejemplo, por un error técnico
# que afecta a cualquier vivienda por igual), quitar esas filas no cambiaría la
# distribución del estrato: sería como sacar una muestra aleatoria más
# pequeña. Si no faltan al azar, las estadísticas cambian. La tabla compara
# la distribución del estrato en las dos poblaciones (todas las filas limpias y
# las que tienen coordenadas tras el paso 6) y añade la **cobertura**: qué
# porcentaje de las viviendas de cada estrato tiene coordenadas.

# %%
# Distribución del estrato en los dos universos y cobertura de coordenadas por
# estrato. Si la cobertura fuera parecida en todos los estratos, faltarían al azar.
casos_todas = df[C.OBJETIVO].value_counts().sort_index()
casos_coord = con_coord[C.OBJETIVO].value_counts().sort_index()
clases = pd.DataFrame({
    "casos (todas)": casos_todas,
    "% (todas)": 100 * casos_todas / casos_todas.sum(),
    "casos (con coordenadas)": casos_coord,
    "% (con coordenadas)": 100 * casos_coord / casos_coord.sum(),
    "cobertura de coordenadas (%)": 100 * casos_coord / casos_todas,
})
clases["diferencia (p.p.)"] = clases["% (con coordenadas)"] - clases["% (todas)"]
clases.index = [C.NOMBRES_CLASES[i] for i in clases.index]   # "1 Bajo-bajo", ...
clases

# %% tags=["hide-input"]
# Figura: % de cada estrato en los dos universos (barras pareadas) y cobertura de
# coordenadas por estrato. Muestra de un vistazo que las faltantes son de estratos bajos.
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [3, 2]})
clases[["% (todas)", "% (con coordenadas)"]].plot.bar(ax=axes[0], color=["#9bb7d4", "#1f3b57"], rot=0, width=0.8)
for cont in axes[0].containers:
    axes[0].bar_label(cont, fmt="%.1f", fontsize=9, padding=1)
axes[0].set_ylabel("% de las filas del universo")
axes[0].set_title("Distribución del estrato: todas las filas vs. con coordenadas")
axes[0].legend(["todas las filas limpias", "con coordenadas (modelado)"])
axes[1].bar(clases.index, clases["cobertura de coordenadas (%)"],
            color=[G.PALETA_ESTRATO[c] for c in C.CLASES])
for i, v in enumerate(clases["cobertura de coordenadas (%)"]):
    axes[1].text(i, v + 1, f"{v:.1f}", ha="center", fontsize=10)
axes[1].set_ylim(0, 110)
axes[1].set_ylabel("% de viviendas con coordenadas")
axes[1].set_title("Cobertura de coordenadas por estrato")
axes[1].tick_params(axis="x", rotation=30)
fig.tight_layout()
G.guardar(fig, "01_universos_estrato")
plt.show()

# %% [markdown]
# ```{admonition} Lectura
# :class: important
# **Cambia poco, pero no faltan al azar.**
#
# - El **estrato 1** pasa de **32.8 %** de todas las filas a **31.1 %** de las
#   que tienen coordenadas (−1.7 puntos porcentuales); los estratos 2 a 6 ganan
#   menos de un punto cada uno. Sin la ubicación por manzana, la caída del
#   estrato 1 habría sido de 6.5 puntos (a 26.3 %).
# - La cobertura de coordenadas es de **92.1 %** en el estrato 1, frente a
#   **98.7 % o más** en los estratos 2 a 6. Dicho de forma sencilla: a **1 de
#   cada 13** viviendas de estrato 1 le falta la ubicación; sin la ubicación
#   por manzana habría sido 1 de cada 3.
# - Consecuencia: las estadísticas calculadas solo con las filas con
#   coordenadas describen casi toda la ciudad, pero con algo menos de estrato 1
#   informal. Por eso se sigue diciendo con qué universo se calculó cada cifra.
# ```
#
# ### 1.7.4 ¿Hace falta quitar las que quedan?
#
# **Para el modelo, sí.** Una vivienda sin ubicación no puede usarse en este
# estudio por tres razones, cada una suficiente por sí sola:
#
# 1. **La partición es espacial.** El conjunto de prueba se separa por bloques
#    de 2 km × 2 km (capítulo 2). Una vivienda sin coordenadas no se puede
#    asignar a ningún bloque, así que no se sabría si va a entrenamiento o a
#    prueba.
# 2. **La ubicación es una predictora.** El modelo principal usa `x_km`,
#    `y_km` y la distancia al centro. Rellenar coordenadas sin una regla con
#    fundamento sería darle al modelo un dato falso justo en la variable más
#    informativa. (La ubicación por manzana sí tiene ese fundamento: es la
#    posición real de la manzana, con un error medido de unos 30 m.)
# 3. **El análisis espacial las necesita.** Los mapas, el índice de Moran y la
#    elección del buffer (capítulo 8) se calculan con distancias entre
#    viviendas.
#
# **Para describir la base, no.** Las 332 718 filas limpias son la población
# real de viviendas con estrato del catastro, y se siguen usando para
# describirla.

# %% [markdown]
# ```{admonition} Decisión: se ubican por manzana y se reportan los dos universos
# :class: important
# 1. **Ubicación por manzana.** Las viviendas sin coordenadas propias reciben
#    el centroide de su manzana cuando la manzana es compacta (paso 6 del
#    embudo): 43 594 viviendas, con un error mediano estimado de 31 m.
# 2. **Para describir la base** (este capítulo) se usan las **332 718 filas
#    limpias**, y las cifras clave se muestran al lado de las del universo con
#    coordenadas (tablas de 1.6, 1.7.3, 1.8 y 1.9). Así el lector ve la
#    población real y, a la vez, cuánto cambia al quitar las filas sin
#    ubicación.
# 3. **Para todo lo que viene después** (partición, análisis exploratorio y
#    modelo) se usan las **322 418 filas con coordenadas** (96.9 %), porque sin
#    ubicación no se puede partir, ni calcular variables espaciales, ni hacer
#    el análisis espacial. Cada capítulo indica al inicio con qué universo
#    trabaja.
# 4. **El sesgo que queda se declara y se cuantifica.** Las 10 300 filas que
#    siguen sin coordenadas (3.1 %) son casi todas vivienda informal de
#    estrato 1. El capítulo 3.3 lo mide, y las conclusiones (11.2) lo declaran
#    como limitación.
# ```
#
# | Universo | Filas | Dónde se usa |
# |---|---|---|
# | **Todas las filas limpias** | 332 718 | Capítulo 1: descripción de la base, calidad y tamaño; capítulo 3.3: sesgo de cobertura |
# | **Con coordenadas (universo de modelado)** | 322 418 | Capítulo 2: partición en entrenamiento (257 273) y prueba (65 145) |
# | **Entrenamiento** | 257 273 | Capítulos 3 a 9 (todo el análisis exploratorio) |
# | **Entrenamiento tras el buffer** | 144 586 | Capítulo 10: modelo final |
# | **Prueba** | 65 145 | Capítulo 10: evaluación final, una sola vez |

# %% [markdown]
# ## 1.8 Tamaño de la muestra y relación n/p
#
# **¿Qué es la relación n/p?** Es el número de filas (**n**) dividido entre el
# número de columnas que entran al modelo (**p**): cuántos datos hay por cada
# cosa que el modelo debe aprender. Si hay pocas filas por columna, el modelo
# memoriza los datos en lugar de aprender patrones generales
# (**sobreajuste**). *Ejemplo:* con 50 filas y 40 columnas, un modelo puede
# "aprenderse de memoria" casi cada fila; con 300 000 filas y 35 columnas,
# no. Una referencia usual para regresión logística es tener al menos 10
# casos de la clase menos frecuente por columna (Peduzzi et al., 1996).
#
# **¿Qué es one-hot?** Los modelos solo trabajan con números, así que cada
# categoría se convierte en una columna de 0 y 1. Por ejemplo,
# `condicion_predio` se vuelve una columna "¿es NPH?", otra "¿es
# PH_Unidad_Predial?", otra "¿es Informal?", etc. Por eso p no son las 16
# variables candidatas, sino las columnas que quedan tras one-hot.
#
# La tabla cuenta, en los dos universos, las filas, los predios, los predios
# matriz y la relación n/p.

# %%
# Tamaño de muestra y relación n/p en los dos universos. Verifica el mínimo de
# 20 000 observaciones de las instrucciones y que haya suficientes filas por
# parámetro. También cuenta entidades (predios, predios matriz), porque las filas
# de un mismo predio matriz no son independientes.
p_num = len(C.NUMERICAS) + len(C.ESPACIALES)   # numéricas + coordenadas/distancia
p_cat = len(C.CATEGORICAS)


def grupo_icc(npn):
    # Grupo = predio matriz (22 primeros dígitos del NPN), salvo las mejoras en
    # terreno ajeno (dígito 22 = "5"), que son viviendas independientes: cada una
    # es su propio grupo (sección 1.3).
    return npn.str[:22].where(npn.str[21] != "5", npn)


def tamano(d):
    # Tras one-hot cada nivel de una categórica se vuelve una columna: se suman niveles.
    p_efectivo = p_num + sum(d[c].nunique() for c in C.CATEGORICAS if c in d)
    minoritaria = d[C.OBJETIVO].value_counts().min()
    return {
        "filas (unidades de vivienda)": len(d),
        "predios (NPN de 30 dígitos)": d["numero_predial_nacional"].nunique(),
        "predios matriz (NPN de 22 dígitos)": d["npn_edificio"].nunique(),
        "grupos para el ICC (predio matriz; mejoras por separado)": grupo_icc(d["numero_predial_nacional"]).nunique(),
        "variables predictoras candidatas": p_num + p_cat,
        "columnas tras one-hot (p)": p_efectivo,
        "n / p": len(d) / p_efectivo,
        "casos de la clase menos frecuente": minoritaria,
        "casos de la clase menos frecuente / p": minoritaria / p_efectivo,
    }


tam = pd.DataFrame({"todas las filas": tamano(df), "con coordenadas": tamano(con_coord)})
tam

# %% [markdown]
# ```{admonition} Interpretación
# :class: note
# - **Tamaño bruto.** Tras el embudo quedan **332 718 viviendas** en **320 792
#   predios** y **168 044 predios matriz** (edificios, conjuntos o lotes). En
#   el universo con coordenadas son **322 418 viviendas**, 310 831 predios y
#   165 954 predios matriz. Los dos universos superan con holgura el mínimo de
#   20 000 observaciones de las instrucciones del proyecto.
# - **Relación n/p.** Tras one-hot hay **35 columnas**: n/p ≈ **9 500** filas
#   por columna con todas las filas y ≈ **9 200** con coordenadas. La clase
#   menos frecuente (estrato 6) tiene 13 207 casos en los dos universos (todas
#   sus viviendas tienen coordenadas), unos **377 por columna**, muy por
#   encima de la referencia de 10. No faltan datos para el número de
#   variables.
# - **Grupos para el ICC.** Al separar las mejoras en terreno ajeno (sección
#   1.3), los 168 044 predios matriz pasan a **182 524 grupos**. En el
#   universo con coordenadas pasan de 165 954 a 172 757: la mayoría de las
#   mejoras también tienen ahora coordenadas, por su manzana (sección
#   1.7.2).
# ```

# %% [markdown]
# ## 1.9 Dependencia dentro del predio matriz: el tamaño efectivo
#
# ### 1.9.1 La idea, con un ejemplo
#
# Las viviendas de un mismo edificio o conjunto casi siempre tienen el mismo
# estrato, porque el estrato se asigna por manzana o lado de manzana. Eso
# significa que **no son observaciones independientes**: si ya sabemos que el
# apartamento 101 de una torre es estrato 4, el 201 no nos dice casi nada
# nuevo. Es como fotocopiar una hoja 1 000 veces: hay 1 000 hojas, pero una
# sola página de información.
#
# El **tamaño efectivo** ($n_\text{ef}$) responde: *¿a cuántas observaciones
# independientes equivale nuestra muestra?* Se calcula en cinco pasos:
#
# 1. **Definir el grupo.** Aquí, el predio matriz (con la excepción de las
#    mejoras, sección 1.3).
# 2. **Medir el parecido dentro del grupo** con la **correlación intraclase
#    (ICC)**: qué fracción de toda la variación del estrato ocurre *entre*
#    grupos. ICC = 0 significa que las viviendas de un mismo grupo no se
#    parecen más que dos viviendas al azar; ICC = 1 significa que dentro de un
#    grupo todas son idénticas.
# 3. **Medir el tamaño típico del grupo** ($\tilde m$), visto desde una
#    vivienda (se explica abajo por qué).
# 4. **Calcular el efecto de diseño de Kish** (deff): cuántas veces es menos
#    precisa la muestra que una de observaciones independientes:
#
#    $$
#    \text{deff} = 1 + (\tilde m - 1)\cdot \text{ICC}
#    $$
#
# 5. **Dividir:** $n_\text{ef} = N / \text{deff}$, con $N$ el número de filas.
#
# *Ejemplo de juguete.* Un barrio con 10 edificios de 10 apartamentos cada uno
# ($N = 100$, $\tilde m = 10$). La tabla de la celda siguiente muestra qué
# pasa con distintos valores de ICC.

# %%
# Ejemplo de juguete del efecto de diseño de Kish: 10 edificios de 10 apartamentos.
# Con ICC = 0 cada apartamento cuenta como una observación; con ICC = 1 cada
# edificio cuenta como una sola, así que n_ef = número de edificios.
N_juguete, m_juguete = 100, 10
juguete = pd.DataFrame({"ICC": [0.0, 0.25, 0.5, 0.9, 1.0]})
juguete["deff = 1 + (m − 1)·ICC"] = 1 + (m_juguete - 1) * juguete["ICC"]
juguete["n_ef = N / deff"] = N_juguete / juguete["deff = 1 + (m − 1)·ICC"]
juguete

# %% [markdown]
# Con ICC = 0 (los apartamentos de un edificio no se parecen entre sí), los
# 100 apartamentos valen como 100 observaciones. Con ICC = 1 (todos los
# apartamentos de un edificio son iguales), valen como 10: uno por edificio.
# Los valores intermedios quedan entre esos dos extremos.
#
# **¿Por qué $\tilde m$ no es el promedio por edificio?** En la vida real los
# grupos tienen tamaños muy distintos, y hay dos formas de calcular el
# "tamaño promedio":
#
# - Preguntarle a cada **edificio** cuántas viviendas tiene y promediar.
# - Preguntarle a cada **vivienda** cuántas viviendas hay en su edificio y
#   promediar las respuestas: $\tilde m = \sum_j m_j^2 / N$, donde $m_j$ es el
#   número de viviendas del grupo $j$.
#
# *Ejemplo.* En un barrio con 99 casas y una torre de 901 apartamentos hay 100
# "edificios" y 1 000 viviendas. Por edificio, el promedio es 1 000 ÷ 100 =
# 10. Pero 901 de las 1 000 viviendas están en la torre, así que, preguntando
# a las viviendas, el promedio es (99 × 1² + 901²) ÷ 1 000 ≈ 812. La segunda
# forma es la que importa para medir la repetición, porque los edificios
# grandes contienen más filas y las filas son lo que usamos para modelar.
#
# ### 1.9.2 El cálculo con los datos reales
#
# La celda calcula el ICC con un ANOVA de un factor (el grupo), tratando el
# estrato como numérico. Se presenta en los dos universos y, para mostrar el
# efecto de la excepción de las mejoras, también sin ella.

# %%
# ICC(1) y efecto de diseño de Kish: cuánto se parecen en estrato las unidades de
# un mismo grupo y a cuántas observaciones independientes equivalen las filas.
# Es solo descriptivo; no alimenta ninguna decisión aprendida.
def icc_kish(d, grupo):
    g = d.groupby(grupo.to_numpy())[C.OBJETIVO]   # el grupo es el factor del ANOVA
    m = g.size()                                   # m_j: unidades por grupo
    k, N = len(m), m.sum()                         # k grupos, N observaciones
    media_total = d[C.OBJETIVO].mean()
    # Sumas de cuadrados entre grupos (SSB) y dentro de grupos (SSW) del ANOVA de una vía.
    ssb = (m * (g.mean() - media_total) ** 2).sum()
    ssw = ((d[C.OBJETIVO].to_numpy() - g.transform("mean").to_numpy()) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k)        # cuadrados medios con sus grados de libertad
    # m0: tamaño de grupo "ajustado" para grupos desiguales (fórmula clásica del ICC(1)).
    m0 = (N - (m ** 2).sum() / N) / (k - 1)
    icc = (msb - msw) / (msb + (m0 - 1) * msw)
    # Tamaño medio ponderado Σm²/N (tamaño del grupo visto desde una vivienda).
    m_pond = (m ** 2).sum() / N
    deff = 1 + (m_pond - 1) * icc                  # efecto de diseño de Kish
    return {"filas N": N, "grupos k": k, "unidades por grupo: media": m.mean(),
            "mediana": m.median(), "máximo": m.max(), "% grupos de 1 unidad": 100 * (m == 1).mean(),
            "m̃ = Σm²/N": m_pond, "ICC(1)": icc, "deff": deff, "n_ef = N/deff": N / deff}


filas_icc = {}
for nombre_u, d in [("todas las filas", df), ("con coordenadas", con_coord)]:
    npn = d["numero_predial_nacional"]
    filas_icc[(nombre_u, "con excepción de mejoras")] = icc_kish(d, grupo_icc(npn))
    filas_icc[(nombre_u, "sin excepción")] = icc_kish(d, npn.str[:22])
tabla_icc = pd.DataFrame(filas_icc).T
tabla_icc

# %% [markdown]
# ```{admonition} Interpretación
# :class: note
# **Los cinco pasos con los datos reales (todas las filas, con la excepción de
# las mejoras):**
#
# 1. **Grupo:** 182 524 grupos. La mediana es **1 vivienda por grupo** y el
#    **89 %** de los grupos tiene una sola vivienda (casas), pero hay conjuntos
#    de hasta **1 860 unidades** (93 torres en un mismo predio matriz).
# 2. **ICC = 0.992:** dentro de un edificio o conjunto el estrato
#    prácticamente no varía.
# 3. **Tamaño visto desde una vivienda:** $\tilde m = \sum m^2 / N = 145.4$.
#    Aunque casi todos los grupos son casas sueltas, una vivienda típica vive
#    en un grupo grande, porque los conjuntos grandes concentran muchas filas.
# 4. **Efecto de diseño:** deff = 1 + (145.4 − 1) × 0.992 ≈ **144.3**. Los
#    errores estándar son unas $\sqrt{144.3} \approx 12$ veces más grandes
#    que si las filas fueran independientes.
# 5. **Tamaño efectivo:** $n_\text{ef}$ = 332 718 ÷ 144.3 ≈ **2 300**.
#
# **Las otras filas de la tabla.**
#
# - *Sin la excepción de las mejoras*, el "edificio" más grande sería un lote
#   del suroccidente con 3 045 viviendas informales independientes; el
#   tamaño efectivo quedaría subestimado (≈ 1 870 en vez de ≈ 2 300).
# - *En el universo con coordenadas*, el ICC es el mismo (0.992), pero
#   $\tilde m$ sube a 149.9 y el tamaño efectivo baja a **≈ 2 170**: al quitar
#   las 10 300 casas informales sueltas que siguen sin ubicación, una parte
#   algo mayor de las filas vive en conjuntos grandes. Aquí la excepción
#   importa poco (≈ 2 160 sin ella), porque el lote con miles de mejoras
#   independientes no tiene coordenadas.
#
# **Cómo leer ese 2 300 (o 2 170).** No es un conteo de viviendas, de predios
# ni de edificios, ni quiere decir que haya 2 300 filas "reales" y el resto
# sobre. Es una medida de **precisión**: las estimaciones sobre el estrato
# tienen la precisión de unas 2 300 observaciones independientes, no de
# 332 718. Hay 168 044 predios matriz distintos, muy por encima del mínimo de
# 20 000; lo que el tamaño efectivo indica es que la precisión está gobernada
# por los edificios, los conjuntos y las zonas, no por las filas.
#
# **Dos consecuencias prácticas:** (1) la partición debe mantener juntas las
# unidades de un mismo predio matriz y de una misma zona (capítulo 2), para que
# una torre nunca quede repartida entre entrenamiento y prueba; y (2) los
# intervalos de confianza deben remuestrear bloques, no filas (capítulo 10). El
# cálculo es solo descriptivo: no alimenta ninguna decisión aprendida.
# ```

# %% [markdown]
# ## 1.10 Sesgos de muestreo y representatividad
#
# El catastro no es una muestra sino (casi) un censo de los predios formales,
# pero tiene sesgos conocidos:
#
# 1. **Viviendas sin coordenadas** (16.2 % de las filas limpias, sección 1.7):
#    casi todas son mejoras en terreno ajeno de estratos 1 y 2. La mayoría se
#    ubica con el centroide de su manzana; el 3.1 % que no se puede ubicar
#    queda fuera del modelo y es casi todo informal de estrato 1. El capítulo
#    3.3 cuantifica este sesgo con la V de Cramér.
# 2. **Predios sin estrato residencial** (`No_Aplica`/`Otro`/vacío: 12 093
#    filas en el paso 3 del embudo, 3.2 % del crudo): quedan fuera por ser la
#    variable objetivo.
# 3. **Desactualización:** la información catastral puede no reflejar
#    reformas no declaradas, lo que añade ruido a las predictoras.

# %% [markdown]
# ## 1.11 Consideraciones éticas
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
# - **Sesgo de cobertura:** la ubicación por manzana incorpora al modelo la
#   mayoría de los predios informales, pero el 3.1 % que no se pudo ubicar es
#   casi todo informal y de estrato 1 (sección 1.7); las conclusiones sobre
#   esos asentamientos deben leerse con cautela.

# %% [markdown]
# ## 1.12 Resumen del capítulo
#
# ```{admonition} Resumen
# :class: note
# - **Base.** De 382 597 filas descargadas quedan **332 718 viviendas** con
#   estrato residencial (−36 497 garajes/depósitos, −11 sin área, −1 044
#   unidades de menos de 10 m², −234 registros agregados, −12 093 sin estrato
#   residencial), en 320 792 predios y 168 044 predios matriz.
# - **Ubicación por manzana y dos universos.** El 16.2 % de las viviendas
#   (53 894, casi todas mejoras en terreno ajeno de estratos 1 y 2) llega sin
#   coordenadas. El paso 6 ubica a 43 594 de ellas en el centroide de su
#   manzana (error mediano de 31 m, medido con los edificios ubicados), y
#   quedan sin ubicar 10 300 (3.1 %). Se describe la base con las 332 718
#   filas y se modela con las **322 418 que tienen coordenadas**; las cifras
#   clave se presentan en los dos universos. Quitar las que siguen sin
#   coordenadas baja el estrato 1 de 32.8 % a 31.1 % (sin la ubicación por
#   manzana habría bajado a 26.3 %).
# - **Tamaño.** Muy por encima del mínimo de 20 000 y con ~9 200–9 500 filas
#   por columna. La precisión, sin embargo, está gobernada por los edificios y
#   conjuntos: ICC = 0.992 y tamaño efectivo ≈ 2 300 (≈ 2 170 con
#   coordenadas).
# - **Calidad.** No hay duplicados exactos; las unidades repetidas (6.8 % por
#   NPN, 38.4 % casi-duplicados) son viviendas reales y se manejan agrupando
#   por predio matriz y bloque al partir. Los valores imposibles que quedan son
#   muy pocos (≤ 0.04 % por variable, salvo el piso, 0.11 %, por códigos como
#   el 99), incluyen códigos centinela (año 2500/1512) y, junto con las
#   incoherencias (396 filas, 0.12 %), se tratan como faltantes a imputar
#   dentro del Pipeline.
# - **Ética.** Riesgos de reidentificación por NPN + coordenadas y de uso
#   discriminatorio del estrato; se mitigan con agregación y declarando el
#   alcance.
# ```
