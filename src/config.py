# -*- coding: utf-8 -*-
"""
Configuración central del Entregable 1: rutas, semillas y constantes.

TODAS las semillas aleatorias del libro salen de aquí (requisito de
reproducibilidad del rubric: "semillas aleatorias fijadas").

Dónde busca el CSV (en este orden):
  1. La variable de entorno CATASTRO_CSV, si existe.
  2. <libro>/datos/predios_residenciales_barranquilla.csv
  3. <carpeta que contiene el libro>/predios_residenciales_barranquilla.csv
     (así funciona si pones la carpeta del libro DENTRO de tu carpeta
     catastro_arquetipos, junto al CSV que ya generaste con
     01_descarga_predios.py).

Por qué un único módulo de configuración: si un umbral, una semilla o una
lista de variables se definiera en cada notebook, sería fácil que dos
capítulos usaran valores distintos sin notarlo. Centralizarlos garantiza que
el EDA, la partición y el modelo base trabajen exactamente con las mismas
reglas, y que cambiar una decisión (p. ej. el tamaño de bloque) se propague
a todo el libro con una sola edición.
"""

import json
import os
from pathlib import Path

# Raíz del libro: este archivo vive en <libro>/src/config.py, así que se sube
# un nivel. Usar __file__ (y no el directorio de trabajo) hace que las rutas
# funcionen igual sin importar desde qué carpeta se ejecute cada notebook.
RAIZ = Path(__file__).resolve().parents[1]
NOMBRE_CSV = "predios_residenciales_barranquilla.csv"

CARPETA_DATOS = RAIZ / "datos"
CARPETA_PROCESADOS = CARPETA_DATOS / "procesados"  # salidas intermedias (partición, decisiones)
CARPETA_FIGURAS = RAIZ / "figuras"
# Registro JSON de decisiones del EDA que luego lee el notebook del modelo base
ARCHIVO_DECISIONES = CARPETA_PROCESADOS / "decisiones_eda.json"
# Caché local de los dominios de códigos del servicio ArcGIS (ver dominios.py)
ARCHIVO_DOMINIOS = CARPETA_DATOS / "dominios_arcgis.json"

# Se crean las carpetas al importar el módulo para que ningún notebook falle
# al guardar una figura o un archivo procesado en una instalación limpia.
for _c in (CARPETA_DATOS, CARPETA_PROCESADOS, CARPETA_FIGURAS):
    _c.mkdir(parents=True, exist_ok=True)

# --- Reproducibilidad -------------------------------------------------------
# Semilla única para particiones, muestreos y modelos: con ella cualquier
# lector obtiene exactamente las mismas cifras y figuras del informe.
SEED = 42

# --- Fuente -----------------------------------------------------------------
# Servicio REST de ArcGIS de la Alcaldía de Barranquilla (datos abiertos de
# catastro). Se usa para citar la fuente y para descargar los dominios.
URL_SERVICIO = ("https://miciudad.barranquilla.gov.co/gis/rest/services/"
                "catastro/datosabiertos/MapServer")

# --- Reglas de dominio (valores físicamente imposibles) --------------------
# Estos límites son reglas FIJAS de conocimiento del dominio (no se estiman
# con los datos). Por eso pueden aplicarse fila a fila antes de la partición
# sin generar fuga de información: valdrían igual para un predio nuevo.
ANIO_ACTUAL = 2026
ANIO_MIN_PLAUSIBLE = 1900      # ver bitácora 3.2: primer dato real = 1900; 1512 es centinela
ANIO_MAX_PLAUSIBLE = ANIO_ACTUAL  # una construcción no puede ser posterior al año en curso
# Rango plausible (mínimo, máximo) por variable; fuera de él el valor se
# considera un error de captura y se marca como NaN en limpieza.py.
LIMITES_PLAUSIBLES = {
    "total_habitaciones": (0, 30),
    "total_banios": (0, 20),
    "total_plantas": (0, 40),
    # 97, 98 y 99 son códigos (99 en 296 filas de train, con área mediana de
    # 10 m2), no pisos. El diagnóstico de rangos mostró pisos REALES hasta 41
    # (41 filas entre 30 y 41: apartamentos en PH de estrato 5-6 con área
    # mediana de 362 m2), así que el techo se sube de 40 a 60: deja pasar las
    # torres reales y sigue atrapando los códigos.
    "planta_ubicacion": (1, 60),
    "area_construida": (1, 20_000),          # m2
    "area_catastral_terreno": (0, 50_000),   # m2 (en los datos reales no hay ceros; PH = cuota del lote)
}

# --- Alcance: qué fila cuenta como UNA unidad de vivienda ----------------------
# Diagnóstico de rangos (diagnostico_rangos.py, solo train; bitácora A): los
# extremos de habitaciones, baños y área no son errores de captura sino filas que
# NO describen una vivienda. Se excluyen del alcance, igual que garajes y depósitos.
# (1) Área mínima: unidades en PH de 2-10 m2 (muchas con piso "99") son
#     parqueaderos o depósitos registrados con uso residencial.
# Umbral conservador: en train hay 215 filas < 5 m2 y 656 < 10 m2, repartidas en
# todos los estratos (no solo informales) y con frecuencia con 0 baños o el piso
# "99". No se usan 15 o 20 m2 para no excluir viviendas mínimas reales de uno
# o dos cuartos (1 411 y 2 203 filas, respectivamente).
AREA_MIN_VIVIENDA = 10  # m2
# (2) Registros agregados: filas NPH que describen un edificio completo
#     (p. ej. 684 habitaciones, 456 baños y 52 168 m2 en una sola fila). Una
#     vivienda individual no supera estos valores en la ciudad.
# En train: 87 filas con más de 20 habitaciones o más de 2 000 m2 (63 NPH).
UMBRALES_REGISTRO_AGREGADO = {
    "total_habitaciones": 20,
    "total_banios": 15,
    "area_construida": 2_000,  # m2
}
# (3) Reglas de COHERENCIA dentro de la fila (valor imposible dado el resto de
#     la fila, aunque cada valor por separado esté en rango). El valor sospechoso
#     pasa a NaN + bandera 'flag_incoherente_<var>'; la fila se conserva.
#     - baños > habitaciones + MAX_BANIOS_SOBRE_HAB  -> total_banios = NaN
#       (p. ej. un apartamento de 2 habitaciones con 82 baños en 82 m2)
#     - área por habitación < MIN_AREA_POR_HABITACION -> total_habitaciones = NaN
MAX_BANIOS_SOBRE_HAB = 3
MIN_AREA_POR_HABITACION = 6  # m2

# Usos catastrales que son "residenciales" en la base pero no son vivienda:
# un garaje o un depósito no tiene estrato de hogar que predecir, así que se
# excluyen del alcance del problema (paso 1 del embudo de limpieza).
USOS_NO_HABITABLES = [
    "Residencial_Garajes_En_PH",
    "Residencial_Garajes_Cubiertos",
    "Residencial_Depositos_Lockers",
]

# --- Variables ----------------------------------------------------------------
# Variable objetivo: estrato socioeconómico como entero 1-6 (se trata como
# clasificación multiclase; el orden de las clases se aprovecha en los
# gráficos y en el análisis de errores entre estratos vecinos).
OBJETIVO = "estrato_num"
CLASES = [1, 2, 3, 4, 5, 6]
# Etiquetas legibles para tablas y figuras (nomenclatura oficial de estratos)
NOMBRES_CLASES = {1: "1 Bajo-bajo", 2: "2 Bajo", 3: "3 Medio-bajo",
                  4: "4 Medio", 5: "5 Medio-alto", 6: "6 Alto"}

# Predictores numéricos (físicos de la unidad y del lote). 'antiguedad' se
# deriva de anio_construccion en limpieza.py.
NUMERICAS = [
    "area_construida",
    "area_catastral_terreno",
    "total_habitaciones",
    "total_banios",
    "total_plantas",
    "antiguedad",
    "planta_ubicacion",
    "altura",
]
# Predictores categóricos nominales (sin orden): se codifican con one-hot
# dentro del Pipeline. tipo_vivienda llega como código numérico, pero se
# convierte a texto en limpieza.py precisamente para que no se trate como número.
CATEGORICAS = [
    "tipo_vivienda",
    "uso",
    "condicion_predio",
    "destinacion_economica",
    "tipo_planta",
]
# Predictores espaciales en km (proyección local, ver limpieza.proyectar_km)
ESPACIALES = ["x_km", "y_km", "dist_centro_km"]

# Columnas que NO pueden entrar al modelo (auditoría de fuga, notebook 09)
# - IDENTIFICADORES: códigos únicos; el modelo podría memorizarlos en lugar
#   de aprender relaciones generalizables (npn_edificio identifica el edificio).
# - METADATOS_PIPELINE: 'estrato' es el objetivo en texto (fuga directa) y
#   'bloque'/'particion' son artefactos de la validación, no atributos del predio.
IDENTIFICADORES = ["numero_predial_nacional", "npn_edificio", "id_fila"]
METADATOS_PIPELINE = ["centroide_fuente", "estrato", "bloque", "particion"]

# --- Espacial -----------------------------------------------------------------
# Barranquilla: caja aproximada para validar coordenadas (WGS84, EPSG:4326)
# Un centroide fuera de esta caja es un error de georreferenciación y se anula.
LAT_MIN, LAT_MAX = 10.85, 11.15
LON_MIN, LON_MAX = -74.90, -74.65
# Punto de referencia para 'dist_centro_km': Paseo Bolívar / centro histórico
# (aproximado; verificar en un mapa si se quiere más precisión).
CENTRO_LAT, CENTRO_LON = 10.9837, -74.7766
RADIO_TIERRA_KM = 6371.0088  # radio medio de la Tierra (IUGG), usado en la proyección
# Tamaño de los bloques espaciales para la partición. Debe ser mayor que el
# alcance típico de la autocorrelación del estrato (un barrio), para que train
# y test no compartan vecindarios casi idénticos.
TAM_BLOQUE_KM = 2.0     # tamaño de los bloques espaciales para la partición
N_FOLDS = 5
FRACCION_TEST = 1 / N_FOLDS  # 1 de 5 grupos estratificados -> ~20 %
K_VECINOS = 8  # vecinos para la matriz de pesos espaciales (p. ej. índice de Moran)

# --- Muestras (solo para cálculos costosos; se documenta en cada notebook) --
# Submuestras aleatorias (con SEED) para gráficos densos, estadísticos
# espaciales O(n^2) e información mutua; no afectan al modelo, que usa todo train.
N_MUESTRA_GRAFICOS = 5_000
N_MUESTRA_ESPACIAL = 25_000
N_MUESTRA_MI = 60_000


def ruta_csv():
    """Devuelve la ruta del CSV de catastro según el orden de búsqueda.

    Recorre los candidatos descritos en el docstring del módulo (variable de
    entorno, carpeta 'datos' del libro, carpeta padre) y devuelve el primero
    que exista.

    Parámetros
    ----------
    (ninguno)

    Devuelve
    --------
    pathlib.Path
        Ruta al CSV encontrado.

    Lanza
    -----
    FileNotFoundError
        Si ningún candidato existe; el mensaje lista todas las rutas probadas
        para que el usuario sepa dónde colocar el archivo.

    Por qué
    -------
    El CSV pesa demasiado para versionarlo con el libro; permitir varias
    ubicaciones evita editar rutas a mano en cada máquina.
    """
    candidatos = []
    # La variable de entorno tiene prioridad: permite apuntar a otro CSV
    # (p. ej. el sintético de los tests) sin tocar el código.
    if os.environ.get("CATASTRO_CSV"):
        candidatos.append(Path(os.environ["CATASTRO_CSV"]))
    candidatos += [CARPETA_DATOS / NOMBRE_CSV, RAIZ.parent / NOMBRE_CSV]
    for c in candidatos:
        if c.exists():
            return c
    raise FileNotFoundError(
        "No encontré el CSV. Busqué en:\n  " + "\n  ".join(str(c) for c in candidatos)
        + "\nCopia predios_residenciales_barranquilla.csv a la carpeta 'datos' del libro "
          "o define la variable de entorno CATASTRO_CSV con su ruta.")


def usando_datos_sinteticos():
    """True si el CSV cargado es el sintético de prueba (tests/).

    Parámetros
    ----------
    (ninguno)

    Devuelve
    --------
    bool
        True cuando la variable de entorno CATASTRO_SINTETICO vale "1".

    Por qué
    -------
    El libro se prueba de punta a punta con datos sintéticos; esta bandera
    permite distinguir esas corridas de las reales y evitar que cifras
    inventadas terminen en el informe.
    """
    return os.environ.get("CATASTRO_SINTETICO", "0") == "1"


def aviso_sintetico():
    """Imprime un aviso visible si se está ejecutando con datos sintéticos.

    Se llama al inicio de cada notebook para que cualquier salida generada con
    el dataset de prueba quede claramente marcada como no válida.

    Parámetros
    ----------
    (ninguno)

    Devuelve
    --------
    None
    """
    if usando_datos_sinteticos():
        print("=" * 78)
        print("ATENCIÓN: se está usando el DATASET SINTÉTICO de prueba.")
        print("Estos números NO son resultados reales y NO deben ir al informe.")
        print("=" * 78)


# --- Registro de decisiones del EDA (trazabilidad EDA -> modelado) ----------
def leer_decisiones():
    """Lee el registro de decisiones del EDA (decisiones_eda.json).

    Parámetros
    ----------
    (ninguno)

    Devuelve
    --------
    dict
        Diccionario {clave: {"valor": ..., "motivo": ...}}; vacío si todavía
        no se ha registrado ninguna decisión.
    """
    if ARCHIVO_DECISIONES.exists():
        return json.loads(ARCHIVO_DECISIONES.read_text(encoding="utf-8"))
    return {}


def guardar_decision(clave, valor, motivo):
    """Guarda una decisión del EDA con su justificación. El notebook del
    modelo base lee este archivo: así cada decisión de preprocesamiento queda
    rastreada al hallazgo que la motivó (requisito del rubric).

    Parámetros
    ----------
    clave : str
        Nombre de la decisión (p. ej. "buffer_km"). Si ya existe, se sobrescribe,
        de modo que re-ejecutar un notebook actualiza la decisión en lugar de
        duplicarla.
    valor : objeto serializable en JSON
        Valor adoptado (número, lista de columnas, texto...).
    motivo : str
        Justificación en lenguaje natural, ligada al hallazgo del EDA.

    Devuelve
    --------
    None
        Escribe el JSON en disco e imprime la decisión como constancia en la
        salida del notebook.

    Por qué
    -------
    Separa "lo que se descubrió" (EDA) de "lo que se usa" (modelo) mediante un
    archivo explícito: el modelo no depende de valores copiados a mano y la
    cadena hallazgo -> decisión -> preprocesamiento es auditable.
    """
    dec = leer_decisiones()  # se parte del registro existente para no borrar otras decisiones
    dec[clave] = {"valor": valor, "motivo": motivo}
    # ensure_ascii=False conserva tildes y eñes legibles en el JSON
    ARCHIVO_DECISIONES.write_text(json.dumps(dec, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    print(f"[decisión registrada] {clave} = {valor}\n   motivo: {motivo}")


# Autoría del proyecto (se cita en el libro y en los entregables).
AUTORES = "María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)"
