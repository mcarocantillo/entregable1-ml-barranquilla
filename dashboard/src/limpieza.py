# -*- coding: utf-8 -*-
"""
Carga y limpieza DETERMINISTA del CSV de catastro (antes de la partición).

Regla de oro de las instrucciones del proyecto (Sección 2, "Orden de trabajo"): toda decisión que
APRENDA algo de los datos (una mediana para imputar, un umbral de outliers
calculado con cuantiles, un escalado...) se calcula SOLO con el conjunto de
entrenamiento. Por eso aquí NO se imputa nada.

Lo que sí se hace aquí son reglas fila-a-fila que no dependen de ninguna otra
fila (y que se aplicarían igual a un predio nuevo):
  1. excluir usos que no son vivienda (garajes, depósitos),
  2. exigir area_construida > 0,
  3. excluir filas sin estrato residencial válido (es la variable OBJETIVO:
     no se imputa, imputarla inventaría la respuesta),
  4. marcar como NaN los valores físicamente imposibles (año 1512, 684
     habitaciones...) y dejar una bandera por variable. La imputación de esos
     NaN (mediana) ocurre dentro del Pipeline del modelo, ajustada solo con
     train (cambio respecto a la versión anterior de limpieza.py, que imputaba
     con la mediana de TODO el dataset: eso era una pequeña fuga).

Además se construyen variables derivadas que también son fila a fila
(antigüedad, identificador de edificio, coordenadas proyectadas en km y
distancia al centro). Como ninguna usa información de otras filas, pueden
calcularse sobre el dataset completo sin contaminar el conjunto de prueba.

Única excepción, deliberada: las viviendas sin coordenadas reciben el
centroide de su MANZANA, calculado con las demás viviendas ubicadas de esa
manzana (recuperar_por_manzana). Usa otras filas, pero solo su posición,
nunca el estrato: equivale a leer la posición de la manzana en la
cartografía, un dato que existiría igual para un predio nuevo.
"""

import numpy as np
import pandas as pd

from . import config as C


def extraer_estrato_numerico(serie_estrato):
    """'Bajo_Bajo_1' -> 1, 'Medio_Alto_5' -> 5; 'No_Aplica'/'Otro'/nulo -> NaN.

    Parámetros
    ----------
    serie_estrato : pandas.Series
        Columna 'estrato' en texto, tal como llega del catastro (ya normalizada
        con guiones bajos).

    Devuelve
    --------
    pandas.Series
        Estrato numérico (float, con NaN donde no hay un dígito final válido).

    Por qué
    -------
    El número del estrato está codificado al final de la etiqueta; extraerlo
    con una expresión regular es más robusto que un diccionario fijo de
    etiquetas, que fallaría ante una variante de escritura no prevista.
    Todo lo que no termine en "_<dígito>" queda como NaN y se excluye después.
    """
    return pd.to_numeric(
        # r"_(\d)$": captura el único dígito que va tras el último guion bajo
        serie_estrato.astype("string").str.strip().str.extract(r"_(\d)$")[0],
        errors="coerce")  # sin coincidencia -> NaN en lugar de error


def _normalizar_texto(s):
    """Quita espacios y unifica 'No Aplica' / 'No_Aplica'.

    Parámetros
    ----------
    s : pandas.Series
        Columna categórica en texto.

    Devuelve
    --------
    pandas.Series (dtype "string")
        Texto sin espacios en los extremos, con espacios internos reemplazados
        por "_" y con las cadenas vacías o "nan"/"None" convertidas en pd.NA.

    Por qué
    -------
    Una misma categoría escrita de dos formas se contaría como dos niveles
    distintos (y generaría dos columnas en el one-hot). Normalizar antes del
    EDA evita categorías duplicadas artificiales y faltantes "disfrazados"
    de texto.
    """
    s = s.astype("string").str.strip()
    s = s.str.replace(r"\s+", "_", regex=True)  # "No Aplica" -> "No_Aplica"
    # Cadenas que en realidad son faltantes (vienen de exportaciones previas)
    return s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def proyectar_km(lat, lon):
    """Proyección equirectangular local centrada en Barranquilla -> (x_km, y_km).
    A la escala de la ciudad (~20 km) el error frente a haversine es < 0.1 %
    (se verifica en el notebook espacial). Es equivalente a trabajar en un CRS
    proyectado, que es lo que piden las instrucciones del proyecto en vez de distancias euclidianas
    sobre grados.

    Parámetros
    ----------
    lat, lon : array-like
        Latitud y longitud en grados (WGS84). Pueden contener NaN.

    Devuelve
    --------
    tuple(numpy.ndarray, numpy.ndarray)
        (x_km, y_km): desplazamiento este y norte, en km, respecto al punto
        (C.CENTRO_LAT, C.CENTRO_LON). Los NaN se propagan.

    Por qué
    -------
    Un grado de longitud no mide lo mismo que un grado de latitud (a 11° N,
    1° de longitud ~ 0.98 veces 1° de latitud), así que la distancia euclidiana
    sobre grados distorsiona las distancias. Con coordenadas en km, KDTree,
    los bloques espaciales y el buffer pueden usar distancia euclidiana
    ordinaria con error despreciable, sin pagar el costo de haversine.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    # Norte-sur: diferencia de latitud en radianes por el radio terrestre
    y = (lat - C.CENTRO_LAT) * (np.pi / 180) * C.RADIO_TIERRA_KM
    # Este-oeste: igual, pero multiplicado por cos(latitud), porque los
    # meridianos convergen hacia los polos y un grado de longitud se acorta
    x = (lon - C.CENTRO_LON) * (np.pi / 180) * C.RADIO_TIERRA_KM * np.cos(np.radians(C.CENTRO_LAT))
    return x, y


def recuperar_por_manzana(df):
    """Asigna el centroide de su manzana a las viviendas sin coordenadas.

    Parámetros
    ----------
    df : pandas.DataFrame
        Datos con 'numero_predial_nacional' (30 dígitos), 'centroide_lat',
        'centroide_lon' y 'centroide_fuente', ya con las coordenadas fuera de
        la caja de Barranquilla anuladas.

    Devuelve
    --------
    tuple(pandas.DataFrame, pandas.DataFrame)
        - df con 'npn_manzana' (17 dígitos) y 'ubicado_por_manzana' (bool), y
          con lat/lon y 'centroide_fuente' = "centroide_manzana" en las filas
          recuperadas.
        - manzanas: una fila por manzana con viviendas ubicadas: número de
          puntos (edificios o lotes distintos), centroide, distancia máxima de
          un punto al centroide (m) y si es válida para recuperar.

    Por qué
    -------
    Los predios informales (mejoras en terreno ajeno) no tienen terreno propio
    ni edificio matriz de donde heredar un centroide, pero su NPN sí dice en
    qué manzana están, y casi todas esas manzanas tienen otras viviendas
    ubicadas. El centroide se calcula con los puntos DISTINTOS de la manzana
    (un edificio de 300 apartamentos cuenta una vez, para no arrastrar el
    centro hacia él). Solo se usan coordenadas, nunca el estrato, y la
    posición de una manzana es un dato cartográfico que también existiría
    para una vivienda nueva; por eso puede hacerse antes de la partición sin
    fuga. Dos reglas fijas (config.py) descartan los códigos que no
    describen un lugar compacto: la manzana "0000" (comodín) y las manzanas
    con algún punto a más de DISPERSION_MAX_MANZANA_M del centroide.
    """
    df = df.copy()
    df["npn_manzana"] = df["numero_predial_nacional"].str[:17]
    lat, lon = df["centroide_lat"], df["centroide_lon"]
    ubicada = lat.notna() & lon.notna()
    # Puntos distintos por manzana (un edificio = un punto)
    pts = (df.loc[ubicada, ["npn_manzana", "centroide_lat", "centroide_lon"]]
           .drop_duplicates())
    manz = pts.groupby("npn_manzana").agg(n_puntos=("centroide_lat", "size"),
                                          lat=("centroide_lat", "mean"),
                                          lon=("centroide_lon", "mean"))
    # Distancia de cada punto al centroide de su manzana (en km proyectados)
    px, py = proyectar_km(pts["centroide_lat"], pts["centroide_lon"])
    cx, cy = proyectar_km(manz.loc[pts["npn_manzana"], "lat"], manz.loc[pts["npn_manzana"], "lon"])
    pts["d_m"] = 1000 * np.hypot(px - cx, py - cy)
    manz["dist_max_m"] = pts.groupby("npn_manzana")["d_m"].max()
    manz["generica"] = manz.index.str[13:17] == C.CODIGO_MANZANA_GENERICA
    manz["valida"] = ~manz["generica"] & (manz["dist_max_m"] <= C.DISPERSION_MAX_MANZANA_M)
    # Filas sin coordenadas cuya manzana es válida -> centroide de la manzana
    validas = manz[manz["valida"]]
    recuperar = ~ubicada & df["npn_manzana"].isin(validas.index)
    df["ubicado_por_manzana"] = recuperar
    df.loc[recuperar, "centroide_lat"] = df.loc[recuperar, "npn_manzana"].map(validas["lat"])
    df.loc[recuperar, "centroide_lon"] = df.loc[recuperar, "npn_manzana"].map(validas["lon"])
    df.loc[recuperar, "centroide_fuente"] = "centroide_manzana"
    return df, manz


def cargar_crudo(path=None):
    """Lee el CSV de catastro sin aplicar ninguna regla de limpieza.

    Parámetros
    ----------
    path : str o pathlib.Path, opcional
        Ruta al CSV. Si es None se usa config.ruta_csv().

    Devuelve
    --------
    pandas.DataFrame
        Datos crudos con una columna adicional 'id_fila' (0..n-1) al inicio.

    Por qué
    -------
    - El número predial nacional se lee como texto: leído como número perdería
      los ceros iniciales y la precisión de sus 30 dígitos.
    - 'id_fila' da un identificador estable de cada registro original, útil
      para rastrear filas a lo largo del embudo; nunca entra al modelo
      (está en config.IDENTIFICADORES).
    """
    path = path or C.ruta_csv()
    # low_memory=False: pandas infiere el tipo de cada columna con el archivo
    # completo y no por trozos, evitando columnas con tipos mezclados
    df = pd.read_csv(path, dtype={"numero_predial_nacional": "string"}, low_memory=False)
    df.insert(0, "id_fila", np.arange(len(df)))
    return df


def limpiar(df_crudo, verbose=True):
    """Aplica las reglas deterministas. Devuelve (df_limpio, embudo), donde
    embudo es una tabla con cuántas filas quedan tras cada paso y por qué.

    Parámetros
    ----------
    df_crudo : pandas.DataFrame
        Salida de cargar_crudo(). No se modifica (se trabaja sobre una copia).
    verbose : bool, por defecto True
        Si True, imprime la tabla del embudo.

    Devuelve
    --------
    tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
        - df_limpio: filas que sobreviven, con índice reiniciado, el objetivo
          'estrato_num' (int), banderas 'flag_imposible_<var>',
          'flag_incoherente_<var>' y
          'flag_coord_fuera_caja', y las variables derivadas (antiguedad,
          npn_edificio, npn_manzana, ubicado_por_manzana, x_km, y_km,
          dist_centro_km, tiene_coordenadas).
        - embudo: pasos de filtrado con filas restantes, cambio y % del total.
        - tabla_imposibles: por variable, rango plausible y cantidad/% de
          valores anulados por imposibles.

    Por qué
    -------
    Todas las reglas son fila a fila y con umbrales fijados de antemano (en
    config.py), no estimados con los datos. Por eso pueden aplicarse ANTES de
    partir en train/test sin fuga: el resultado para una fila no depende de
    ninguna otra. La excepción es el paso 6 (centroide de la manzana), que
    usa las coordenadas de otras filas pero nunca su estrato. El embudo documenta cuánto se pierde en cada paso, lo que
    permite juzgar si la limpieza sesga la muestra.
    """
    df = df_crudo.copy()  # no se altera el DataFrame crudo del notebook
    # El embudo se arma como lista de tuplas (paso, filas restantes, motivo)
    embudo = [("0. CSV cargado (una fila por unidad residencial física)", len(df), "")]

    # Normalización de texto en categóricas (no cambia el número de filas)
    for col in ["uso", "condicion_predio", "destinacion_economica", "tipo_planta",
                "estrato", "centroide_fuente"]:
        if col in df.columns:  # tolera versiones del CSV sin alguna columna
            df[col] = _normalizar_texto(df[col])
    if "tipo_vivienda" in df.columns:
        # llega como código numérico del dominio ArcGIS (ej. 3.0, 4.0): se trata
        # como categoría, no como número (no tiene orden ni distancia)
        tv = pd.to_numeric(df["tipo_vivienda"], errors="coerce")
        # El prefijo "tv_" impide que algún paso posterior lo reinterprete como número
        df["tipo_vivienda"] = tv.map(lambda v: pd.NA if pd.isna(v) else f"tv_{int(v)}").astype("string")

    # Paso 1: fuera garajes y depósitos (no son hogares, no tienen estrato propio)
    n = len(df)
    df = df[~df["uso"].isin(C.USOS_NO_HABITABLES)]
    embudo.append(("1. Excluir usos no habitables (garajes, depósitos)", len(df),
                   "no son vivienda: fuera del alcance del problema"))

    # Paso 2: sin área construida positiva no hay unidad de vivienda real
    area = pd.to_numeric(df["area_construida"], errors="coerce")
    df = df[area.notna() & (area > 0)]
    embudo.append(("2. Exigir area_construida > 0", len(df),
                   "sin área construida no hay unidad de vivienda que describir"))

    # Paso 2b: área mínima de una vivienda. Unidades de pocos m2 (muchas en PH
    # con piso "99") son parqueaderos o depósitos mal clasificados como vivienda
    # (diagnóstico de rangos, bitácora A). Regla fija de dominio, fila a fila.
    area = pd.to_numeric(df["area_construida"], errors="coerce")
    df = df[area >= C.AREA_MIN_VIVIENDA]
    embudo.append((f"2b. Excluir unidades con área < {C.AREA_MIN_VIVIENDA} m2", len(df),
                   "no son vivienda: probables parqueaderos o depósitos con uso residencial"))

    # Paso 2c: registros agregados (una fila = un edificio completo). Superar
    # CUALQUIERA de los umbrales indica que la fila no describe una vivienda.
    agregado = pd.Series(False, index=df.index)
    for col, umbral in C.UMBRALES_REGISTRO_AGREGADO.items():
        agregado |= pd.to_numeric(df[col], errors="coerce") > umbral
    df = df[~agregado]
    embudo.append(("2c. Excluir registros agregados (edificio completo en una fila)", len(df),
                   "habitaciones > {total_habitaciones}, baños > {total_banios} o área > "
                   "{area_construida} m2: describen un edificio, no una vivienda".format(
                       **C.UMBRALES_REGISTRO_AGREGADO)))

    # Paso 3: construir el objetivo y quedarse solo con estratos 1-6.
    # El objetivo NUNCA se imputa: sería inventar la respuesta que el modelo
    # debe predecir.
    df = df.copy()  # copia explícita tras el filtrado (evita SettingWithCopyWarning)
    df[C.OBJETIVO] = extraer_estrato_numerico(df["estrato"])
    df = df[df[C.OBJETIVO].isin(C.CLASES)].copy()
    df[C.OBJETIVO] = df[C.OBJETIVO].astype(int)  # ya sin NaN, se puede pasar a entero
    embudo.append(("3. Excluir estrato no residencial/nulo (variable objetivo)", len(df),
                   "No_Aplica/Otro/nulo: imputar el objetivo inventaría la respuesta"))

    # 4. Valores físicamente imposibles -> NaN + bandera (no elimina filas)
    # Se anula solo el valor erróneo y se conserva el resto de la fila: eliminarla
    # completa desperdiciaría información válida de las demás variables.
    num_cols = ["anio_construccion", "planta_ubicacion", "altura"] + list(C.LIMITES_PLAUSIBLES)
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # texto no numérico -> NaN
    # Reglas = límites de config + el rango de años (que depende del año actual)
    reglas = dict(C.LIMITES_PLAUSIBLES)
    reglas["anio_construccion"] = (C.ANIO_MIN_PLAUSIBLE, C.ANIO_MAX_PLAUSIBLE)
    resumen_imposibles = []
    for col, (mn, mx) in reglas.items():
        # Un NaN original NO cuenta como imposible: es un faltante, no un error
        fuera = df[col].notna() & ~df[col].between(mn, mx)
        # La bandera conserva la información de que hubo un error de captura;
        # el EDA puede verificar si ese error se asocia con el estrato
        df[f"flag_imposible_{col}"] = fuera
        resumen_imposibles.append((col, f"[{mn}, {mx}]", int(fuera.sum()),
                                   100 * fuera.mean()))
        # NaN y no mediana: la imputación se aprende dentro del Pipeline, solo con train
        df.loc[fuera, col] = np.nan
    n_con_alguno = int(df.filter(like="flag_imposible_").any(axis=1).sum())
    embudo.append(("4. Valores imposibles -> NaN (se imputan luego, dentro del Pipeline)",
                   len(df), f"{n_con_alguno} filas tenían al menos un valor imposible; "
                            "no se elimina ninguna fila"))

    # 5. Coherencia dentro de la fila -> NaN + bandera (no elimina filas).
    # Cada valor puede estar en rango y aun así ser imposible DADO el resto de la
    # fila. Se anula el valor más sospechoso de la pareja y se conserva la fila.
    hab = df["total_habitaciones"]
    banios_incoh = df["total_banios"] > hab + C.MAX_BANIOS_SOBRE_HAB
    hab_incoh = (hab > 0) & (df["area_construida"] / hab < C.MIN_AREA_POR_HABITACION)
    df["flag_incoherente_total_banios"] = banios_incoh.fillna(False)
    df["flag_incoherente_total_habitaciones"] = hab_incoh.fillna(False)
    df.loc[df["flag_incoherente_total_banios"], "total_banios"] = np.nan
    df.loc[df["flag_incoherente_total_habitaciones"], "total_habitaciones"] = np.nan
    resumen_imposibles.append(("total_banios (coherencia)",
                               f"<= habitaciones + {C.MAX_BANIOS_SOBRE_HAB}",
                               int(banios_incoh.sum()), 100 * banios_incoh.mean()))
    resumen_imposibles.append(("total_habitaciones (coherencia)",
                               f"área / habitación >= {C.MIN_AREA_POR_HABITACION} m2",
                               int(hab_incoh.sum()), 100 * hab_incoh.mean()))
    n_incoh = int((df["flag_incoherente_total_banios"] | df["flag_incoherente_total_habitaciones"]).sum())
    embudo.append(("5. Incoherencias dentro de la fila -> NaN", len(df),
                   f"{n_incoh} filas con baños o habitaciones incoherentes con el resto de la fila; "
                   "no se elimina ninguna fila"))

    # Variables derivadas (fila a fila, sin mirar otras filas)
    # Antigüedad en años: más interpretable que el año; clip(0) por seguridad
    df["antiguedad"] = (C.ANIO_ACTUAL - df["anio_construccion"]).clip(lower=0)
    # Si el NPN pasó por un float en alguna exportación, quedó con ".0" al final
    npn = df["numero_predial_nacional"].astype("string").str.replace(r"\.0$", "", regex=True)
    # si el NPN se guardó alguna vez como número, pierde el 0 inicial ("08..."):
    # se restituye para que el prefijo de 22 dígitos identifique bien el edificio
    npn = npn.where(npn.isna() | (npn.str.len() >= 30), npn.str.zfill(30))
    df["numero_predial_nacional"] = npn
    # Los primeros 22 dígitos del NPN identifican el predio matriz (edificio);
    # sirve para auditar que unidades del mismo edificio no queden a ambos lados
    # de la partición. Es identificador: nunca entra al modelo.
    df["npn_edificio"] = npn.str[:22]
    lat = pd.to_numeric(df["centroide_lat"], errors="coerce")
    lon = pd.to_numeric(df["centroide_lon"], errors="coerce")
    # Coordenadas fuera de la caja de Barranquilla = error de georreferenciación
    fuera_caja = lat.notna() & ~(lat.between(C.LAT_MIN, C.LAT_MAX) & lon.between(C.LON_MIN, C.LON_MAX))
    df["flag_coord_fuera_caja"] = fuera_caja
    # Se anulan (no se corrigen): un punto erróneo distorsionaría bloques y vecinos
    lat[fuera_caja] = np.nan
    lon[fuera_caja] = np.nan
    df["centroide_lat"], df["centroide_lon"] = lat, lon
    # 6. Viviendas sin coordenadas -> centroide de su manzana (si es compacta)
    n_sin = int((lat.isna() | lon.isna()).sum())
    df, _ = recuperar_por_manzana(df)
    n_rec = int(df["ubicado_por_manzana"].sum())
    embudo.append(("6. Sin coordenadas -> centroide de su manzana", len(df),
                   f"{n_rec} de {n_sin} filas sin coordenadas reciben el centroide de su manzana; "
                   "no se elimina ninguna fila"))
    lat, lon = df["centroide_lat"], df["centroide_lon"]
    # Coordenadas en km para poder usar distancia euclidiana (ver proyectar_km)
    x, y = proyectar_km(lat, lon)
    df["x_km"], df["y_km"] = x, y
    # Distancia al centro histórico: resume en una variable el gradiente
    # centro-periferia del estrato; como el origen es el centro, es la norma de (x, y)
    df["dist_centro_km"] = np.sqrt(x ** 2 + y ** 2)
    # Indicador para filtrar filas utilizables en los análisis espaciales
    df["tiene_coordenadas"] = lat.notna() & lon.notna()

    # Tabla del embudo: filas restantes, cambio respecto al paso anterior y
    # porcentaje respecto al CSV original
    embudo = pd.DataFrame(embudo, columns=["paso", "filas", "motivo"])
    embudo["cambio"] = embudo["filas"].diff().fillna(0).astype(int)
    embudo["% del total inicial"] = (100 * embudo["filas"] / embudo["filas"].iloc[0]).round(2)
    tabla_imposibles = pd.DataFrame(resumen_imposibles,
                                    columns=["variable", "rango plausible", "n imposibles", "%"])
    if verbose:
        print(embudo.to_string(index=False))
    # reset_index: índice 0..n-1 contiguo, necesario porque la partición y los
    # folds trabajan con posiciones (iloc)
    return df.reset_index(drop=True), embudo, tabla_imposibles


def cargar_datos_limpios(path=None, verbose=False):
    """Atajo: carga el CSV crudo y aplica limpiar() en un solo paso.

    Parámetros
    ----------
    path : str o pathlib.Path, opcional
        Ruta al CSV; None usa config.ruta_csv().
    verbose : bool, por defecto False
        Si True, imprime el embudo de limpieza.

    Devuelve
    --------
    pandas.DataFrame
        Solo el DataFrame limpio (se descartan el embudo y la tabla de
        imposibles, útiles únicamente en el notebook de limpieza).
    """
    df, _, _ = limpiar(cargar_crudo(path), verbose=verbose)
    return df
