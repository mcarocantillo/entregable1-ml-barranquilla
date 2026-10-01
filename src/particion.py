# -*- coding: utf-8 -*-
"""
Partición train/test y validación cruzada por BLOQUES ESPACIALES.

Por qué bloques: el estrato tiene autocorrelación espacial muy fuerte (Moran
≈ 0.91 entre edificios, cap. 8) y las unidades de un mismo edificio comparten
centroide y casi siempre estrato. Un split aleatorio pondría apartamentos del
mismo edificio en train y en test: el modelo "reconocería" el edificio en vez
de generalizar (notas del profesor Lihki 9.1.9 GroupKFold y 9.1.11 Spatial Block CV;
Roberts et al., 2017).

Por qué ESTRATIFICADA por grupos: hay desbalance (4.6:1 en train entre la clase más y
menos frecuente, cap. 4), así que se usa una partición estratificada por grupos:
respeta los bloques (nunca parte uno) y además intenta que cada fold tenga
proporciones de estrato parecidas a las del total.

Por qué una implementación propia (folds_estratificados_por_grupo) y no
StratifiedGroupKFold de scikit-learn: la asignación de bloques a folds de
StratifiedGroupKFold(shuffle=True) cambia entre versiones de scikit-learn. Al
ejecutar el libro en Google Colab (scikit-learn 1.6.1) el fold de test quedó
con el 46 % de estrato 3 y casi sin estratos 4-6 (divergencia de
Jensen-Shannon 0.12, frente a 0.009 con la versión 1.9.1 del equipo local). Una partición que
depende de la versión instalada no es reproducible. La función propia usa el
mismo algoritmo voraz de scikit-learn, pero con un generador aleatorio de
numpy de semilla fija, así que da los mismos folds en cualquier máquina.

Por qué un buffer: aun con bloques, los predios del borde de un bloque de
entrenamiento son vecinos inmediatos de los del bloque de validación contiguo.
Excluir del entrenamiento los puntos a menos de buffer_km de la validación
separa ambos conjuntos en el espacio y hace la estimación del error más
honesta (más parecida a predecir una zona de la ciudad no vista).
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from . import config as C


def asignar_bloques(x_km, y_km, tam_km=C.TAM_BLOQUE_KM):
    """Grilla regular de tam_km x tam_km sobre coordenadas proyectadas (km).

    Parámetros
    ----------
    x_km, y_km : array-like
        Coordenadas proyectadas en km (salida de limpieza.proyectar_km).
    tam_km : float, por defecto config.TAM_BLOQUE_KM
        Lado de cada celda cuadrada, en km.

    Devuelve
    --------
    numpy.ndarray de int
        Identificador de bloque por punto, codificado como fila * 1000 + columna.

    Por qué
    -------
    Se trabaja en km (no en grados) para que todos los bloques tengan el mismo
    tamaño físico en ambas direcciones. La grilla es una regla geométrica fija
    (no aprende nada del objetivo), así que puede calcularse sobre todos los
    datos sin fuga. Ojo: se asume que x_km e y_km no tienen NaN; un NaN
    produciría un identificador de bloque sin sentido al convertir a entero.
    """
    x_km = np.asarray(x_km, float)
    y_km = np.asarray(y_km, float)
    # Índice de columna (cx) y de fila (cy) de la celda, contando desde la
    # esquina suroeste de la nube de puntos; nanmin ignora faltantes al buscarla
    cx = np.floor((x_km - np.nanmin(x_km)) / tam_km).astype(int)
    cy = np.floor((y_km - np.nanmin(y_km)) / tam_km).astype(int)
    # Código único por celda (válido mientras haya menos de 1000 columnas,
    # holgadísimo para una ciudad de ~20 km con bloques de 2 km)
    return cy * 1000 + cx


def folds_estratificados_por_grupo(y, grupos, n_splits=C.N_FOLDS, seed=C.SEED):
    """K folds por grupos (bloques) con proporciones de clase parecidas.

    Parámetros
    ----------
    y : array-like
        Clase de cada fila (estrato).
    grupos : array-like
        Grupo de cada fila (bloque espacial). Un grupo nunca se reparte entre
        folds.
    n_splits : int, por defecto config.N_FOLDS
        Número de folds.
    seed : int, por defecto config.SEED
        Semilla del barajado inicial de los grupos.

    Devuelve
    --------
    list[tuple(numpy.ndarray, numpy.ndarray)]
        Por fold, (índices posicionales de entrenamiento, índices de validación).

    Por qué
    -------
    Es el algoritmo voraz de StratifiedGroupKFold (scikit-learn): se barajan
    los grupos con una semilla, se ordenan de más a menos "desbalanceado"
    (desviación estándar de sus conteos por clase) y cada grupo se asigna al
    fold donde la dispersión entre folds de las proporciones de cada clase
    queda más baja (en empate, al fold con menos filas). Se reimplementa con
    numpy.random.default_rng, cuyo resultado no depende de la versión de
    scikit-learn, para que la partición sea reproducible en cualquier entorno.
    """
    y = np.asarray(y)
    grupos = np.asarray(grupos)
    clases, y_cod = np.unique(y, return_inverse=True)
    g_unicos, g_cod = np.unique(grupos, return_inverse=True)
    # Conteo de filas por (grupo, clase)
    conteos = np.zeros((len(g_unicos), len(clases)))
    np.add.at(conteos, (g_cod, y_cod), 1)
    total_clase = conteos.sum(axis=0)
    # Barajado reproducible y orden estable por desbalance del grupo (desc.)
    orden = np.random.default_rng(seed).permutation(len(g_unicos))
    orden = orden[np.argsort(-conteos[orden].std(axis=1), kind="mergesort")]
    por_fold = np.zeros((n_splits, len(clases)))
    fold_de_grupo = np.empty(len(g_unicos), dtype=int)
    for g in orden:
        mejor, mejor_costo, mejor_tam = None, np.inf, np.inf
        for f in range(n_splits):
            por_fold[f] += conteos[g]
            # dispersión entre folds de la fracción de cada clase que tiene cada fold
            costo = np.mean(np.std(por_fold / total_clase, axis=0))
            por_fold[f] -= conteos[g]
            tam = por_fold[f].sum()
            if costo < mejor_costo - 1e-12 or (abs(costo - mejor_costo) <= 1e-12 and tam < mejor_tam):
                mejor, mejor_costo, mejor_tam = f, costo, tam
        por_fold[mejor] += conteos[g]
        fold_de_grupo[g] = mejor
    fold_fila = fold_de_grupo[g_cod]
    idx = np.arange(len(y))
    return [(idx[fold_fila != f], idx[fold_fila == f]) for f in range(n_splits)]


def particion_test_espacial(df, tam_km=C.TAM_BLOQUE_KM, seed=C.SEED):
    """Reserva ~20 % de los BLOQUES como test (1 de 5 folds de
    folds_estratificados_por_grupo). Devuelve una Serie 'train'/'test' alineada con df.

    Parámetros
    ----------
    df : pandas.DataFrame
        Datos limpios con 'x_km', 'y_km' y la columna objetivo. Debe tener un
        índice posicional coherente (salida de limpieza.limpiar), porque los
        índices que devuelve el splitter son posiciones.
    tam_km : float, por defecto config.TAM_BLOQUE_KM
        Lado de los bloques espaciales.
    seed : int, por defecto config.SEED
        Semilla del barajado de bloques (reproducibilidad).

    Devuelve
    --------
    tuple(pandas.Series, numpy.ndarray)
        - particion: "train" o "test" por fila, con el mismo índice que df.
        - bloques: identificador de bloque de cada fila (se guarda como
          columna 'bloque' para reutilizarlo en la validación cruzada).

    Por qué
    -------
    El test queda formado por zonas enteras de la ciudad que el modelo nunca
    vio, de modo que mide la capacidad de generalizar a nuevas zonas y no la
    de reconocer edificios o manzanas ya vistos. La estratificación mantiene
    en test una distribución de estratos parecida a la global, pese al
    desbalance de clases.
    """
    bloques = asignar_bloques(df["x_km"], df["y_km"], tam_km)
    # shuffle=True + random_state: asignación de bloques a folds aleatoria
    # pero reproducible
    # Solo se usa el PRIMER fold: su parte de validación (~1/5 de los bloques)
    # pasa a ser el test definitivo; el resto es train
    idx_train, idx_test = folds_estratificados_por_grupo(df[C.OBJETIVO], bloques, C.N_FOLDS, seed)[0]
    particion = pd.Series("train", index=df.index)
    particion.iloc[idx_test] = "test"  # iloc: los índices del splitter son posicionales
    return particion, bloques


def mascara_buffer(xy_train, xy_test, buffer_km):
    """True para los puntos de train que están a >= buffer_km de TODO punto de
    test (se conservan). Los que caen dentro del buffer se excluyen del
    entrenamiento para que el modelo no aprenda de vecinos inmediatos de la
    zona que se va a evaluar.

    Parámetros
    ----------
    xy_train : array-like de forma (n_train, 2)
        Coordenadas (x_km, y_km) de los puntos candidatos a entrenamiento.
    xy_test : array-like de forma (n_test, 2)
        Coordenadas (x_km, y_km) de los puntos de evaluación.
    buffer_km : float
        Radio de exclusión en km. Si es <= 0 no se excluye nada.

    Devuelve
    --------
    numpy.ndarray de bool, longitud n_train
        True = el punto de train se conserva.

    Por qué
    -------
    Solo se recorta el conjunto de ENTRENAMIENTO; el de evaluación queda
    intacto, así que la métrica se sigue calculando sobre la misma zona. La
    distancia es euclidiana porque las coordenadas están en km proyectados
    (error despreciable frente a haversine a escala urbana), lo que permite
    usar un KDTree eficiente.
    """
    # Caso trivial: sin buffer (o sin puntos de test) se conserva todo train
    if buffer_km <= 0 or len(xy_test) == 0:
        return np.ones(len(xy_train), dtype=bool)
    # KDTree sobre los puntos de test: consultar el vecino más cercano de cada
    # punto de train cuesta O(n log m) en lugar de O(n*m) con fuerza bruta
    arbol = KDTree(np.asarray(xy_test))
    # k=1: basta la distancia al punto de test MÁS cercano
    d, _ = arbol.query(np.asarray(xy_train), k=1)
    return d.ravel() >= buffer_km


def folds_espaciales_con_buffer(df_train, buffer_km, n_splits=C.N_FOLDS, seed=C.SEED,
                                verbose=True):
    """Folds de validación cruzada DENTRO de train: estratificados por
    bloque + buffer. Devuelve lista de (idx_train, idx_val) posicionales, lista
    que acepta directamente GridSearchCV(cv=...), cross_validate y
    learning_curve.

    Parámetros
    ----------
    df_train : pandas.DataFrame
        Solo el conjunto de entrenamiento, con 'x_km', 'y_km', 'bloque' y la
        columna objetivo. Los índices devueltos son posiciones sobre este
        DataFrame (conviene que venga con reset_index).
    buffer_km : float
        Radio de exclusión alrededor de cada fold de validación.
    n_splits : int, por defecto config.N_FOLDS
        Número de folds.
    seed : int, por defecto config.SEED
        Semilla del barajado de bloques.
    verbose : bool, por defecto True
        Si True, imprime por fold cuántos puntos de train se pierden por el
        buffer y cuántos bloques hay en validación.

    Devuelve
    --------
    list[tuple(numpy.ndarray, numpy.ndarray)]
        Un par (índices de entrenamiento tras el buffer, índices de validación)
        por fold.

    Por qué
    -------
    El test se reserva solo para la evaluación final; la selección de
    hiperparámetros usa estos folds, construidos con la misma lógica espacial
    (bloques + buffer) para que la estimación de validación no sea optimista
    respecto al test. Reutilizar la columna 'bloque' de la partición garantiza
    que los bloques de la CV coincidan con los de train/test.
    """
    xy = df_train[["x_km", "y_km"]].to_numpy()
    grupos = df_train["bloque"].to_numpy()  # mismos bloques que en la partición train/test
    y = df_train[C.OBJETIVO].to_numpy()
    folds = []
    for i, (itr, iva) in enumerate(folds_estratificados_por_grupo(y, grupos, n_splits, seed)):
        # Se quitan de itr los puntos a menos de buffer_km de cualquier punto de iva
        keep = mascara_buffer(xy[itr], xy[iva], buffer_km)
        folds.append((itr[keep], iva))
        if verbose:
            # Reporte del costo del buffer: % de train que se sacrifica en cada fold
            print(f"  fold {i}: train {len(itr):>7} -> {keep.sum():>7} tras buffer "
                  f"{buffer_km:.2f} km ({100 * (1 - keep.mean()):.1f}% excluido) | "
                  f"validación {len(iva):>6} | bloques val: {len(np.unique(grupos[iva]))}")
    return folds


def guardar_particion(df, path=None):
    """Guarda en disco el dataset de modelado con su columna 'particion'.

    Parámetros
    ----------
    df : pandas.DataFrame
        Datos limpios con las columnas 'particion' y 'bloque' ya asignadas.
    path : pathlib.Path, opcional
        Destino; por defecto datos/procesados/dataset_modelado.parquet.

    Devuelve
    --------
    pathlib.Path
        Ruta efectivamente escrita (.parquet o, como respaldo, .csv).

    Por qué
    -------
    La partición se calcula UNA vez y se persiste: todos los notebooks
    posteriores (EDA y modelo) leen exactamente la misma separación
    train/test, en lugar de recalcularla y arriesgar diferencias.
    """
    path = path or (C.CARPETA_PROCESADOS / "dataset_modelado.parquet")
    try:
        # Parquet conserva los tipos (texto, booleanos, enteros) sin ambigüedad
        df.to_parquet(path, index=False)
    except Exception:  # sin pyarrow/fastparquet: CSV como respaldo
        path = path.with_suffix(".csv")
        df.to_csv(path, index=False)
    return path


def cargar_particion(solo=None):
    """Carga el dataset de modelado con su columna 'particion'.
    solo='train' devuelve únicamente entrenamiento (lo que debe usar TODO el EDA).

    Parámetros
    ----------
    solo : {None, "train", "test"}, por defecto None
        None devuelve todas las filas; "train" o "test" filtra esa partición
        y reinicia el índice.

    Devuelve
    --------
    pandas.DataFrame
        Dataset de modelado, con las categóricas como object y faltantes como
        np.nan (formato compatible con scikit-learn).

    Por qué
    -------
    Que el EDA use solo train evita que decisiones tomadas mirando los datos
    (qué variables usar, qué transformar) estén informadas por el test, lo
    que sería una forma sutil de fuga.
    """
    p = C.CARPETA_PROCESADOS / "dataset_modelado.parquet"
    if p.exists():
        df = pd.read_parquet(p)
    else:
        # Respaldo CSV: los identificadores se fuerzan a texto para no perder
        # los ceros iniciales del número predial
        df = pd.read_csv(p.with_suffix(".csv"), dtype={"numero_predial_nacional": "string",
                                                       "npn_edificio": "string"})
    for col in C.CATEGORICAS + ["centroide_fuente"]:
        if col in df.columns:
            # object + np.nan (no pd.NA): SimpleImputer de scikit-learn falla con pd.NA
            df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    if solo is not None:
        df = df[df["particion"] == solo].reset_index(drop=True)  # índice posicional 0..n-1
    return df
