# -*- coding: utf-8 -*-
"""
Herramientas espaciales implementadas con numpy/scipy/scikit-learn (sin
geopandas/esda/libpysal, que dependen de binarios compilados y suelen tardar
en soportar versiones nuevas de Python como la 3.14). Las fórmulas son las
estándar (Moran 1950; Anselin 1995 para LISA; Getis & Ord 1992 para Gi*;
Clark & Evans 1954; Ripley 1976).

Organización del módulo
-----------------------
- Distancias y utilidades: haversine, muestreo estratificado y deduplicación
  por edificio (evita pseudo-replicación de unidades que comparten punto).
- Matriz de pesos W por k vecinos más cercanos.
- Autocorrelación global (I de Moran), local (LISA, Gi*) y por distancia
  (correlograma / semivariograma).
- Patrón de puntos: Clark-Evans, L de Ripley y DBSCAN con distancia haversine.
- Escala / MAUP: agregación en grilla y Moran por tamaño de celda.

Por qué importa para el modelo: si el estrato está autocorrelacionado en el
espacio (vecinos parecidos), una validación cruzada aleatoria mezclaría en
train y test predios casi idénticos del mismo sector y sobreestimaría el
desempeño. Estas funciones cuantifican esa dependencia y su alcance, lo que
justifica la validación cruzada por BLOQUES espaciales y el tamaño del buffer.

Convención de coordenadas: x_km, y_km son coordenadas PROYECTADAS en
kilómetros (distancia euclidiana válida a escala urbana); lat/lon en grados
se usan solo con fórmulas esféricas (haversine).
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors

from . import config as C


# ---------------------------------------------------------------------------
# Distancias
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia de gran círculo (en km) entre puntos dados en grados.

    Parámetros
    ----------
    lat1, lon1, lat2, lon2 : float o array-like
        Latitudes y longitudes en GRADOS decimales (se admiten arreglos
        con broadcasting de numpy).

    Devuelve
    --------
    float o numpy.ndarray
        Distancia en km sobre una esfera de radio C.RADIO_TIERRA_KM
        (radio medio terrestre, 6371.0088 km).

    Fórmula
    -------
    a = sin²(Δφ/2) + cos φ1 · cos φ2 · sin²(Δλ/2)
    d = 2 R · arcsin( sqrt(a) )
    con φ = latitud y λ = longitud en radianes.

    Por qué
    -------
    La fórmula haversine es numéricamente estable para distancias cortas
    (a diferencia de la ley esférica de cosenos), lo que importa a escala
    urbana. El error por suponer la Tierra esférica (< 0.5 %) es despreciable
    para este análisis.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))  # grados -> radianes
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * C.RADIO_TIERRA_KM * np.arcsin(np.sqrt(a))


def muestra_estratificada(df, n, col=C.OBJETIVO, seed=C.SEED):
    """Muestra con las mismas proporciones de 'col' que df (loop simple: evita
    el cambio de comportamiento de groupby.apply en pandas >= 2.2).

    Parámetros
    ----------
    df : pandas.DataFrame
        Datos completos.
    n : int
        Tamaño aproximado de la muestra deseada.
    col : str, default C.OBJETIVO
        Columna por la que se estratifica (el estrato).
    seed : int, default C.SEED
        Semilla para reproducibilidad.

    Devuelve
    --------
    pandas.DataFrame
        Muestra con índice reiniciado. Si n >= len(df) devuelve una copia
        completa. El tamaño final puede diferir ligeramente de n por el
        redondeo por grupo.

    Por qué
    -------
    Algunos análisis espaciales son O(n²) (correlograma, Ripley, LISA con
    permutaciones) y requieren submuestrear. Estratificar conserva la
    distribución del estrato, de modo que las clases minoritarias (p. ej. el
    estrato 6) no desaparezcan de la muestra. Cada estrato recibe al menos
    1 fila (max(1, ...)).
    """
    if n >= len(df):
        return df.copy()
    partes = []
    for _, g in df.groupby(col):
        k = max(1, round(n * len(g) / len(df)))  # cuota proporcional al tamaño del grupo
        partes.append(g.sample(n=min(k, len(g)), random_state=seed))
    return pd.concat(partes).reset_index(drop=True)


def deduplicar_por_edificio(df):
    """Una fila por edificio (prefijo NPN de 22 dígitos) / por coordenada.
    Las unidades de un mismo edificio comparten centroide propagado y casi
    siempre estrato: si no se deduplican, inflan artificialmente Moran's I.

    Parámetros
    ----------
    df : pandas.DataFrame
        Debe contener "npn_edificio", "x_km" y "y_km".

    Devuelve
    --------
    pandas.DataFrame
        Una fila por clave de edificio con la MEDIA de cada columna numérica
        (incluidas x_km, y_km y el estrato, que en un edificio suele ser
        constante). Las columnas no numéricas se descartan.

    Por qué
    -------
    Un edificio de 100 apartamentos aporta 100 puntos en la misma coordenada
    con el mismo estrato: son vecinos entre sí a distancia 0 y con valores
    idénticos, lo que fabrica autocorrelación (pseudo-replicación). Tomar una
    fila por edificio mide la dependencia ENTRE edificios, que es la
    relevante. Si falta el NPN, la clave de respaldo es la coordenada
    redondeada a 4 decimales de km (0.1 m).
    """
    d = df.copy()
    # clave = NPN del edificio; si es nulo, "x_y" de la coordenada redondeada
    clave = d["npn_edificio"].astype("string").fillna(
        d["x_km"].round(4).astype(str) + "_" + d["y_km"].round(4).astype(str))
    d["_clave_edificio"] = clave
    # se promedian todas las columnas numéricas dentro de cada edificio
    agg = {c: "mean" for c in d.select_dtypes("number").columns if c != "_clave_edificio"}
    out = d.groupby("_clave_edificio", as_index=False).agg(agg)
    return out


# ---------------------------------------------------------------------------
# Matriz de pesos
# ---------------------------------------------------------------------------

def pesos_knn(xy, k=C.K_VECINOS, estandarizar_filas=True):
    """W de k vecinos más cercanos sobre coordenadas proyectadas (km).
    k-vecinos se elige frente a 'banda de distancia' porque la densidad de
    predios es muy desigual (centro denso, periferia dispersa): una banda fija
    dejaría puntos sin vecinos en la periferia y con cientos en el centro;
    kNN garantiza el mismo número de vecinos para todos.

    Parámetros
    ----------
    xy : array-like (n, 2)
        Coordenadas proyectadas en km.
    k : int, default C.K_VECINOS (8)
        Número de vecinos por punto (sin contarse a sí mismo).
    estandarizar_filas : bool, default True
        Si True, w_ij = 1/k para cada vecino (cada fila suma 1); si False,
        pesos binarios w_ij = 1.

    Devuelve
    --------
    tuple (W, dist, idx)
        W : scipy.sparse.csr_matrix (n, n), sin diagonal (w_ii = 0).
        dist : (n, k) distancias a los k vecinos (km), de menor a mayor.
        idx : (n, k) índices de los k vecinos de cada punto.

    Por qué fila-estandarizada
    --------------------------
    Con W fila-estandarizada, (W z)_i es el PROMEDIO de los vecinos de i
    (el "rezago espacial"), lo que hace interpretable el diagrama de
    dispersión de Moran y comparables los puntos. Nota: la W kNN NO es
    simétrica (que j sea vecino de i no implica lo contrario).
    """
    xy = np.asarray(xy, float)
    # se piden k+1 vecinos porque el más cercano de cada punto es él mismo (distancia 0)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    dist, idx = nn.kneighbors(xy)
    dist, idx = dist[:, 1:], idx[:, 1:]  # se descarta la columna 0 (el propio punto)
    # OJO: si hay puntos duplicados, la columna 0 podría ser otro punto en la misma
    # coordenada; por eso conviene deduplicar (deduplicar_por_edificio) antes de W
    n = len(xy)
    filas = np.repeat(np.arange(n), k)  # índice de fila i repetido k veces (formato COO)
    val = np.full(n * k, 1.0 / k if estandarizar_filas else 1.0)
    W = sparse.csr_matrix((val, (filas, idx.ravel())), shape=(n, n))
    return W, dist, idx


# ---------------------------------------------------------------------------
# Autocorrelación global y local
# ---------------------------------------------------------------------------

def moran_global(x, W, n_perm=999, seed=C.SEED):
    """I de Moran global con inferencia por permutaciones.

    Parámetros
    ----------
    x : array-like (n,)
        Variable a evaluar (p. ej. el estrato de cada edificio).
    W : matriz (n, n), típicamente scipy.sparse
        Matriz de pesos espaciales (p. ej. de pesos_knn).
    n_perm : int, default 999
        Número de permutaciones aleatorias de x sobre las ubicaciones.
    seed : int, default C.SEED
        Semilla del generador.

    Devuelve
    --------
    dict
        I observado, E[I] teórico bajo H0, media y desviación de la
        distribución de permutación, z (estandarizado con la permutación),
        pseudo p-valor, n y n_perm.

    Fórmula
    -------
    I = (n / S0) · (z' W z) / (z' z),  con z = x - x̄ y S0 = sum_ij w_ij.
    E[I] = -1/(n-1) bajo H0 de aleatoriedad espacial.
    Pseudo p (Hope 1968): p = (#{I_perm tan o más extremos que I} + 1) / (n_perm + 1),
    unilateral en la dirección observada (si I está por encima de la media de
    la permutación se cuentan los >= I; si está por debajo, los <= I).

    Interpretación
    --------------
    I > E[I]: autocorrelación positiva (valores similares se agrupan);
    I ≈ E[I]: patrón aleatorio; I < E[I]: autocorrelación negativa (patrón
    tipo tablero de ajedrez). Con 999 permutaciones el p mínimo es 0.001.

    Por qué permutaciones
    ---------------------
    La inferencia analítica supone normalidad o una W simétrica; la
    aleatorización (reubicar al azar los valores entre las ubicaciones) no
    requiere supuestos distribucionales y es válida para el estrato, que es
    discreto y ordinal.
    """
    x = np.asarray(x, float)
    n = len(x)
    z = x - x.mean()  # desviaciones respecto a la media
    S0 = W.sum()  # suma de todos los pesos (= n si W está fila-estandarizada)
    I = (n / S0) * (z @ (W @ z)) / (z @ z)
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        zp = rng.permutation(z)  # reasigna los valores al azar entre las ubicaciones (H0)
        perm[i] = (n / S0) * (zp @ (W @ zp)) / (zp @ zp)
    # pseudo p unilateral en la dirección del valor observado; el +1 incluye
    # al observado como una permutación más (evita p = 0)
    if I >= perm.mean():
        p = (np.sum(perm >= I) + 1) / (n_perm + 1)
    else:
        p = (np.sum(perm <= I) + 1) / (n_perm + 1)
    z_score = (I - perm.mean()) / perm.std()  # z respecto a la distribución de permutación
    return {"I": I, "E[I]": -1 / (n - 1), "media_perm": perm.mean(), "sd_perm": perm.std(),
            "z": z_score, "p_perm": p, "n": n, "n_perm": n_perm}


def lisa(x, idx_vecinos, n_perm=199, seed=C.SEED, alpha=0.05):
    """Moran local (Anselin 1995) con pesos kNN fila-estandarizados y
    pseudo p-valor por aleatorización condicional (se fija z_i y se permutan
    los valores de sus vecinos). Devuelve DataFrame con I_i, p, cuadrante.

    Parámetros
    ----------
    x : array-like (n,)
        Variable a evaluar.
    idx_vecinos : numpy.ndarray (n, k)
        Índices de los k vecinos de cada punto (salida idx de pesos_knn).
    n_perm : int, default 199
        Permutaciones por punto (p mínimo = 1/200 = 0.005).
    seed : int, default C.SEED
        Semilla del generador.
    alpha : float, default 0.05
        Nivel para declarar un punto significativo.

    Devuelve
    --------
    pandas.DataFrame
        I_local, lag (promedio de z en los vecinos), z (valor estandarizado),
        p (pseudo p-valor) y cuadrante: HH, LL, HL, LH o "no significativo".

    Fórmula
    -------
    z_i = (x_i - x̄) / s;   lag_i = (1/k) sum_{j in N(i)} z_j;   I_i = z_i · lag_i.
    Pseudo p bilateral: p_i = (#{|I_i^perm| >= |I_i|} + 1) / (n_perm + 1).

    Interpretación de cuadrantes (diagrama de dispersión de Moran)
    --------------------------------------------------------------
    HH: valor alto rodeado de altos (clúster de estratos altos);
    LL: bajo rodeado de bajos; HL / LH: atípicos espaciales (un valor
    distinto a su entorno). Solo se etiquetan los puntos con p < alpha.

    Por qué excluir i en la permutación
    -----------------------------------
    La aleatorización es CONDICIONAL: se mantiene fijo z_i en su lugar y se
    reasignan al azar los valores de los OTROS n-1 puntos a su vecindario.
    Si i pudiera aparecer como su propio vecino, se introduciría una
    correlación positiva artificial entre z_i y su lag permutado.

    Advertencia
    -----------
    Se hacen n pruebas simultáneas (una por punto), por lo que con
    alpha = 0.05 se esperan ~5 % de puntos "significativos" por azar; los
    mapas LISA se leen como exploratorios.
    """
    x = np.asarray(x, float)
    z = (x - x.mean()) / x.std()  # estandarización (s poblacional, ddof=0)
    k = idx_vecinos.shape[1]
    lag = z[idx_vecinos].mean(axis=1)  # = (W z)_i con W fila-estandarizada
    I = z * lag
    rng = np.random.default_rng(seed)
    n = len(z)
    mas_extremos = np.zeros(n)  # contador por punto de permutaciones tan o más extremas
    fila = np.arange(n)[:, None]  # índice i de cada fila, como columna para broadcasting
    for _ in range(n_perm):  # vectorizado: una permutación simultánea para TODOS los puntos
        # k vecinos al azar entre los OTROS n-1 puntos (se excluye i); el
        # muestreo es con reemplazo, pero con n grande la probabilidad de
        # repetir un índice es despreciable (~k^2 / 2n)
        j = rng.integers(0, n - 1, size=(n, k))  # enteros en [0, n-2]
        j = j + (j >= fila)  # desplaza +1 los >= i: mapea [0, n-2] a {0..n-1} \ {i}
        muestra = z[j].mean(axis=1)  # lag bajo H0 (vecindario aleatorio)
        Ip = z * muestra  # z_i se mantiene fijo (aleatorización condicional)
        mas_extremos += (np.abs(Ip) >= np.abs(I))  # prueba bilateral
    p = (mas_extremos + 1) / (n_perm + 1)
    # cuadrante según el signo de z_i (fila) y de su lag (columna)
    cuad = np.where(z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
    cuad = np.where(p < alpha, cuad, "no significativo")
    return pd.DataFrame({"I_local": I, "lag": lag, "z": z, "p": p, "cuadrante": cuad})


def getis_ord_gi_star(x, idx_vecinos):
    """Gi* (Getis & Ord 1992) con pesos binarios kNN que INCLUYEN al propio
    punto (eso lo distingue de Gi). Devuelve z-scores: > 1.96 hotspot, < -1.96
    coldspot al 95 %.

    Parámetros
    ----------
    x : array-like (n,)
        Variable a evaluar (se asume no negativa, como el estrato).
    idx_vecinos : numpy.ndarray (n, k)
        Índices de los k vecinos (sin el propio punto); la función le suma
        el punto i, de modo que cada vecindario tiene k+1 elementos.

    Devuelve
    --------
    numpy.ndarray (n,)
        Estadístico Gi* de cada punto, ya en escala z.

    Fórmula (pesos binarios, w_ij = 1 para los k' = k+1 elementos)
    ---------------------------------------------------------------
    Gi* = ( sum_j w_ij x_j - x̄ · sum_j w_ij ) /
          ( S · sqrt( [n · sum_j w_ij² - (sum_j w_ij)²] / (n - 1) ) )
    con S = sqrt( (1/n) sum x_j² - x̄² ). Con pesos binarios
    sum w_ij = sum w_ij² = k', y el denominador queda
    S · sqrt( (n k' - k'²) / (n - 1) ).

    Diferencia con LISA
    -------------------
    LISA compara cada punto con sus vecinos (detecta clústeres Y atípicos);
    Gi* mide si la SUMA local (punto + vecinos) es anormalmente alta o baja
    (solo detecta concentraciones de valores altos o bajos: hot/cold spots).
    """
    x = np.asarray(x, float)
    n = len(x)
    k = idx_vecinos.shape[1] + 1  # k' = vecinos + el propio punto (lo que hace "estrella")
    suma = x + x[idx_vecinos].sum(axis=1)  # suma local sum_j w_ij x_j
    xbar = x.mean()
    s = np.sqrt((x ** 2).mean() - xbar ** 2)  # desviación estándar poblacional
    num = suma - xbar * k  # suma observada menos la esperada bajo H0
    den = s * np.sqrt((n * k - k ** 2) / (n - 1))
    return num / den


def correlograma(xy, x, bandas_km, n_max=4000, seed=C.SEED):
    """Moran's I y semivarianza por bandas de distancia (pares de puntos).
    Sirve para ver hasta qué distancia 'se parecen' los vecinos: el alcance
    (range) orienta el tamaño del buffer de la validación espacial.

    Parámetros
    ----------
    xy : array-like (n, 2)
        Coordenadas proyectadas en km.
    x : array-like (n,)
        Variable a evaluar.
    bandas_km : array-like
        Bordes de las bandas, p. ej. [0, 0.25, 0.5, 1, 2, ...]; la banda b es
        el intervalo (a, b] entre bordes consecutivos.
    n_max : int, default 4000
        Si hay más puntos se toma una submuestra aleatoria de este tamaño
        (el número de pares crece como n²/2: 4000 puntos ~ 8 millones).
    seed : int, default C.SEED
        Semilla de la submuestra.

    Devuelve
    --------
    pandas.DataFrame
        Una fila por banda: desde_km, hasta_km, n_pares, moran_I,
        semivarianza, centro_km y varianza_total (meseta de referencia).

    Fórmulas (para los N(h) pares cuya distancia cae en la banda h)
    ----------------------------------------------------------------
    I(h) = [ (1/N(h)) sum z_i z_j ] / [ (1/n) sum z_i² ],  z = x - x̄
    γ(h) = (1 / (2 N(h))) · sum (x_i - x_j)²   (semivariograma empírico)

    Interpretación
    --------------
    I(h) decrece con la distancia; la distancia a la que I(h) ≈ 0 o a la que
    γ(h) alcanza la varianza total (la "meseta" o sill) es el ALCANCE: más
    allá, los puntos ya no comparten información. Un bloque de validación
    más grande que ese alcance (o un buffer de ese tamaño) reduce la fuga
    de información espacial entre train y test.
    """
    rng = np.random.default_rng(seed)
    xy = np.asarray(xy, float)
    x = np.asarray(x, float)
    if len(x) > n_max:
        sel = rng.choice(len(x), n_max, replace=False)
        xy, x = xy[sel], x[sel]
    d = pdist(xy)  # distancias condensadas de todos los pares i < j
    iu, ju = np.triu_indices(len(x), k=1)  # mismo orden de pares que pdist
    z = x - x.mean()
    var = (z ** 2).mean()  # varianza poblacional (denominador de I)
    filas = []
    for a, b in zip(bandas_km[:-1], bandas_km[1:]):  # bandas consecutivas (a, b]
        m = (d > a) & (d <= b)  # máscara de pares dentro de la banda
        npares = int(m.sum())
        if npares == 0:
            filas.append((a, b, 0, np.nan, np.nan))
            continue
        I = (z[iu[m]] * z[ju[m]]).mean() / var  # covarianza en la banda / varianza
        gamma = 0.5 * ((x[iu[m]] - x[ju[m]]) ** 2).mean()  # semivarianza
        filas.append((a, b, npares, I, gamma))
    out = pd.DataFrame(filas, columns=["desde_km", "hasta_km", "n_pares", "moran_I", "semivarianza"])
    out["centro_km"] = (out["desde_km"] + out["hasta_km"]) / 2  # eje x para graficar
    out["varianza_total"] = x.var()  # meseta teórica del semivariograma
    return out


# ---------------------------------------------------------------------------
# Patrón de puntos
# ---------------------------------------------------------------------------

def clark_evans(xy):
    """Índice del vecino más cercano R = d_obs / d_esp (Clark & Evans 1954).
    R < 1 agrupado, R ~ 1 aleatorio (CSR), R > 1 disperso/regular.
    Área = envolvente convexa (sin corrección de borde: se reporta).

    Parámetros
    ----------
    xy : array-like (n, 2)
        Coordenadas proyectadas en km (se eliminan duplicados exactos).

    Devuelve
    --------
    dict
        n_puntos_unicos, area_km2, d_obs_km, d_esp_km, R y z.

    Fórmulas
    --------
    λ = n / A (intensidad, puntos por km²);
    d_obs = media de la distancia de cada punto a su vecino más cercano;
    d_esp = 1 / (2 sqrt(λ))  (esperanza bajo aleatoriedad espacial completa);
    SE = 0.26136 / sqrt(n λ);   z = (d_obs - d_esp) / SE.
    z < -1.96 indica agrupamiento significativo al 5 %.

    Por qué se eliminan duplicados
    ------------------------------
    Varias unidades en la misma coordenada tendrían distancia 0 a su vecino
    y forzarían R hacia 0 (agrupamiento artificial).

    Limitación
    ----------
    Sin corrección de borde, los puntos cercanos al borde tienen su vecino
    real "fuera" y d_obs se sobrestima levemente; además la envolvente
    convexa incluye zonas sin predios (cuerpos de agua, áreas no urbanas),
    lo que infla A y hace que R parezca más agrupado.
    """
    xy = np.unique(np.asarray(xy, float), axis=0)  # puntos únicos
    n = len(xy)
    area = ConvexHull(xy).volume  # en 2D, 'volume' es el área
    d, _ = KDTree(xy).query(xy, k=2)  # k=2: el primero es el propio punto (d=0)
    d_obs = d[:, 1].mean()
    lam = n / area
    d_esp = 0.5 / np.sqrt(lam)
    se = 0.26136 / np.sqrt(n * lam)
    return {"n_puntos_unicos": n, "area_km2": area, "d_obs_km": d_obs, "d_esp_km": d_esp,
            "R": d_obs / d_esp, "z": (d_obs - d_esp) / se}


def _muestra_csr(hull_xy, n, rng):
    """Genera n puntos uniformes (CSR) dentro de un polígono convexo.

    Parámetros
    ----------
    hull_xy : numpy.ndarray (m, 2)
        Vértices de la envolvente convexa.
    n : int
        Número de puntos a generar.
    rng : numpy.random.Generator
        Generador aleatorio (se comparte para reproducibilidad).

    Devuelve
    --------
    numpy.ndarray (n, 2)
        Puntos con distribución uniforme en el polígono.

    Método
    ------
    Muestreo por rechazo: se generan puntos uniformes en el rectángulo que
    contiene al polígono y se conservan los que caen dentro. La pertenencia
    se decide con una triangulación de Delaunay de los vértices (que cubre
    exactamente la envolvente convexa): find_simplex devuelve -1 fuera.
    """
    tri = Delaunay(hull_xy)
    mn, mx = hull_xy.min(0), hull_xy.max(0)  # rectángulo envolvente
    pts = []
    while sum(len(p) for p in pts) < n:  # repetir hasta acumular n aceptados
        cand = rng.uniform(mn, mx, size=(n * 2, 2))
        pts.append(cand[tri.find_simplex(cand) >= 0])  # se aceptan solo los de dentro
    return np.vstack(pts)[:n]


def ripley_L(xy, radios_km, n_sim=19, n_max=3000, seed=C.SEED):
    """Función L de Ripley, L(r) = sqrt(K(r)/pi) - r, con envolvente de
    n_sim simulaciones de aleatoriedad espacial completa (CSR) dentro de la
    envolvente convexa. L por encima de la envolvente = agrupamiento a esa
    escala. Sin corrección de borde (subestima K cerca del borde).

    Parámetros
    ----------
    xy : array-like (n, 2)
        Coordenadas proyectadas en km (se eliminan duplicados).
    radios_km : numpy.ndarray
        Radios r en los que se evalúa L.
    n_sim : int, default 19
        Número de simulaciones CSR para la envolvente.
    n_max : int, default 3000
        Tamaño máximo de la submuestra (el conteo por radio es costoso).
    seed : int, default C.SEED
        Semilla.

    Devuelve
    --------
    pandas.DataFrame
        r_km, L_obs, L_csr_min, L_csr_max (envolvente puntual de las
        simulaciones).

    Fórmulas
    --------
    K(r) = A / (n (n-1)) · sum_i sum_{j != i} 1(d_ij <= r)
    Bajo CSR, K(r) = π r², por lo que L(r) = sqrt(K/π) - r ≈ 0.

    Interpretación
    --------------
    L(r) > 0 y por encima de la envolvente: agrupamiento a la escala r;
    L(r) < 0 y por debajo: regularidad/inhibición. Con 19 simulaciones, la
    envolvente mín-máx corresponde a una prueba puntual con alfa ≈ 2/20 =
    0.10 bilateral (0.05 por cada lado). A diferencia de Clark-Evans (un solo
    número), L describe el patrón a VARIAS escalas.
    """
    rng = np.random.default_rng(seed)
    xy = np.unique(np.asarray(xy, float), axis=0)
    if len(xy) > n_max:
        xy = xy[rng.choice(len(xy), n_max, replace=False)]
    hull = ConvexHull(xy)
    area = hull.volume  # área (2D) de la envolvente convexa
    hull_xy = xy[hull.vertices]  # vértices del polígono para simular CSR dentro

    def L_de(p):
        """Calcula L(r) - r para el conjunto de puntos p en todos los radios.

        Parámetros
        ----------
        p : numpy.ndarray (m, 2)
            Puntos (observados o simulados).

        Devuelve
        --------
        numpy.ndarray
            L(r) = sqrt(K(r)/π) - r para cada r de radios_km, usando el área
            de la envolvente observada (misma ventana para todos).
        """
        n = len(p)
        arbol = KDTree(p)
        # query_radius cuenta también al propio punto (d = 0): se resta 1 por punto
        K = np.array([(arbol.query_radius(p, r, count_only=True) - 1).sum() for r in radios_km])
        K = area * K / (n * (n - 1))  # normalización por intensidad: K = A·(pares)/(n(n-1))
        return np.sqrt(K / np.pi) - radios_km

    L_obs = L_de(xy)
    # cada simulación: mismo n de puntos, uniformes en la misma envolvente
    sims = np.array([L_de(_muestra_csr(hull_xy, len(xy), rng)) for _ in range(n_sim)])
    return pd.DataFrame({"r_km": radios_km, "L_obs": L_obs,
                         "L_csr_min": sims.min(0), "L_csr_max": sims.max(0)})


def dbscan_haversine(lat, lon, eps_km=0.3, min_samples=20):
    """Agrupamiento por densidad DBSCAN con distancia haversine.

    Parámetros
    ----------
    lat, lon : array-like
        Coordenadas en GRADOS.
    eps_km : float, default 0.3
        Radio de vecindad en km.
    min_samples : int, default 20
        Mínimo de puntos (incluido el propio) dentro de eps para que un punto
        sea "núcleo".

    Devuelve
    --------
    numpy.ndarray
        Etiqueta de clúster por punto; -1 = ruido (punto aislado).

    Por qué
    -------
    DBSCAN no exige fijar el número de clústeres, encuentra formas
    arbitrarias (barrios alargados, corredores) y marca como ruido los
    puntos dispersos. La métrica haversine de scikit-learn trabaja en
    RADIANES sobre la esfera unitaria, por eso tanto las coordenadas como
    eps se pasan a radianes: eps_rad = eps_km / R_tierra. El BallTree es
    el índice que soporta la métrica haversine.
    """
    from sklearn.cluster import DBSCAN
    X = np.radians(np.column_stack([lat, lon]))  # orden (lat, lon) que exige 'haversine'
    modelo = DBSCAN(eps=eps_km / C.RADIO_TIERRA_KM, min_samples=min_samples,
                    metric="haversine", algorithm="ball_tree").fit(X)
    return modelo.labels_


# ---------------------------------------------------------------------------
# Escala / MAUP
# ---------------------------------------------------------------------------

def agregar_en_grilla(df, tam_km, col=C.OBJETIVO, min_n=5):
    """Agrega puntos en celdas cuadradas de lado tam_km (media y conteo).

    Parámetros
    ----------
    df : pandas.DataFrame
        Debe contener "x_km", "y_km" y la columna col.
    tam_km : float
        Lado de la celda en km.
    col : str, default C.OBJETIVO
        Variable a promediar por celda.
    min_n : int, default 5
        Celdas con menos puntos se descartan.

    Devuelve
    --------
    pandas.DataFrame
        _cx, _cy (índices enteros de la celda), mean, size y x_km, y_km
        (centro geométrico de la celda).

    Por qué
    -------
    Permite estudiar el Problema de la Unidad de Área Modificable (MAUP):
    los resultados espaciales cambian según el tamaño de la unidad de
    agregación. Se descartan celdas con pocos puntos porque su media es muy
    ruidosa y distorsionaría el I de Moran.
    """
    cx = np.floor(df["x_km"] / tam_km).astype(int)  # índice de columna de la celda
    cy = np.floor(df["y_km"] / tam_km).astype(int)  # índice de fila de la celda
    g = df.assign(_cx=cx, _cy=cy).groupby(["_cx", "_cy"])[col].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= min_n]
    g["x_km"] = (g["_cx"] + 0.5) * tam_km  # centro de la celda (no el centroide de los puntos)
    g["y_km"] = (g["_cy"] + 0.5) * tam_km
    return g


def moran_por_escala(df, tamanos_km, col=C.OBJETIVO, k=8, n_perm=199):
    """I de Moran de la media por celda para varios tamaños de grilla (MAUP).

    Parámetros
    ----------
    df : pandas.DataFrame
        Puntos con "x_km", "y_km" y col.
    tamanos_km : iterable de float
        Tamaños de celda a evaluar.
    col : str, default C.OBJETIVO
        Variable a agregar.
    k : int, default 8
        Vecinos para la W kNN entre celdas (se reduce si hay pocas celdas).
    n_perm : int, default 199
        Permutaciones para el pseudo p-valor.

    Devuelve
    --------
    pandas.DataFrame
        celda_km, n_celdas, moran_I, p_perm y sd_media_celda (desviación de
        las medias por celda: cuánta variación sobrevive a la agregación).

    Interpretación
    --------------
    Al agregar en celdas más grandes la variabilidad local se promedia
    (sd_media_celda baja) y el I de Moran suele subir porque se conserva la
    tendencia de gran escala. Si la conclusión (autocorrelación fuerte) se
    mantiene en todas las escalas, es robusta al MAUP; también orienta el
    tamaño de celda de la línea base BaselineModaZona y de los bloques de
    validación.
    """
    filas = []
    for t in tamanos_km:
        g = agregar_en_grilla(df, t, col)
        if len(g) <= k + 2:
            continue  # muy pocas celdas para una W con k vecinos y un Moran estable
        W, _, _ = pesos_knn(g[["x_km", "y_km"]].to_numpy(), k=min(k, len(g) - 2))
        r = moran_global(g["mean"].to_numpy(), W, n_perm=n_perm)
        filas.append({"celda_km": t, "n_celdas": len(g), "moran_I": r["I"], "p_perm": r["p_perm"],
                      "sd_media_celda": g["mean"].std()})
    return pd.DataFrame(filas)
