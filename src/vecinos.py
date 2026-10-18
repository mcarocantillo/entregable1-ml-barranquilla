# -*- coding: utf-8 -*-
"""
Validación y modelos basados en vecinos más cercanos (versión 3 del proyecto).

Organización del módulo
-----------------------
- Distancias al vecino más cercano y su función de distribución acumulada
  empírica (ECDF), ponderada por viviendas.
- kNNDM: validación cruzada de k folds que empareja distancias al vecino más
  cercano (Milà et al. 2022; Linnenbrink et al. 2024). Reemplaza el buffer
  fijo por folds cuya geometría se parece a la del escenario de predicción.
- Logística ordinal (odds proporcionales) con penalización L2, compatible con
  Pipeline y GridSearchCV de scikit-learn.
- Sesgos de decisión por clase que maximizan el F1 macro (se aprenden con
  predicciones fuera de muestra).
- Lag espacial con un grafo KNN: elegir k por el I de Moran y promediar a los
  k vecinos (la matriz A normalizada por filas del libro de Modelos
  Espacio-Temporales del profesor Rubio, sección 2.5.1).

Convención: x_km, y_km son coordenadas proyectadas en km (distancia
euclidiana válida a escala urbana; frente a haversine el error es < 0.1 %).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import expit
from scipy.stats import wasserstein_distance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score

from . import config as C
from .particion import folds_estratificados_por_grupo


# ---------------------------------------------------------------------------
# Distancias al vecino más cercano
# ---------------------------------------------------------------------------

def puntos_unicos(xy, grupos):
    """Un punto por grupo (edificio): coordenada del grupo, número de viviendas
    y código de grupo de cada fila.

    Las unidades de un mismo edificio comparten coordenada; trabajar con un
    punto por edificio evita que la distancia "al vecino más cercano" sea 0
    solo porque el vecino es otra unidad del mismo edificio.
    """
    g_unicos, cod = np.unique(np.asarray(grupos), return_inverse=True)
    xy = np.asarray(xy, float)
    n = np.bincount(cod)
    xy_g = np.c_[np.bincount(cod, xy[:, 0]) / n, np.bincount(cod, xy[:, 1]) / n]
    return xy_g, n, cod


def distancia_vecino(xy_desde, xy_hacia):
    """Distancia (km) de cada punto de xy_desde al punto más cercano de xy_hacia."""
    return cKDTree(np.asarray(xy_hacia, float)).query(np.asarray(xy_desde, float))[0]


def ecdf(d, pesos=None, rejilla=None):
    """ECDF ponderada evaluada en una rejilla de distancias (km)."""
    d = np.asarray(d, float)
    w = np.ones_like(d) if pesos is None else np.asarray(pesos, float)
    if rejilla is None:
        rejilla = np.linspace(0, np.quantile(d, 0.995), 200)
    o = np.argsort(d)
    acum = np.cumsum(w[o]) / w.sum()
    pos = np.searchsorted(d[o], rejilla, side="right")
    return rejilla, np.where(pos > 0, acum[np.clip(pos - 1, 0, None)], 0.0)


def cuantiles_ponderados(d, pesos, qs=(0.1, 0.5, 0.9)):
    """Cuantiles de una distancia ponderada por viviendas."""
    d, w = np.asarray(d, float), np.asarray(pesos, float)
    o = np.argsort(d)
    acum = np.cumsum(w[o]) / w.sum()
    return [float(d[o][min(np.searchsorted(acum, q), len(d) - 1)]) for q in qs]


# ---------------------------------------------------------------------------
# kNNDM: validación cruzada por emparejamiento de distancias al vecino
# ---------------------------------------------------------------------------

def distancias_fold(xy_g, fold_g):
    """Para cada punto, distancia al punto más cercano de OTRO fold (G*_j):
    lo que "ve" el modelo de ese fold como vecino conocido más cercano."""
    d = np.empty(len(xy_g))
    for f in np.unique(fold_g):
        dentro = fold_g == f
        d[dentro] = distancia_vecino(xy_g[dentro], xy_g[~dentro])
    return d


def knndm(xy_train, grupos_train, y_train, xy_pred=None, pesos_pred=None, k=C.N_FOLDS, q_rejilla=None,
          seed=C.SEED, verbose=True, d_pred=None):
    """k-fold Nearest Neighbour Distance Matching (kNNDM).

    Parámetros
    ----------
    xy_train : array (n, 2)
        Coordenadas (km) de las viviendas de entrenamiento.
    grupos_train : array (n,)
        Edificio de cada vivienda (las unidades de un edificio nunca se separan).
    y_train : array (n,)
        Estrato, para que los folds tengan proporciones de clase parecidas.
    xy_pred : array (m, 2)
        Dónde se va a predecir (define el escenario). Solo se usan sus
        coordenadas, nunca su estrato.
    pesos_pred : array (m,) o None
        Peso de cada punto de predicción (p. ej. viviendas por edificio).
    d_pred : array (m,) o None
        Distancias del escenario ya calculadas (en lugar de xy_pred). Sirve
        cuando el escenario no es "puntos nuevos", p. ej. ocultar manzanas
        completas dentro de train (distancia a la manzana conocida más cercana).
    k : int
        Número de folds.
    q_rejilla : lista de int o None
        Números de grupos de K-Means a probar.

    Devuelve
    --------
    dict con
        folds : lista de (idx_train, idx_val) posicionales sobre las n viviendas.
        fold_fila, grupo_fila : fold y grupo de K-Means de cada vivienda.
        q : número de grupos elegido; W : su distancia de Wasserstein.
        tabla : DataFrame q -> W.
        d_pred : distancias predicción -> entrenamiento (G_ij) y sus pesos.
        d_cv : distancias validación -> entrenamiento de los folds elegidos (G*_j).
        d_loo : distancias de la validación aleatoria por edificios (G_j).

    Cómo funciona (Linnenbrink et al. 2024)
    ---------------------------------------
    1. G_ij: distancia de cada punto de predicción al edificio de entrenamiento
       más cercano. Es la distancia a la que el modelo tendrá que "extrapolar".
    2. Para cada q se agrupan los edificios de entrenamiento con K-Means en q
       grupos compactos y los grupos se reparten en k folds (estratificados por
       estrato, con el mismo algoritmo voraz de la partición del cap. 2).
       q pequeño = folds grandes y lejanos (como bloques); q grande = folds
       salpicados (como una validación aleatoria por edificios).
    3. G*_j: distancia de cada edificio al edificio más cercano de otro fold.
    4. Se elige el q cuya distribución de G*_j está más cerca de la de G_ij
       (distancia de Wasserstein-1 = área entre las dos ECDF, en km).

    Así el tamaño de la "zona prohibida" no se fija a mano (el buffer de 1 km):
    sale de comparar la geometría de la validación con la del uso real.
    """
    xy_g, n_g, cod = puntos_unicos(xy_train, grupos_train)
    y_train = np.asarray(y_train)
    # estrato del edificio (casi constante dentro de él: ICC 0.992), para estratificar
    y_g = pd.Series(y_train).groupby(cod).agg(lambda s: s.mode().iloc[0]).to_numpy()
    if d_pred is None:
        d_pred = distancia_vecino(xy_pred, xy_g)
    d_pred = np.asarray(d_pred, float)
    pesos_pred = np.ones(len(d_pred)) if pesos_pred is None else np.asarray(pesos_pred, float)
    # G_j: validación aleatoria por edificios (cada edificio frente a todos los demás)
    d_loo = cKDTree(xy_g).query(xy_g, k=2)[0][:, 1]
    if q_rejilla is None:
        q_rejilla = [k, 8, 12, 16, 20, 25, 30, 35, 40, 50, 65, 80, 120, 200, 300, 500, 800, 1200, 2000, 3500, 6000]
    filas, mejor = [], None
    for q in q_rejilla:
        km = MiniBatchKMeans(n_clusters=q, random_state=seed, batch_size=4096, n_init=3)
        grupo_q = km.fit_predict(xy_g, sample_weight=n_g)
        # los q grupos se reparten en k folds estratificados (un grupo nunca se divide);
        # se usa una fila por edificio repetida según sus viviendas para respetar los pesos
        filas_rep = np.repeat(np.arange(len(xy_g)), n_g)
        folds_rep = folds_estratificados_por_grupo(y_g[filas_rep], grupo_q[filas_rep], k, seed)
        fold_g = np.empty(len(xy_g), int)
        for f, (_, iva) in enumerate(folds_rep):
            fold_g[filas_rep[iva]] = f
        d_cv = distancias_fold(xy_g, fold_g)
        W = wasserstein_distance(d_cv, d_pred, u_weights=n_g, v_weights=pesos_pred)
        filas.append({"q": q, "W_km": W, "mediana_cv_km": cuantiles_ponderados(d_cv, n_g, [0.5])[0]})
        if verbose:
            print(f"  q = {q:>5}: W = {W:.3f} km | mediana de G*_j = {filas[-1]['mediana_cv_km']:.3f} km", flush=True)
        if mejor is None or W < mejor["W"]:
            mejor = {"q": q, "W": W, "fold_g": fold_g, "d_cv": d_cv, "grupo_g": grupo_q}
    fold_fila = mejor["fold_g"][cod]
    idx = np.arange(len(cod))
    return {"folds": [(idx[fold_fila != f], idx[fold_fila == f]) for f in range(k)],
            "fold_fila": fold_fila, "grupo_fila": mejor["grupo_g"][cod], "q": mejor["q"], "W": mejor["W"], "tabla": pd.DataFrame(filas),
            "d_pred": d_pred, "pesos_pred": pesos_pred, "d_cv": mejor["d_cv"], "d_loo": d_loo, "pesos_g": n_g,
            "W_loo": wasserstein_distance(d_loo, d_pred, u_weights=n_g, v_weights=pesos_pred)}


def distancias_de_folds(xy_train, grupos_train, folds):
    """G*_j de unos folds dados (p. ej. los de bloques + buffer de la versión 2):
    distancia de cada vivienda de validación a la vivienda de entrenamiento más
    cercana de su fold. Devuelve (distancias por edificio, viviendas por edificio)."""
    xy_train = np.asarray(xy_train, float)
    d = np.full(len(xy_train), np.nan)
    for itr, iva in folds:
        d[iva] = distancia_vecino(xy_train[iva], np.unique(xy_train[itr], axis=0))
    _, n_g, cod = puntos_unicos(xy_train, grupos_train)
    d_g = pd.Series(d).groupby(cod).first().to_numpy()
    return d_g, n_g


# ---------------------------------------------------------------------------
# Logística ordinal (odds proporcionales)
# ---------------------------------------------------------------------------

class LogisticaOrdinal(ClassifierMixin, BaseEstimator):
    """Regresión logística ordinal de odds proporcionales con L2.

    P(y <= j | x) = sigmoide(theta_j - x·beta),   j = 1..K-1,
    con umbrales theta_1 < ... < theta_{K-1} y UN solo vector beta para todas
    las clases (la multinomial usa K vectores). Respeta el orden 1 < ... < 6:
    subir x·beta mueve la probabilidad hacia estratos más altos.

    Parámetros
    ----------
    C : float
        Inverso de la regularización, con la misma convención de
        LogisticRegression: se minimiza sum(pérdida) + ||beta||² / (2 C).
    class_weight : None o "balanced"
        "balanced" pondera cada clase por n / (K · n_clase).
    max_iter : int
        Iteraciones máximas de L-BFGS.

    Los umbrales se parametrizan como theta_1 = a_1, theta_j = theta_{j-1} +
    exp(a_j), lo que garantiza que queden ordenados sin restricciones.
    """

    def __init__(self, C=1.0, class_weight=None, max_iter=500):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter

    def _umbrales(self, a):
        return np.cumsum(np.r_[a[0], np.exp(a[1:])])

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_, yc = np.unique(np.asarray(y), return_inverse=True)
        K, (n, p) = len(self.classes_), X.shape
        if self.class_weight == "balanced":
            w = (n / (K * np.bincount(yc, minlength=K)))[yc]
        else:
            w = np.ones(n)
        # arranque: umbrales en los logits de las proporciones acumuladas
        acum = np.clip(np.cumsum(np.bincount(yc, minlength=K))[:-1] / n, 1e-4, 1 - 1e-4)
        th0 = np.log(acum / (1 - acum))
        a0 = np.r_[th0[0], np.log(np.maximum(np.diff(th0), 1e-3))]
        x0 = np.r_[np.zeros(p), a0]
        lam = 1.0 / self.C

        def objetivo(par):
            beta, a = par[:p], par[p:]
            th = self._umbrales(a)
            eta = X @ beta
            th_ext = np.r_[-np.inf, th, np.inf]
            u, l = th_ext[yc + 1] - eta, th_ext[yc] - eta
            Fu, Fl = expit(u), expit(l)
            prob = np.clip(Fu - Fl, 1e-15, None)
            fu, fl = Fu * (1 - Fu), Fl * (1 - Fl)
            valor = -np.sum(w * np.log(prob)) + 0.5 * lam * beta @ beta
            # derivadas de -log p respecto a eta y a cada umbral
            g_eta = w * (fu - fl) / prob
            g_beta = X.T @ g_eta + lam * beta
            g_th = np.zeros(K - 1)
            sup = yc < K - 1
            np.add.at(g_th, yc[sup], -(w * fu / prob)[sup])
            inf = yc > 0
            np.add.at(g_th, yc[inf] - 1, (w * fl / prob)[inf])
            # regla de la cadena: theta_j depende de a_1 y de exp(a_m) para m <= j
            g_a = np.empty(K - 1)
            g_a[0] = g_th.sum()
            g_a[1:] = np.exp(a[1:]) * np.cumsum(g_th[::-1])[::-1][1:]
            return valor, np.r_[g_beta, g_a]

        res = minimize(objetivo, x0, jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.coef_ = res.x[:p]
        self.umbrales_ = self._umbrales(res.x[p:])
        self.n_iter_ = res.nit
        self.convergio_ = bool(res.success)
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.coef_

    def predict_proba(self, X):
        eta = self.decision_function(X)
        F = expit(self.umbrales_[None, :] - eta[:, None])
        F = np.c_[np.zeros(len(eta)), F, np.ones(len(eta))]
        return np.clip(np.diff(F, axis=1), 1e-12, None)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Sesgos de decisión para el F1 macro
# ---------------------------------------------------------------------------

def ajustar_sesgos_f1(y, proba, clases=C.CLASES, rejilla=np.linspace(-1.5, 1.5, 31), rondas=3):
    """Sesgos aditivos b_k sobre log p_k que maximizan el F1 macro.

    La predicción por defecto es argmax_k p_k; con los sesgos pasa a ser
    argmax_k (log p_k + b_k). Mover b_k equivale a mover el umbral a partir
    del cual se elige el estrato k: sube su recall a costa de su precisión.
    Se ajusta por ascenso coordenado (un estrato a la vez, varias rondas) con
    b del primer estrato fijo en 0 (solo importan las diferencias).

    Debe aplicarse a predicciones FUERA DE MUESTRA (validación cruzada); si se
    ajusta y se evalúa en las mismas filas, el resultado es optimista.
    """
    y, lp = np.asarray(y), np.log(np.clip(proba, 1e-12, None))
    clases = np.asarray(clases)
    b = np.zeros(len(clases))

    def f1(bb):
        return f1_score(y, clases[np.argmax(lp + bb, 1)], average="macro", zero_division=0)

    mejor = f1(b)
    for _ in range(rondas):
        for j in range(1, len(clases)):
            for v in rejilla:
                bb = b.copy()
                bb[j] = v
                s = f1(bb)
                if s > mejor + 1e-9:
                    mejor, b = s, bb
    return b, mejor


def aplicar_sesgos(proba, b, clases=C.CLASES):
    """Estrato predicho con los sesgos de decisión."""
    return np.asarray(clases)[np.argmax(np.log(np.clip(proba, 1e-12, None)) + b, 1)]


# ---------------------------------------------------------------------------
# Lag espacial con grafo KNN
# ---------------------------------------------------------------------------

def moran_por_k(xy_g, valor_g, ks, n_perm=0):
    """I de Moran del valor por edificio para varios k (grafo KNN normalizado
    por filas). Criterio del libro del profesor (sec. 2.2.4.2): elegir el k que
    maximiza el I de Moran, es decir, el grafo que mejor refleja la
    autocorrelación real."""
    z = np.asarray(valor_g, float) - np.mean(valor_g)
    arbol = cKDTree(np.asarray(xy_g, float))
    _, idx = arbol.query(xy_g, k=max(ks) + 1)
    filas = []
    for k in ks:
        lag = z[idx[:, 1:k + 1]].mean(1)  # (A_norm z)_i: promedio de los k vecinos
        filas.append({"k": k, "moran_I": float(z @ lag / (z @ z))})
    return pd.DataFrame(filas)


def lag_knn(xy_destino, xy_fuente, valores_fuente, k, grupo_destino=None, grupo_fuente=None):
    """Promedio de los k vecinos más cercanos de la fuente para cada destino.

    valores_fuente puede ser (m,) o (m, f). Si se dan grupos, un destino nunca
    usa a su propio grupo (su edificio) como vecino: con ICC = 0.992 sería
    copiarse a sí mismo. Devuelve también la distancia media a esos vecinos.
    """
    xy_fuente = np.asarray(xy_fuente, float)
    V = np.asarray(valores_fuente, float)
    V = V[:, None] if V.ndim == 1 else V
    extra = 1 if grupo_fuente is not None else 0
    dist, idx = cKDTree(xy_fuente).query(np.asarray(xy_destino, float), k=k + extra)
    dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
    if extra:
        gf, gd = np.asarray(grupo_fuente), np.asarray(grupo_destino)
        propio = gf[idx] == gd[:, None]
        # se descarta el propio grupo si aparece; si no, el vecino más lejano sobrante
        quitar = np.where(propio.any(1), propio.argmax(1), k)
        keep = np.ones_like(idx, bool)
        keep[np.arange(len(idx)), quitar] = False
        idx = idx[keep].reshape(len(idx), k)
        dist = dist[keep].reshape(len(idx), k)
    lag = V[idx].mean(1)
    return (lag[:, 0] if lag.shape[1] == 1 else lag), dist.mean(1)


def lag_fuera_de_grupo(xy_destino, grupo_destino, xy_fuente, valores_fuente, grupo_fuente, k):
    """Promedio de los k vecinos más cercanos de la fuente que NO están en el
    mismo grupo que el destino (grupo = grupo de K-Means de kNNDM).

    Sirve para el estrato de los vecinos en las filas de ENTRENAMIENTO: si se
    calculara con sus vecinos inmediatos (a metros), el modelo aprendería
    "estrato ≈ estrato del vecino de al lado", pero al predecir una zona nueva
    el vecino conocido más cercano está a cientos de metros. Excluir el propio
    grupo hace que la variable tenga en entrenamiento la misma distribución de
    distancias que al predecir (el mismo principio anti-fuga del lag espacial
    "hasta t-1" del libro de Modelos Espacio-Temporales, sec. 2.5.2).
    """
    xy_destino = np.asarray(xy_destino, float)
    xy_fuente = np.asarray(xy_fuente, float)
    v = np.asarray(valores_fuente, float)
    gd, gf = np.asarray(grupo_destino), np.asarray(grupo_fuente)
    lag = np.empty(len(xy_destino))
    dist = np.empty(len(xy_destino))
    grupos = np.unique(gd)
    if len(grupos) <= 200:
        # pocos grupos grandes (p. ej. q = 30): un árbol por grupo con la fuente de los demás grupos
        for g in grupos:
            dd = gd == g
            ff = gf != g
            dk, ik = cKDTree(xy_fuente[ff]).query(xy_destino[dd], k=k)
            dk, ik = np.atleast_2d(dk), np.atleast_2d(ik)
            lag[dd] = v[ff][ik].mean(1)
            dist[dd] = dk.mean(1)
        return lag, dist
    # muchos grupos pequeños (p. ej. q = 6000): un solo árbol; se piden vecinos de sobra y se descartan los
    # del grupo propio, ampliando la búsqueda solo para los puntos que no alcanzan k vecinos válidos
    arbol = cKDTree(xy_fuente)
    pend = np.arange(len(xy_destino))
    m = k + 32
    while len(pend):
        mm = min(m, len(xy_fuente))
        dk, ik = arbol.query(xy_destino[pend], k=mm)
        dk, ik = dk.reshape(len(pend), mm), ik.reshape(len(pend), mm)
        valido = gf[ik] != gd[pend][:, None]
        listo = (valido.sum(1) >= k) | (mm == len(xy_fuente))
        # primeros k vecinos válidos de cada fila, en orden de distancia
        orden = np.argsort(~valido[listo], axis=1, kind="stable")[:, :k]
        filas = np.arange(listo.sum())[:, None]
        lag[pend[listo]] = v[ik[listo][filas, orden]].mean(1)
        dist[pend[listo]] = dk[listo][filas, orden].mean(1)
        pend = pend[~listo]
        m *= 4
    return lag, dist
