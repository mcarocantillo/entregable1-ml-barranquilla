# -*- coding: utf-8 -*-
"""
Piezas del modelo base: preprocesamiento en Pipeline (ajustado solo con
train), línea base espacial por zona, métricas multiclase/ordinales e
intervalos de confianza bootstrap POR BLOQUES espaciales.

Contenido
---------
- Winsorizador: transformador compatible con scikit-learn que recorta
  extremos con cuantiles aprendidos en train.
- construir_preprocesador: ColumnTransformer con ramas numérica sesgada
  (log1p), numérica lineal y categórica (one-hot).
- BaselineModaZona: clasificador de referencia "estrato más frecuente de la
  zona", la vara mínima que el modelo debe superar.
- metricas: métricas de clasificación, ordinales y probabilísticas.
- bootstrap_por_bloques / bootstrap_diferencia: incertidumbre de las métricas
  respetando la dependencia espacial.

Principio transversal: todo parámetro que se "aprende" de los datos
(cuantiles, medianas, medias, desviaciones, categorías, frecuencias por zona)
se estima SOLO con el fold de entrenamiento. Si se estimara con todos los
datos, información del test se filtraría al modelo (data leakage) y las
métricas serían optimistas.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as C


class Winsorizador(TransformerMixin, BaseEstimator):
    """Recorta cada columna a [q_inf, q_sup] APRENDIDOS EN TRAIN (fit). Limita
    la influencia de valores extremos pero plausibles (p. ej. registros NPH de
    edificios completos) sobre la regresión logística sin borrar filas.

    Parámetros
    ----------
    q_inf : float, default 0.001
        Cuantil inferior de recorte (0.1 %).
    q_sup : float, default 0.999
        Cuantil superior de recorte (99.9 %).

    Atributos aprendidos (tras fit)
    -------------------------------
    lim_inf_, lim_sup_ : numpy.ndarray
        Límites por columna, calculados ignorando NaN.

    Fórmula
    -------
    x' = min( max(x, L_j), U_j ),  con L_j = Q_{q_inf}(x_j), U_j = Q_{q_sup}(x_j)
    estimados en train.

    Por qué
    -------
    - Winsorizar (recortar) en vez de eliminar conserva la fila y el orden de
      los valores: el predio sigue siendo "el más grande", pero ya no domina
      la media, la desviación del StandardScaler ni los coeficientes.
    - Los límites se aprenden en fit (solo train dentro del Pipeline); en
      transform se aplican los MISMOS límites al test, sin mirarlo.
    - Los NaN pasan intactos (np.clip conserva NaN), así el imputador
      posterior los sigue viendo como faltantes.
    - Hereda de BaseEstimator para que get_params/set_params y clone funcionen
      (necesario en validación cruzada y GridSearchCV).
    """

    def __init__(self, q_inf=0.001, q_sup=0.999):
        """Guarda los hiperparámetros sin validarlos ni transformarlos.

        Parámetros
        ----------
        q_inf, q_sup : float
            Cuantiles de recorte en [0, 1].

        Por qué
        -------
        Convención de scikit-learn: __init__ solo asigna los argumentos con el
        mismo nombre, para que clone() pueda reconstruir el estimador.
        """
        self.q_inf = q_inf
        self.q_sup = q_sup

    @staticmethod
    def _a_float(X):
        """Convierte X (DataFrame o arreglo) a una matriz float con NaN.

        Parámetros
        ----------
        X : array-like o pandas.DataFrame

        Devuelve
        --------
        numpy.ndarray de float
            Lo no convertible a número y los pd.NA quedan como np.nan
            (np.nanquantile y np.clip no aceptan pd.NA).
        """
        return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, na_value=np.nan)

    def fit(self, X, y=None):
        """Aprende los cuantiles de recorte de cada columna.

        Parámetros
        ----------
        X : array-like (n, p)
            Datos de ENTRENAMIENTO.
        y : ignorado
            Presente por compatibilidad con la API de scikit-learn.

        Devuelve
        --------
        self
        """
        X = self._a_float(X)
        self.lim_inf_ = np.nanquantile(X, self.q_inf, axis=0)  # un límite por columna
        self.lim_sup_ = np.nanquantile(X, self.q_sup, axis=0)
        return self

    def transform(self, X):
        """Recorta X a los límites aprendidos en fit.

        Parámetros
        ----------
        X : array-like (n, p)
            Datos a transformar (train o test).

        Devuelve
        --------
        numpy.ndarray (n, p)
            Valores recortados; los NaN se conservan.
        """
        return np.clip(self._a_float(X), self.lim_inf_, self.lim_sup_)

    def get_feature_names_out(self, input_features=None):
        """Nombres de salida = nombres de entrada (transformación uno a uno).

        Parámetros
        ----------
        input_features : array-like de str
            Nombres de las columnas de entrada (los pasa el ColumnTransformer).

        Devuelve
        --------
        numpy.ndarray de object
            Los mismos nombres; permite recuperar el nombre de cada
            coeficiente al final del Pipeline.
        """
        return np.asarray(input_features, dtype=object)


def _categoricas_a_objeto(X):
    """Convierte a object con np.nan (pd.NA rompe SimpleImputer).

    Parámetros
    ----------
    X : array-like o pandas.DataFrame
        Columnas categóricas (pueden venir con dtype "string" o "category").

    Devuelve
    --------
    pandas.DataFrame de dtype object
        Mismos valores, con todo faltante (pd.NA, None, NaN) unificado como
        np.nan, que es lo que SimpleImputer reconoce como missing_values.
    """
    X = pd.DataFrame(X).astype(object)
    return X.where(X.notna(), np.nan)  # reemplaza cada faltante por np.nan


def construir_preprocesador(num_log, num_lineal, categoricas, q_inf=0.001, q_sup=0.999,
                            min_frecuencia=0.01):
    """ColumnTransformer:
    - numéricas sesgadas: winsorizar -> imputar mediana (+ indicador de
      faltante) -> log1p -> estandarizar
    - numéricas no sesgadas / espaciales: winsorizar -> imputar mediana (+
      indicador) -> estandarizar
    - categóricas: imputar 'faltante' -> one-hot (categorías con <1 % agrupadas
      en 'infrequent', las no vistas en train se ignoran)
    Todo se ajusta DENTRO del Pipeline, o sea solo con los datos de
    entrenamiento de cada fold.

    Parámetros
    ----------
    num_log : list[str]
        Numéricas con asimetría fuerte y no negativas (áreas, avalúos...).
    num_lineal : list[str]
        Numéricas aproximadamente simétricas o coordenadas (x_km, y_km...).
    categoricas : list[str]
        Variables categóricas (destino, tipo de predio, etc.).
    q_inf, q_sup : float
        Cuantiles de winsorización (ver Winsorizador).
    min_frecuencia : float, default 0.01
        Proporción mínima en train para que una categoría tenga su propia
        columna one-hot; las menos frecuentes se agrupan en 'infrequent'.

    Devuelve
    --------
    sklearn.compose.ColumnTransformer
        Sin ajustar; se inserta como primer paso de un Pipeline con el
        clasificador. Las columnas no listadas se descartan (remainder="drop")
        y los nombres de salida llevan el prefijo de la rama
        (p. ej. "num_log__area_construida").

    Por qué cada paso
    -----------------
    - Mediana (no media) para imputar: robusta a la asimetría.
    - add_indicator=True: agrega una columna 0/1 "estaba faltante"; el hecho
      de que falte un dato catastral puede ser informativo del estrato.
    - log1p = ln(1 + x): comprime la cola derecha y admite ceros (supone
      x > -1; las variables de esta rama son no negativas).
    - StandardScaler (z = (x - media) / desv.): la regresión logística con
      regularización penaliza igual todos los coeficientes, lo que solo es
      justo si las variables están en la misma escala.
    - One-hot con handle_unknown="infrequent_if_exist": una categoría que
      aparece en test pero no en train se codifica como 'infrequent' (o como
      todo ceros si no existe esa columna), en vez de lanzar un error.
    - Todo dentro del Pipeline: en cada fold de la validación espacial las
      medianas, cuantiles, medias/desviaciones y categorías se estiman solo
      con train -> sin fuga de información hacia el fold de prueba.
    """
    # el winsorizado va ANTES de imputar (es robusto a NaN) para no recortar
    # las columnas indicadoras 0/1 que agrega add_indicator=True
    log_pipe = Pipeline([
        ("winsor", Winsorizador(q_inf, q_sup)),
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        # log1p también se aplica a los indicadores 0/1 (quedan 0 y ln 2): sigue
        # siendo binaria y el escalado posterior la deja en la misma escala
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("escalar", StandardScaler()),
    ])
    lin_pipe = Pipeline([
        ("winsor", Winsorizador(q_inf, q_sup)),
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("escalar", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("a_objeto", FunctionTransformer(_categoricas_a_objeto, feature_names_out="one-to-one")),
        # el faltante se trata como una categoría más ("faltante"), no se descarta
        ("imputar", SimpleImputer(strategy="constant", fill_value="faltante")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=min_frecuencia, sparse_output=False)),
    ])
    # solo se agregan las ramas que tienen columnas (una rama vacía daría error)
    transformadores = []
    if num_log:
        transformadores.append(("num_log", log_pipe, list(num_log)))
    if num_lineal:
        transformadores.append(("num_lin", lin_pipe, list(num_lineal)))
    if categoricas:
        transformadores.append(("cat", cat_pipe, list(categoricas)))
    return ColumnTransformer(transformadores, remainder="drop", verbose_feature_names_out=True)


class BaselineModaZona(ClassifierMixin, BaseEstimator):
    """Línea base espacial para clasificación: la clase más frecuente (y las
    frecuencias de clase) de la ZONA gruesa (celda de tam_km) aprendidas en
    train. Si la zona no existe en train, usa la distribución global.
    Es el análogo, para clases, de la 'media por zona' que piden las instrucciones del proyecto.

    Parámetros
    ----------
    tam_km : float, default 6.0
        Lado de la celda cuadrada que define la zona.
    col_x, col_y : str
        Nombres de las columnas de coordenadas proyectadas (km) en X.

    Atributos aprendidos (tras fit)
    -------------------------------
    classes_ : numpy.ndarray
        Clases vistas en train, ordenadas.
    proba_zona_ : pandas.DataFrame
        P(clase | zona) = conteo(zona, clase) / conteo(zona); filas = zonas.
    proba_global_ : pandas.Series
        P(clase) en todo train (respaldo para zonas no vistas).

    Por qué
    -------
    Un modelo complejo solo aporta si supera a una regla trivial que explota
    la autocorrelación espacial ("en esta zona casi todo es estrato 2").
    Si el modelo no le gana a esta línea base, las variables catastrales no
    están añadiendo información más allá de la ubicación gruesa. Con la
    validación por bloques, muchas zonas del test no existen en train y se
    recurre a la distribución global, que es justamente el escenario de
    generalizar a zonas nuevas.
    """

    def __init__(self, tam_km=6.0, col_x="x_km", col_y="y_km"):
        """Guarda los hiperparámetros (convención de scikit-learn para clone).

        Parámetros
        ----------
        tam_km : float
            Tamaño de celda en km.
        col_x, col_y : str
            Columnas de coordenadas.
        """
        self.tam_km = tam_km
        self.col_x = col_x
        self.col_y = col_y

    def _zona(self, X):
        """Asigna a cada fila la clave de su celda de grilla, "cx_cy".

        Parámetros
        ----------
        X : pandas.DataFrame
            Debe tener las columnas col_x y col_y.

        Devuelve
        --------
        pandas.Series de str
            Clave de zona con cx = floor(x / tam_km), cy = floor(y / tam_km).
            El tipo Int64 admite nulos: una coordenada faltante produce la
            clave "<NA>_<NA>", que no coincidirá con ninguna zona de train.
        """
        cx = np.floor(np.asarray(X[self.col_x], float) / self.tam_km)
        cy = np.floor(np.asarray(X[self.col_y], float) / self.tam_km)
        # Int64 evita claves "3.0" y tolera NaN; la clave es texto para usarla como índice
        return pd.Series(cx).astype("Int64").astype(str) + "_" + pd.Series(cy).astype("Int64").astype(str)

    def fit(self, X, y):
        """Estima P(clase | zona) y P(clase) con los datos de entrenamiento.

        Parámetros
        ----------
        X : pandas.DataFrame
            Datos de train con las coordenadas.
        y : array-like
            Estrato de cada fila.

        Devuelve
        --------
        self
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        z = self._zona(X).to_numpy()
        # tabla zona x clase con TODAS las clases como columnas (0 si no aparece)
        tabla = pd.crosstab(z, y).reindex(columns=self.classes_, fill_value=0)
        self.proba_zona_ = tabla.div(tabla.sum(1), axis=0)  # normaliza cada fila a 1
        self.proba_global_ = pd.Series(y).value_counts(normalize=True).reindex(self.classes_, fill_value=0)
        return self

    def predict_proba(self, X):
        """Probabilidades de clase = frecuencias de la zona de cada fila.

        Parámetros
        ----------
        X : pandas.DataFrame
            Datos con las coordenadas.

        Devuelve
        --------
        numpy.ndarray (n, n_clases)
            Columnas en el orden de classes_. Filas de zonas no vistas en
            train reciben la distribución global.
        """
        z = self._zona(X)
        P = self.proba_zona_.reindex(z.to_numpy())  # zonas no vistas -> fila de NaN
        # fillna con una Series rellena cada COLUMNA (clase) con su P(clase) global
        P = P.fillna(self.proba_global_).to_numpy()
        return P

    def predict(self, X):
        """Predice la clase modal de la zona (argmax de predict_proba).

        Parámetros
        ----------
        X : pandas.DataFrame

        Devuelve
        --------
        numpy.ndarray
            Estrato predicho; en caso de empate gana la clase menor
            (argmax devuelve el primer máximo).
        """
        return self.classes_[self.predict_proba(X).argmax(1)]


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def metricas(y, pred, proba=None, clases=C.CLASES):
    """Calcula el tablero de métricas multiclase, ordinales y probabilísticas.

    Parámetros
    ----------
    y : array-like
        Estrato real (enteros 1-6).
    pred : array-like
        Estrato predicho.
    proba : numpy.ndarray (n, len(clases)) o None
        Probabilidades predichas; las columnas deben seguir el orden de
        `clases`. Si es None solo se calculan las métricas de etiquetas.
    clases : list, default C.CLASES ([1, ..., 6])
        Conjunto completo de clases.

    Devuelve
    --------
    dict
        accuracy, balanced_accuracy, precision/recall/F1 macro, F1
        ponderado, MAE ordinal, accuracy ±1, kappa cuadrático y, si hay
        proba: AUC OvR macro, log-loss y Brier multiclase.

    Fórmulas e interpretación
    -------------------------
    - balanced_accuracy = promedio del recall por clase: no se deja engañar
      por el desbalance (predecir siempre la clase mayoritaria da 1/K).
    - macro = promedio simple entre clases (cada estrato pesa igual);
      weighted = promedio ponderado por el soporte de cada clase.
    - MAE_ordinal = (1/n) sum |y_i - ŷ_i|: en "estratos" de distancia;
      confundir 2 con 3 cuesta 1, confundir 1 con 6 cuesta 5.
    - accuracy_±1 = proporción con |y - ŷ| <= 1.
    - Kappa ponderado cuadrático: kappa_w = 1 - sum w_ij O_ij / sum w_ij E_ij,
      w_ij = (i - j)² / (K - 1)²; acuerdo corregido por azar que penaliza
      cuadráticamente los errores lejanos (1 = perfecto, 0 = azar).
    - AUC OvR macro: promedio del AUC de cada clase contra el resto.
    - log_loss = -(1/n) sum_i ln p_i,y_i  (penaliza la confianza errónea).
    - Brier multiclase = (1/n) sum_i sum_k (p_ik - 1[y_i = k])², en [0, 2].

    Por qué métricas ordinales
    --------------------------
    El estrato es ORDINAL: accuracy y F1 tratan igual cualquier error, pero
    predecir 3 para un predio de estrato 2 es mucho menos grave que predecir
    6. MAE, accuracy ±1 y kappa cuadrático recogen esa gravedad.
    """
    y = np.asarray(y)
    pred = np.asarray(pred)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        # zero_division=0: una clase nunca predicha aporta precisión 0 (sin warning)
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        # el estrato es ORDINAL: estas dos métricas premian equivocarse "por poco"
        "MAE_ordinal": np.mean(np.abs(y - pred)),
        "accuracy_±1": np.mean(np.abs(y - pred) <= 1),
        "kappa_cuadrático": cohen_kappa_score(y, pred, weights="quadratic"),
    }
    if proba is not None:
        # en un fold espacial puede faltar algún estrato en y (p. ej. sin estrato 6):
        # roc_auc_score exige que las etiquetas estén presentes y que las
        # probabilidades sumen 1, así que se restringe a las clases presentes y se
        # renormaliza cada fila
        presentes = np.isin(clases, np.unique(y))
        try:
            out["AUC_OvR_macro"] = roc_auc_score(y, proba[:, presentes] / proba[:, presentes].sum(1, keepdims=True),
                                                 multi_class="ovr", average="macro",
                                                 labels=np.asarray(clases)[presentes])
        except ValueError:  # p. ej. una sola clase presente: el AUC no está definido
            out["AUC_OvR_macro"] = np.nan
        # clip evita ln(0) = -inf cuando el modelo asigna probabilidad 0 a la clase real
        out["log_loss"] = log_loss(y, np.clip(proba, 1e-12, 1), labels=clases)
        # Y = codificación one-hot de y (n x K) para el Brier
        Y = (y[:, None] == np.asarray(clases)[None, :]).astype(float)
        out["brier_multiclase"] = np.mean(((proba - Y) ** 2).sum(1))
    return out


def bootstrap_por_bloques(y, pred, proba, bloques, B=500, seed=C.SEED, funciones=None):
    """IC 95 % por bootstrap de BLOQUES (se remuestrean bloques completos, no
    filas): con autocorrelación espacial las filas no son independientes y el
    bootstrap por filas daría intervalos demasiado estrechos (Efron 1979;
    idea de 'block bootstrap' de Künsch 1989).

    Parámetros
    ----------
    y, pred : array-like (n,)
        Estrato real y predicho (predicciones fuera de muestra, p. ej. de la
        validación cruzada espacial).
    proba : numpy.ndarray (n, K) o None
        Probabilidades, por si alguna función las usa.
    bloques : array-like (n,)
        Identificador del bloque espacial de cada fila.
    B : int, default 500
        Número de réplicas bootstrap.
    seed : int, default C.SEED
        Semilla.
    funciones : dict[str, callable] o None
        {nombre: f(y, pred, proba) -> float}. Por defecto: accuracy, F1 macro,
        MAE ordinal y kappa cuadrático.

    Devuelve
    --------
    pandas.DataFrame
        Una fila por métrica con IC95_inf, IC95_sup (percentiles 2.5 y 97.5
        de las réplicas: método del percentil) y desv_boot (error estándar
        bootstrap).

    Procedimiento
    -------------
    En cada réplica se sortean CON reemplazo tantos bloques como bloques
    distintos hay (G), y se juntan todas las filas de los bloques elegidos
    (un bloque puede entrar varias veces). El número de filas por réplica
    varía, porque los bloques tienen tamaños distintos.

    Por qué por bloques
    -------------------
    El bootstrap clásico supone observaciones i.i.d. Predios vecinos
    comparten estrato y errores, así que n filas equivalen a muchas menos
    observaciones independientes; remuestrear bloques completos preserva la
    dependencia interna y da intervalos honestos (más anchos). La unidad de
    remuestreo coincide con la unidad de la validación cruzada espacial.
    """
    rng = np.random.default_rng(seed)
    y, pred, bloques = np.asarray(y), np.asarray(pred), np.asarray(bloques)
    ub, inv = np.unique(bloques, return_inverse=True)  # ub = bloques únicos; inv = código 0..G-1
    idx_por_bloque = [np.where(inv == b)[0] for b in range(len(ub))]  # filas de cada bloque
    if funciones is None:
        # cada función recibe (y, pred, proba) de la réplica; pr se ignora en estas cuatro
        funciones = {
            "accuracy": lambda yy, pp, pr: accuracy_score(yy, pp),
            "f1_macro": lambda yy, pp, pr: f1_score(yy, pp, average="macro", zero_division=0),
            "MAE_ordinal": lambda yy, pp, pr: np.mean(np.abs(yy - pp)),
            "kappa_cuadrático": lambda yy, pp, pr: cohen_kappa_score(yy, pp, weights="quadratic"),
        }
    res = {k: [] for k in funciones}
    for _ in range(B):
        elegidos = rng.integers(0, len(ub), len(ub))  # G bloques con reemplazo
        ii = np.concatenate([idx_por_bloque[b] for b in elegidos])  # filas de la réplica
        for k, f in funciones.items():
            try:
                res[k].append(f(y[ii], pred[ii], None if proba is None else proba[ii]))
            except ValueError:
                pass  # réplica degenerada para esa métrica (p. ej. una sola clase): se omite
    # método del percentil: IC = [P2.5, P97.5] de la distribución bootstrap
    return pd.DataFrame({k: {"IC95_inf": np.percentile(v, 2.5), "IC95_sup": np.percentile(v, 97.5),
                             "desv_boot": np.std(v)} for k, v in res.items()}).T


def bootstrap_diferencia(y, pred_a, pred_b, bloques, metrica, B=500, seed=C.SEED):
    """Bootstrap PAREADO por bloques de metrica(a) - metrica(b).

    Parámetros
    ----------
    y : array-like (n,)
        Estrato real.
    pred_a, pred_b : array-like (n,)
        Predicciones de los dos modelos a comparar (p. ej. regresión
        logística vs BaselineModaZona) sobre las MISMAS filas.
    bloques : array-like (n,)
        Bloque espacial de cada fila.
    metrica : callable
        f(y, pred) -> float (p. ej. f1 macro o MAE ordinal).
    B : int, default 500
        Réplicas bootstrap.
    seed : int, default C.SEED
        Semilla.

    Devuelve
    --------
    dict
        diferencia observada (a - b), IC95_inf, IC95_sup (percentiles) y
        p_bootstrap (bilateral, H0: diferencia = 0).

    Fórmula del p-valor
    -------------------
    p = 2 · min( P*(Δ* <= 0), P*(Δ* >= 0) ), con Δ* las diferencias
    bootstrap; se acota inferiormente en 1/B porque con B réplicas no se
    puede afirmar un p menor.

    Por qué pareado
    ---------------
    Ambos modelos se evalúan sobre la MISMA réplica (mismos bloques) en cada
    iteración. Así la variabilidad común (bloques "fáciles" o "difíciles")
    se cancela en la diferencia y el intervalo es mucho más estrecho que si
    se compararan dos IC independientes. Si el IC de la diferencia excluye
    0, la ventaja de un modelo es consistente entre zonas de la ciudad. Para
    métricas donde "menor es mejor" (MAE) una diferencia negativa favorece a a.
    """
    rng = np.random.default_rng(seed)
    y, pred_a, pred_b, bloques = map(np.asarray, (y, pred_a, pred_b, bloques))
    ub, inv = np.unique(bloques, return_inverse=True)
    idx_por_bloque = [np.where(inv == b)[0] for b in range(len(ub))]
    difs = []
    for _ in range(B):
        # mismos índices ii para ambos modelos: esto es lo que hace "pareado" al bootstrap
        ii = np.concatenate([idx_por_bloque[b] for b in rng.integers(0, len(ub), len(ub))])
        difs.append(metrica(y[ii], pred_a[ii]) - metrica(y[ii], pred_b[ii]))
    difs = np.array(difs)
    obs = metrica(y, pred_a) - metrica(y, pred_b)  # diferencia en la muestra original
    p = 2 * min((difs <= 0).mean(), (difs >= 0).mean())  # bilateral por inversión del IC
    return {"diferencia": obs, "IC95_inf": np.percentile(difs, 2.5),
            "IC95_sup": np.percentile(difs, 97.5), "p_bootstrap": max(p, 1 / B)}
