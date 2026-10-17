# -*- coding: utf-8 -*-
"""
Funciones estadísticas del EDA con las tablas de interpretación de las notas
de clase del profesor Lihki Rubio (09_model_evaluation.md, 9.1.1) y
convenciones estándar de tamaño de efecto (Cohen 1988; Tomczak & Tomczak 2014).

Organización del módulo
-----------------------
1. Interpretaciones (umbrales): traducen un número (asimetría, eta², V de
   Cramér, VIF, AUC...) a una lectura verbal. Se centralizan aquí para que
   todas las tablas del libro usen EXACTAMENTE los mismos cortes.
2. Univariado: resumen descriptivo robusto, pruebas de normalidad y
   frecuencias de variables categóricas.
3. Bidimensional: asociación de cada variable con el estrato (Kruskal-Wallis
   + eta²_H para numéricas, V de Cramér para categóricas), correcciones por
   comparaciones múltiples (Holm, Benjamini-Hochberg), Pearson vs Spearman,
   multicolinealidad (VIF) y entropía de la variable objetivo.

Por qué se prefieren métodos no paramétricos: las variables catastrales
(áreas, avalúos, número de pisos) son muy asimétricas y con colas pesadas, y el
estrato es ORDINAL (1 < 2 < ... < 6), no de intervalo. Las pruebas basadas en
rangos (Kruskal-Wallis, Spearman) no suponen normalidad ni homocedasticidad.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Interpretaciones (umbrales)
# ---------------------------------------------------------------------------

def interpretar_asimetria(s):
    """Traduce el coeficiente de asimetría (skewness) a una lectura verbal.

    Parámetros
    ----------
    s : float
        Coeficiente de asimetría g1 = m3 / m2^(3/2) (momentos centrales).

    Devuelve
    --------
    str
        "aprox. simétrica" si |s| < 0.5; "moderadamente asimétrica" si
        0.5 <= |s| < 1; "altamente asimétrica" si |s| >= 1; "sin dato" si NaN.

    Por qué
    -------
    Se usa el valor absoluto porque el corte es el mismo para colas a la
    derecha (s > 0, típico de áreas y avalúos) y a la izquierda (s < 0). Los
    umbrales 0.5 / 1 son la convención habitual (Bulmer 1979). Una asimetría
    alta sugiere transformar la variable (p. ej. log1p) antes de modelos
    lineales o basados en distancias.
    """
    a = abs(s)
    if pd.isna(a):
        return "sin dato"
    if a < 0.5:
        return "aprox. simétrica"
    if a < 1.0:
        return "moderadamente asimétrica"
    return "altamente asimétrica"


def interpretar_curtosis(k):
    """Traduce la curtosis en EXCESO (Fisher, normal = 0) a una lectura verbal.

    Parámetros
    ----------
    k : float
        Curtosis de Fisher g2 = m4 / m2^2 - 3 (0 para la distribución normal).

    Devuelve
    --------
    str
        "similar a la normal" si |k| < 0.5; "moderada, <forma>" si
        0.5 <= |k| < 1; "fuerte, <forma>" si |k| >= 1, donde <forma> es
        "leptocúrtica (colas pesadas)" si k > 0 y "platicúrtica" si k <= 0.

    Por qué
    -------
    Una curtosis positiva grande indica colas pesadas: más valores extremos
    de los que predice una normal, lo que justifica winsorizar y usar
    estadísticos robustos (mediana, IQR) en lugar de media y desviación.
    """
    a = abs(k)
    if pd.isna(a):
        return "sin dato"
    # el signo decide la forma; el valor absoluto decide la intensidad
    forma = "leptocúrtica (colas pesadas)" if k > 0 else "platicúrtica"
    if a < 0.5:
        return "similar a la normal"
    if a < 1.0:
        return f"moderada, {forma}"
    return f"fuerte, {forma}"


def interpretar_eta2(e):
    """Clasifica un tamaño de efecto eta² (proporción de varianza explicada).

    Parámetros
    ----------
    e : float
        Tamaño de efecto eta² (aquí, eta²_H de Kruskal-Wallis), en [0, 1].

    Devuelve
    --------
    str
        "trivial" (< 0.01), "pequeño" (0.01-0.06), "mediano" (0.06-0.14) o
        "grande" (>= 0.14); "sin dato" si NaN.

    Por qué
    -------
    Son los cortes de Cohen (1988) para eta². Con n de decenas de miles
    cualquier diferencia minúscula es "significativa" (p ~ 0); el tamaño de
    efecto responde a la pregunta relevante: ¿CUÁNTO separa la variable los
    estratos?
    """
    if pd.isna(e):
        return "sin dato"
    if e < 0.01:
        return "trivial"
    if e < 0.06:
        return "pequeño"
    if e < 0.14:
        return "mediano"
    return "grande"


def interpretar_v(v):
    """Clasifica la fuerza de asociación de la V de Cramér.

    Parámetros
    ----------
    v : float
        V de Cramér, en [0, 1] (0 = independencia, 1 = asociación perfecta).

    Devuelve
    --------
    str
        "despreciable" (< 0.1), "débil" (0.1-0.3), "moderada" (0.3-0.5) o
        "fuerte" (>= 0.5); "sin dato" si NaN.

    Nota
    ----
    Son los cortes habituales de Cohen para gl* = min(r, k) - 1 = 1. Con
    más grados de libertad los cortes "reales" son algo menores, por lo que
    esta lectura es conservadora (tiende a calificar como más débil).
    """
    if pd.isna(v):
        return "sin dato"
    if v < 0.1:
        return "despreciable"
    if v < 0.3:
        return "débil"
    if v < 0.5:
        return "moderada"
    return "fuerte"


def interpretar_dif_pearson_spearman(d):
    """Regla de decisión entre Pearson y Spearman según |r - rho|.

    Parámetros
    ----------
    d : float
        Diferencia absoluta |r_Pearson - rho_Spearman| de un par de variables.

    Devuelve
    --------
    str
        < 0.1 -> relación esencialmente lineal (se reporta Pearson);
        0.1-0.2 -> revisar gráficamente (scatter/hexbin);
        >= 0.2 -> relación monótona no lineal u outliers (priorizar Spearman).

    Por qué
    -------
    Pearson mide asociación LINEAL y es sensible a valores extremos;
    Spearman es Pearson sobre los rangos y mide asociación MONÓTONA. Si ambos
    coinciden, la relación es aproximadamente lineal; si difieren mucho, la
    no linealidad o los outliers están distorsionando a Pearson.
    """
    if d < 0.1:
        return "trivial -> relación lineal (reportar Pearson)"
    if d < 0.2:
        return "moderada -> revisar scatter/hexbin"
    return "notable -> no lineal u outliers (priorizar Spearman)"


def interpretar_vif(v):
    """Clasifica el factor de inflación de la varianza (VIF).

    Parámetros
    ----------
    v : float
        VIF_j = 1 / (1 - R²_j).

    Devuelve
    --------
    str
        "aceptable" (< 5), "moderada (vigilar)" (5-10) o
        "alta multicolinealidad" (>= 10).

    Interpretación
    --------------
    VIF = 5 equivale a R²_j = 0.8 y VIF = 10 a R²_j = 0.9: la variable está
    explicada en 80 % / 90 % por las demás. sqrt(VIF) es el factor en que se
    infla el error estándar de su coeficiente en un modelo lineal.
    """
    if v < 5:
        return "aceptable"
    if v < 10:
        return "moderada (vigilar)"
    return "alta multicolinealidad"


def interpretar_auc_univariado(a):
    """Interpreta el AUC que alcanza UNA sola variable para separar clases.

    Parámetros
    ----------
    a : float
        AUC de la ROC usando solo esa variable como puntaje (0.5 = azar).

    Devuelve
    --------
    str
        >= 0.95 alerta de posible fuga de información (leakage) o proxy;
        0.85-0.95 muy alto (revisar); 0.7-0.85 informativo; 0.6-0.7 débil;
        < 0.6 casi nulo.

    Por qué
    -------
    Una variable individual que casi separa perfectamente el estrato suele
    ser sospechosa: puede derivarse del propio estrato (p. ej. una tarifa
    o un código asignado a partir de él) y no estaría disponible, o no sería
    legítima, al momento de predecir. Detectarlo en el EDA evita un modelo
    con desempeño artificialmente inflado.
    """
    if a >= 0.95:
        return "ALERTA: casi perfecto, posible fuga/proxy"
    if a >= 0.85:
        return "muy alto: revisar que no sea proxy del objetivo"
    if a >= 0.7:
        return "informativo"
    if a >= 0.6:
        return "débil"
    return "casi nulo"


# ---------------------------------------------------------------------------
# Univariado
# ---------------------------------------------------------------------------

def resumen_numerico(df, cols):
    """Tabla descriptiva completa de variables numéricas (una fila por variable).

    Calcula tamaño, % de nulos, tendencia central (media, mediana), dispersión
    (desviación estándar, IQR), percentiles extremos (p1, p5, p95, p99),
    forma (asimetría, curtosis en exceso) y outliers según las cercas de Tukey.

    Parámetros
    ----------
    df : pandas.DataFrame
        Datos de entrada.
    cols : list[str]
        Columnas a resumir; se convierten a número (lo no convertible -> NaN).

    Devuelve
    --------
    pandas.DataFrame
        Índice = variable. Las columnas "lectura_asimetría" y
        "lectura_curtosis" traen la interpretación verbal de los umbrales.

    Fórmulas
    --------
    IQR = Q3 - Q1. Outlier de Tukey: x < Q1 - 1.5·IQR  o  x > Q3 + 1.5·IQR.
    Asimetría g1 = m3 / m2^(3/2); curtosis de Fisher g2 = m4 / m2^2 - 3
    (scipy, estimadores sesgados por defecto; diferencia despreciable con n
    grande).

    Por qué
    -------
    Con variables sesgadas la media y la desviación pueden engañar; por eso
    se reportan junto a mediana, IQR y percentiles, que son robustos. Los
    percentiles 1 y 99 anticipan los límites de winsorización del modelo.
    Las cercas de Tukey no "borran" nada: solo cuantifican la cola.
    """
    filas = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        v = s.dropna()  # estadísticos sobre valores observados; nulos se reportan aparte
        if len(v) == 0:
            continue  # columna totalmente vacía: no hay nada que resumir
        q1, q3 = v.quantile([.25, .75])
        iqr = q3 - q1
        # cercas de Tukey (1977): 1.5·IQR fuera de la caja del boxplot
        fuera = ((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()
        sk, ku = stats.skew(v), stats.kurtosis(v)  # Fisher: normal = 0
        filas.append({
            "variable": c, "n": len(v), "% nulos": 100 * s.isna().mean(),
            "media": v.mean(), "mediana": v.median(), "desv_est": v.std(),
            "min": v.min(), "p1": v.quantile(.01), "p5": v.quantile(.05), "p25": q1,
            "p75": q3, "p95": v.quantile(.95), "p99": v.quantile(.99), "max": v.max(),
            "IQR": iqr, "asimetría": sk, "curtosis": ku,
            "outliers_Tukey": int(fuera), "% outliers": 100 * fuera / len(v),
            "lectura_asimetría": interpretar_asimetria(sk),
            "lectura_curtosis": interpretar_curtosis(ku),
        })
    return pd.DataFrame(filas).set_index("variable")


def pruebas_normalidad(df, cols, n=5000, seed=42):
    """D'Agostino-Pearson y Shapiro-Wilk sobre una submuestra (Shapiro no es
    fiable con n > 5000). Con n grande casi siempre se rechaza normalidad
    aunque la desviación sea irrelevante: se reportan con cautela.

    Parámetros
    ----------
    df : pandas.DataFrame
        Datos de entrada.
    cols : list[str]
        Columnas numéricas a evaluar.
    n : int, default 5000
        Tamaño máximo de la submuestra aleatoria (sin reemplazo).
    seed : int, default 42
        Semilla para que la submuestra sea reproducible.

    Devuelve
    --------
    pandas.DataFrame
        Índice = variable; estadístico K² y su p, W de Shapiro y su p, y la
        decisión "¿normal al 5%?" ("no" si CUALQUIERA de las dos rechaza).

    Fórmulas
    --------
    K² = Z(g1)² + Z(g2)² ~ chi²(2) bajo H0 (combina asimetría y curtosis
    transformadas a normales estándar). W de Shapiro-Wilk = (sum a_i x_(i))²
    / sum (x_i - x̄)², cercano a 1 bajo normalidad.

    Por qué
    -------
    H0 en ambas pruebas es "los datos son normales". La potencia crece con n,
    así que con miles de datos se rechaza por desviaciones sin importancia
    práctica: la decisión de transformar se toma mirando asimetría/curtosis
    y los gráficos (histograma, QQ-plot), no solo el p-valor. Se toma
    min(p_K2, p_W) como criterio conservador hacia "no normal".
    """
    rng = np.random.default_rng(seed)
    filas = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy()
        # normaltest exige n >= 20 (usa la prueba de curtosis) y ambas pruebas
        # fallan o carecen de sentido con una variable constante (varianza 0)
        if len(v) < 20 or np.std(v) == 0:
            continue
        m = v[rng.choice(len(v), min(n, len(v)), replace=False)]  # submuestra sin reemplazo
        k2, p_k2 = stats.normaltest(m)
        w, p_w = stats.shapiro(m)
        filas.append({"variable": c, "n_submuestra": len(m), "D'Agostino K2": k2,
                      "p (K2)": p_k2, "Shapiro W": w, "p (Shapiro)": p_w,
                      "¿normal al 5%?": "no" if min(p_k2, p_w) < 0.05 else "no se rechaza"})
    return pd.DataFrame(filas).set_index("variable")


def resumen_categorico(df, col, umbral_rara=0.01):
    """Tabla de frecuencias de una variable categórica, marcando las raras.

    Parámetros
    ----------
    df : pandas.DataFrame
        Datos de entrada.
    col : str
        Columna categórica.
    umbral_rara : float, default 0.01
        Proporción por debajo de la cual una categoría se marca como "rara".

    Devuelve
    --------
    pandas.DataFrame
        Índice = categoría (incluye los nulos como una categoría más, gracias
        a dropna=False); columnas "frecuencia", "%" y "rara (<1%)".

    Por qué
    -------
    Las categorías raras generan columnas one-hot casi vacías, inestables
    entre folds, y pueden no aparecer en train. El mismo umbral (1 %) se usa
    luego en OneHotEncoder(min_frequency=0.01) del preprocesador, que las
    agrupa en 'infrequent'.
    """
    s = df[col].astype("string")  # tipo texto homogéneo (evita mezclar 1 y "1")
    tabla = s.value_counts(dropna=False).rename("frecuencia").to_frame()
    tabla["%"] = 100 * tabla["frecuencia"] / len(s)
    tabla["rara (<1%)"] = tabla["%"] < 100 * umbral_rara  # umbral pasado a porcentaje
    return tabla


# ---------------------------------------------------------------------------
# Bidimensional
# ---------------------------------------------------------------------------

def kruskal_eta2(df, col, objetivo, ordinal=True):
    """Kruskal-Wallis (alternativa no paramétrica al ANOVA de una vía; la
    generalización de Mann-Whitney a >2 grupos) + tamaño de efecto
    eta2_H = (H - k + 1) / (n - k)  (Tomczak & Tomczak 2014).

    Parámetros
    ----------
    df : pandas.DataFrame
        Datos de entrada.
    col : str
        Variable numérica cuya distribución se compara entre grupos.
    objetivo : str
        Variable que define los grupos (el estrato, 1-6).
    ordinal : bool, default True
        Si True, además calcula rho de Spearman entre col y objetivo (solo
        tiene sentido si los grupos tienen orden natural).

    Devuelve
    --------
    dict
        variable, n, H, p, eta2_H, lectura del efecto, rho de Spearman y la
        mediana de col en cada grupo (claves "mediana_e1", ..., "mediana_e6").

    Fórmulas
    --------
    H = [12 / (n(n+1))] · sum_g n_g · R̄_g² - 3(n+1), con R̄_g el rango medio
    del grupo g (scipy aplica además la corrección por empates). Bajo H0
    ("todas las distribuciones son iguales") H ~ chi²(k-1).
    eta²_H = (H - k + 1) / (n - k): proporción de la variabilidad de los
    rangos explicada por el grupo; se interpreta con los cortes de Cohen
    (0.01 / 0.06 / 0.14).

    Por qué
    -------
    Con n muy grande el p-valor de H es prácticamente 0 para casi cualquier
    variable, por lo que no sirve para ORDENAR variables por importancia;
    eta²_H sí. Spearman complementa: H detecta cualquier diferencia entre
    grupos, rho indica si la variable crece (o decrece) de forma MONÓTONA con
    el estrato. Las medianas por grupo muestran la dirección y forma.
    """
    d = df[[col, objetivo]].dropna()  # casos completos para el par (col, objetivo)
    grupos = [g[col].to_numpy() for _, g in d.groupby(objetivo)]
    grupos = [g for g in grupos if len(g) > 0]
    n, k = len(d), len(grupos)
    try:
        H, p = stats.kruskal(*grupos)
        # eta²_H puede salir negativo si H < k - 1 (efecto nulo + ruido): se trunca en 0
        eta2 = max(0.0, (H - k + 1) / (n - k))
    except ValueError:  # < 2 grupos o todos los valores idénticos
        H, p, eta2 = np.nan, np.nan, np.nan
    # Spearman solo tiene sentido si el agrupador es ordinal (el estrato)
    rho = stats.spearmanr(d[col], d[objetivo])[0] if (ordinal and k > 1) else np.nan
    medianas = d.groupby(objetivo)[col].median()
    # las medianas por grupo se despliegan como columnas: "mediana_e<estrato>" si
    # la etiqueta es entera, "mediana_<etiqueta>" en otro caso
    return {"variable": col, "n": n, "H": H, "p": p, "eta2_H": eta2,
            "efecto": interpretar_eta2(eta2), "spearman_con_objetivo": rho,
            **{(f"mediana_e{int(c)}" if isinstance(c, (int, np.integer)) else f"mediana_{c}"): v
               for c, v in medianas.items()}}


def cramers_v(x, y, corregido=True):
    """V de Cramér; con corrección de sesgo de Bergsma (2013) si corregido.

    Mide la fuerza de asociación entre dos variables categóricas a partir
    del chi² de independencia de su tabla de contingencia.

    Parámetros
    ----------
    x, y : array-like o pandas.Series
        Las dos variables categóricas (misma longitud).
    corregido : bool, default True
        Si True aplica la corrección de sesgo de Bergsma (2013).

    Devuelve
    --------
    dict
        chi2, gl (grados de libertad), p, V, n y la tabla de contingencia.

    Fórmulas
    --------
    phi² = chi² / n;  V = sqrt( phi² / min(k - 1, r - 1) ), con r filas y
    k columnas.
    Corrección de Bergsma:
        phi²_c = max(0, phi² - (k-1)(r-1)/(n-1))
        r_c = r - (r-1)²/(n-1),  k_c = k - (k-1)²/(n-1)
        V_c = sqrt( phi²_c / min(k_c - 1, r_c - 1) )

    Por qué
    -------
    La V clásica está sesgada hacia arriba: aun con variables independientes
    E[chi²] ≈ (r-1)(k-1), así que V > 0 por puro azar, y el sesgo crece con
    el número de categorías y cuando n es pequeño. Esto haría parecer más
    asociadas a variables con muchas categorías (p. ej. barrio). Bergsma resta
    ese valor esperado bajo independencia y ajusta las dimensiones, lo que
    hace comparables las V de variables con distinta cardinalidad.
    Se usa correction=False (sin corrección de Yates), que solo aplica a
    tablas 2x2 y es demasiado conservadora.
    """
    tabla = pd.crosstab(x, y)  # filas = categorías de x, columnas = categorías de y
    chi2, p, dof, _ = stats.chi2_contingency(tabla, correction=False)
    n = tabla.to_numpy().sum()
    r, k = tabla.shape
    phi2 = chi2 / n
    if corregido:
        # Bergsma (2013): se resta el phi² esperado bajo independencia
        phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        rc = r - (r - 1) ** 2 / (n - 1)  # número de filas "efectivo"
        kc = k - (k - 1) ** 2 / (n - 1)  # número de columnas "efectivo"
        den = min(kc - 1, rc - 1)
        v = np.sqrt(phi2c / den) if den > 0 else np.nan  # den <= 0: tabla degenerada (1 fila/col)
    else:
        den = min(k - 1, r - 1)
        v = np.sqrt(phi2 / den) if den > 0 else np.nan
    return {"chi2": chi2, "gl": dof, "p": p, "V": v, "n": n, "tabla": tabla}


def ajuste_holm(pvals):
    """Corrección de Holm-Bonferroni (controla el error de familia; más
    potente que Bonferroni). Devuelve p ajustados en el orden original.

    Parámetros
    ----------
    pvals : array-like
        p-valores de las m pruebas (una por variable, por ejemplo).

    Devuelve
    --------
    numpy.ndarray
        p-valores ajustados, en el mismo orden de entrada, acotados a 1.

    Fórmula
    -------
    Ordenando p_(1) <= ... <= p_(m):
        p̃_(i) = min(1, max_{j <= i} (m - j + 1) · p_(j))
    El máximo acumulado garantiza que los p ajustados sean monótonos
    (procedimiento "step-down").

    Por qué
    -------
    Al probar muchas variables contra el estrato, la probabilidad de al menos
    un falso positivo (FWER) crece como 1 - (1 - alpha)^m. Holm controla la
    FWER <= alpha sin supuestos de independencia y rechaza siempre al menos
    lo mismo que Bonferroni (que multiplica todos por m).
    """
    p = np.asarray(pvals, float)
    m = len(p)
    orden = np.argsort(p)  # índices de menor a mayor p
    ajustado = np.empty(m)
    acumulado = 0.0
    for rango, i in enumerate(orden):
        # rango 0-based: el menor p se multiplica por m, el siguiente por m-1, ...
        acumulado = max(acumulado, (m - rango) * p[i])  # máximo acumulado -> monotonía
        ajustado[i] = min(1.0, acumulado)  # se devuelve en la posición original
    return ajustado


def ajuste_bh(pvals):
    """Benjamini-Hochberg (controla la tasa de falsos descubrimientos, FDR).

    Parámetros
    ----------
    pvals : array-like
        p-valores de las m pruebas.

    Devuelve
    --------
    numpy.ndarray
        q-valores (p ajustados por BH) en el orden original, <= 1.

    Fórmula
    -------
    Ordenando p_(1) <= ... <= p_(m):
        q_(i) = min_{j >= i} min(1, p_(j) · m / j)
    Se recorre de mayor a menor aplicando el mínimo acumulado
    (procedimiento "step-up").

    Por qué
    -------
    En un EDA exploratorio interesa más no perder asociaciones reales que
    blindarse contra cualquier falso positivo: la FDR (proporción esperada
    de falsos positivos entre los rechazados) es menos estricta que la FWER
    de Holm y, por tanto, más potente. Se reportan ambas para contrastar.
    Válido bajo independencia o dependencia positiva (PRDS) entre pruebas.
    """
    p = np.asarray(pvals, float)
    m = len(p)
    orden = np.argsort(p)
    ajustado = np.empty(m)
    minimo = 1.0  # arrancar en 1 acota los q-valores a [0, 1]
    for rango in range(m - 1, -1, -1):  # del p más grande al más pequeño
        i = orden[rango]
        minimo = min(minimo, p[i] * m / (rango + 1))  # rango+1 = posición 1-based j
        ajustado[i] = minimo
    return ajustado


def pearson_vs_spearman(df, cols):
    """Compara las matrices de correlación de Pearson y Spearman por pares.

    Parámetros
    ----------
    df : pandas.DataFrame
        Datos de entrada.
    cols : list[str]
        Variables numéricas a comparar.

    Devuelve
    --------
    tuple (P, S, tabla)
        P : matriz de Pearson; S : matriz de Spearman (ambas con eliminación
        de nulos por pares, comportamiento por defecto de pandas);
        tabla : un par por fila con r, rho, |dif| y la lectura de
        interpretar_dif_pearson_spearman, ordenada de mayor a menor |dif|.

    Fórmulas
    --------
    r = cov(X, Y) / (s_X · s_Y);  rho = r calculado sobre rangos(X), rangos(Y).

    Por qué
    -------
    La diferencia |r - rho| es un diagnóstico sencillo de no linealidad o de
    influencia de outliers; los pares con mayor diferencia (arriba en la
    tabla) son los que conviene inspeccionar gráficamente.
    """
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    P = d.corr("pearson")
    S = d.corr("spearman")
    filas = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:  # solo el triángulo superior: cada par una vez
            dif = abs(P.loc[a, b] - S.loc[a, b])
            filas.append({"var1": a, "var2": b, "pearson": P.loc[a, b], "spearman": S.loc[a, b],
                          "|dif|": dif, "lectura": interpretar_dif_pearson_spearman(dif)})
    return P, S, pd.DataFrame(filas).sort_values("|dif|", ascending=False)


def vif(df, cols):
    """VIF_j = 1 / (1 - R2_j), R2_j de regresar la variable j sobre las demás
    (con intercepto). Implementado con mínimos cuadrados de numpy para no
    depender de statsmodels.

    Parámetros
    ----------
    df : pandas.DataFrame
        Datos de entrada.
    cols : list[str]
        Variables numéricas candidatas a predictores.

    Devuelve
    --------
    pandas.DataFrame
        Índice = variable; columnas R2_con_las_demás, VIF y lectura
        (interpretar_vif), ordenado de mayor a menor VIF.

    Por qué
    -------
    La multicolinealidad no daña la capacidad predictiva, pero vuelve
    inestables e ininterpretables los coeficientes de la regresión logística
    (el modelo base). El VIF detecta qué variables son casi combinación lineal
    de otras (p. ej. área construida y área de terreno) para decidir si
    eliminar, combinar o regularizar.

    Notas
    -----
    Se usan solo filas completas (eliminación por lista). Se estandarizan las
    columnas antes de la regresión: R² no cambia con la escala, pero mejora el
    condicionamiento numérico. El piso 1e-12 evita dividir por cero si una
    variable es combinación lineal exacta de otras (VIF -> ~1e12).
    Supone que ninguna columna es constante (su desviación sería 0).
    """
    X = df[cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)  # estandarización z por columna
    filas = []
    for j, c in enumerate(cols):
        y = X[:, j]  # variable j como respuesta
        # diseño = intercepto + todas las demás columnas
        otros = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(otros, y, rcond=None)  # MCO: min ||y - X·beta||²
        r2 = 1 - ((y - otros @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()  # 1 - SSE/SST
        v = 1 / max(1e-12, 1 - r2)
        filas.append({"variable": c, "R2_con_las_demás": r2, "VIF": v, "lectura": interpretar_vif(v)})
    return pd.DataFrame(filas).set_index("variable").sort_values("VIF", ascending=False)


def entropia(y):
    """Entropía de Shannon (en nats) de la distribución de clases de y.

    Parámetros
    ----------
    y : array-like
        Etiquetas (p. ej. el estrato de cada predio).

    Devuelve
    --------
    float
        H(Y) = - sum_c p_c · ln(p_c), con p_c la frecuencia relativa de c.

    Interpretación
    --------------
    H = 0 si hay una sola clase; el máximo es ln(K) cuando las K clases son
    equiprobables (ln 6 ≈ 1.792 para 6 estratos). H / ln(K) da una medida
    normalizada de balance: valores bajos indican clases desbalanceadas, lo
    que justifica reportar balanced accuracy y F1 macro además de accuracy.
    También es la log-loss de un clasificador que predice siempre las
    frecuencias globales, una referencia natural para la log-loss del modelo.
    """
    p = pd.Series(y).value_counts(normalize=True).to_numpy()  # solo clases observadas (p > 0)
    return float(-(p * np.log(p)).sum())
