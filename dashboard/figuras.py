"""Datos y figuras del dashboard del Entregable 1 (versiones 1 y 2).

Autores: María Carolina Cantillo Orozco (200179105) y
         Juan Camilo Oñoro Araujo (200177329)

Este módulo es la ÚNICA fuente de las figuras: lo usan la app Dash
(``app.py``) y los cuadernos del Jupyter Book, así que lo que se ve en el
dashboard y lo que se interpreta en el informe es exactamente lo mismo.
Todas las funciones leen los archivos de ``datos/<versión>/`` (v1 o v2). La versión activa se fija con
``with usar("v2"):``; un candado evita que dos peticiones con versiones distintas se mezclen.
"""
import json
import os
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support, roc_curve, auc)

DATOS_RAIZ = Path(__file__).resolve().parent / "datos"
VERSIONES = {"v1": "Versión 1 (entregada)", "v2": "Versión 2 (con ubicación por manzana)"}
_ESTADO = {"v": "v2"}
_CANDADO = threading.RLock()


@contextmanager
def usar(version):
    """Fija la versión de los datos mientras dura el bloque (con candado: seguro con varios hilos)."""
    with _CANDADO:
        previa = _ESTADO["v"]
        _ESTADO["v"] = version
        try:
            yield
        finally:
            _ESTADO["v"] = previa


def _dir():
    return DATOS_RAIZ / _ESTADO["v"]


@lru_cache(maxsize=None)
def _leer(version, nombre):
    ruta = DATOS_RAIZ / version / nombre
    if nombre.endswith(".json"):
        return json.loads(ruta.read_text(encoding="utf-8"))
    if nombre.endswith(".joblib"):
        import joblib
        return joblib.load(ruta)
    return pd.read_pickle(ruta)
CLASES = [1, 2, 3, 4, 5, 6]

# ---------------------------------------------------------------------------
# Paleta (validada): rampa azul ordinal para el estrato, colores fijos por
# modelo, divergente azul-rojo con gris en el centro.
# ---------------------------------------------------------------------------
COLOR_ESTRATO = {1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6", 4: "#1c5cab", 5: "#104281", 6: "#0d366b"}
COLOR_MODELO = {
    "Dummy (mayoritaria)": "#bdbcb7", "Dummy (estratificada)": "#a3a29d", "Dummy (uniforme)": "#8a8984",
    "Moda por zona": "#eb6834", "Logística A_fisicas": "#1baf7a", "Logística B_fisicas+ubicacion": "#2a78d6",
}
NOMBRE_CORTO = {
    "Dummy (mayoritaria)": "Dummy mayoritaria", "Dummy (estratificada)": "Dummy estratificada",
    "Dummy (uniforme)": "Dummy uniforme", "Moda por zona": "Moda por zona (2 km)",
    "Logística A_fisicas": "Logística A (físicas)", "Logística B_fisicas+ubicacion": "Logística B (físicas + ubicación)",
}
SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGENTE = [[0, "#c22f2e"], [0.25, "#e98a89"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1, "#1c5cab"]]
# Estilo del mapa base (MapLibre). "white-bg" sirve sin conexión a internet.
ESTILO_MAPA = os.environ.get("ESTILO_MAPA", "carto-positron")
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e6e5e1"

NUMERICAS = {
    "area_construida": "Área construida (m²)", "area_catastral_terreno": "Área de terreno (m²)",
    "total_habitaciones": "Habitaciones", "total_banios": "Baños", "total_plantas": "Pisos del edificio",
    "planta_ubicacion": "Piso de ubicación", "altura": "Altura", "antiguedad": "Antigüedad (años)",
    "dist_centro_km": "Distancia al centro (km)",
}
CATEGORICAS = {"condicion_predio": "Condición del predio (régimen)", "uso": "Uso", "tipo_vivienda": "Tipo de vivienda"}

# Embudo de limpieza del Entregable 1 (cap. 1).
EMBUDO = [("Filas descargadas", 382_597), ("Usos no habitacionales", -36_497), ("Área construida = 0", -11),
          ("Área < 10 m² (parqueaderos, depósitos)", -1_044), ("Registros agregados (edificio en una fila)", -234),
          ("Sin estrato residencial válido", -12_093), ("Viviendas analizadas", 332_718)]


def _estilo(fig, alto=420, leyenda=True):
    """Estilo común: fondo claro, rejilla tenue, tipografía grande."""
    fig.update_layout(template="simple_white", height=alto, margin=dict(l=60, r=20, t=95 if leyenda else 60, b=50),
                      font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=15, color=TINTA),
                      title=dict(y=0.985, yanchor="top", x=0.01, xanchor="left", font=dict(size=18)),
                      hoverlabel=dict(font_size=14), showlegend=leyenda,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title_text="",
                                  traceorder="normal"),
                      paper_bgcolor="white", plot_bgcolor="white")
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7", tickfont_color=TINTA_2)
    fig.update_yaxes(showgrid=True, gridcolor=REJILLA, linecolor="#c3c2b7", tickfont_color=TINTA_2)
    return fig


# ---------------------------------------------------------------------------
# Carga (con caché: la app lee los archivos una sola vez)
# ---------------------------------------------------------------------------
def viviendas():
    return _leer(_ESTADO['v'], "viviendas.pkl.gz")


def predicciones():
    return _leer(_ESTADO['v'], "test_predicciones.pkl.gz")


def resultados():
    return _leer(_ESTADO['v'], "resultados.json")


def diagnosticos():
    return _leer(_ESTADO['v'], "diagnosticos.json")


def coeficientes():
    return _leer(_ESTADO['v'], "coeficientes.pkl.gz")


def base_eda(particion="train"):
    """Datos para el EDA: por defecto SOLO train, como en el Entregable 1."""
    v = viviendas()
    return v if particion == "todas" else v[v.particion == particion]


MODELOS = list(COLOR_MODELO)


class _NombresCortos(dict):
    """Nombre corto de cada modelo; el de la moda por zona lleva el tamaño de zona elegido por CV."""

    def __getitem__(self, modelo):
        if modelo == "Moda por zona":
            return f"Moda por zona ({float(resultados().get('tam_zona_moda_km', 2.0)):g} km)"
        return dict.__getitem__(self, modelo)


NOMBRE_CORTO = _NombresCortos(NOMBRE_CORTO)

# Modelos avanzados (entrenar_avanzados.py). Colores fijos por modelo, distintos de los del modelo base.
COLOR_MODELO.update({"XGBoost": "#e87ba4", "XGBoost geográfico": "#4a3aa7", "Rotation Forest": "#eda100",
                     "SVM RBF (Nyström) + TPE": "#008300"})
NOMBRE_CORTO.update({"XGBoost": "XGBoost", "XGBoost geográfico": "XGBoost geográfico",
                     "Rotation Forest": "Rotation Forest", "SVM RBF (Nyström) + TPE": "SVM RBF + TPE"})
AVANZADOS = ["XGBoost", "XGBoost geográfico", "Rotation Forest", "SVM RBF (Nyström) + TPE"]


def hay_avanzados():
    return False   # los modelos avanzados son del Entregable 2


def modelos_disponibles():
    return MODELOS + (AVANZADOS if hay_avanzados() else [])


# ---------------------------------------------------------------------------
# Pestaña 1: contexto
# ---------------------------------------------------------------------------
def kpis():
    v = viviendas()
    r = resultados()
    return {"viviendas": len(v), "edificios": int(v.edificio_id.nunique()),
            "sin_coord": float((~v.tiene_coordenadas).mean()), "moran": diagnosticos()["moran_estrato"]["I"],
            "f1_test": r["resultados_test"]["f1_macro"]["Logística B_fisicas+ubicacion"]}


def fig_embudo():
    etiquetas = [e for e, _ in EMBUDO]
    valores = [v for _, v in EMBUDO]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=etiquetas, x=valores,
        measure=["absolute"] + ["relative"] * 5 + ["total"],
        text=[f"{v:+,}".replace(",", " ") if 0 < i < 6 else f"{v:,}".replace(",", " ") for i, v in enumerate(valores)],
        textposition="outside", connector=dict(line=dict(color="#c3c2b7", width=1)),
        decreasing=dict(marker_color="#e98a89"), totals=dict(marker_color="#2a78d6"),
        increasing=dict(marker_color="#2a78d6"),
        hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, 470_000], title="filas")
    fig.update_layout(title="Embudo de limpieza: de 382 597 filas a 332 718 viviendas")
    return _estilo(fig, alto=430, leyenda=False)


def fig_distribucion_estrato(particion="todas"):
    v = base_eda(particion)
    c = v.estrato_num.value_counts().reindex(CLASES, fill_value=0)
    pct = c / c.sum()
    fig = go.Figure(go.Bar(x=[f"Estrato {k}" for k in CLASES], y=pct.values,
                           marker_color=[COLOR_ESTRATO[k] for k in CLASES],
                           text=[f"{p:.1%}" for p in pct.values], textposition="outside",
                           customdata=c.values, hovertemplate="%{x}: %{y:.1%} (%{customdata:,} viviendas)<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", title="proporción de viviendas", range=[0, pct.max() * 1.18])
    fig.update_layout(title=f"Distribución del estrato (razón mayor/menor = {c.max() / c.min():.1f}:1)")
    return _estilo(fig, alto=400, leyenda=False)


# ---------------------------------------------------------------------------
# Pestaña 2: EDA
# ---------------------------------------------------------------------------
def tabla_resumen_numericas(particion="train"):
    v = base_eda(particion)
    filas = []
    for c, et in NUMERICAS.items():
        x = v[c]
        filas.append({"Variable": et, "Media": x.mean(), "Mediana": x.median(), "Desv. est.": x.std(),
                      "P1": x.quantile(0.01), "P99": x.quantile(0.99), "Asimetría": x.skew(),
                      "Curtosis": x.kurt(), "% faltante": 100 * x.isna().mean()})
    return pd.DataFrame(filas)


def efecto_numerica(var, particion="train"):
    """η² de Kruskal-Wallis: fracción de la variación (en rangos) explicada por el estrato."""
    v = base_eda(particion)[[var, "estrato_num"]].dropna()
    grupos = [g[var].to_numpy() for _, g in v.groupby("estrato_num")]
    h, p = stats.kruskal(*grupos)
    k, n = len(grupos), len(v)
    return {"H": float(h), "p": float(p), "eta2_H": float(max(0.0, (h - k + 1) / (n - k))), "n": n}


def fig_numerica(var, log=True, particion="train"):
    v = base_eda(particion)[[var, "estrato_num"]].dropna()
    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58], horizontal_spacing=0.09,
                        subplot_titles=("Distribución", "Por estrato"))
    x = np.log1p(v[var]) if log else v[var]
    fig.add_trace(go.Histogram(x=x, nbinsx=60, marker_color="#3987e5", marker_line_color="white",
                               marker_line_width=1, hovertemplate="%{x}: %{y:,}<extra></extra>"), 1, 1)
    for k in CLASES:
        xs = v.loc[v.estrato_num == k, var]
        fig.add_trace(go.Box(y=np.log1p(xs) if log else xs, name=f"E{k}", marker_color=COLOR_ESTRATO[k],
                             boxpoints=False, line_width=1.5), 1, 2)
    eje = f"log(1 + {NUMERICAS[var]})" if log else NUMERICAS[var]
    fig.update_xaxes(title_text=eje, row=1, col=1)
    fig.update_yaxes(title_text="viviendas", row=1, col=1)
    fig.update_yaxes(title_text=eje, row=1, col=2)
    ef = efecto_numerica(var, particion)
    fig.update_layout(title=f"{NUMERICAS[var]} — η² (Kruskal-Wallis) = {ef['eta2_H']:.3f}")
    return _estilo(fig, alto=430, leyenda=False)


def cramer_v(tabla):
    chi2 = stats.chi2_contingency(tabla, correction=False)[0]
    n = tabla.to_numpy().sum()
    r, k = tabla.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))), float(chi2)


def fig_categorica(var, particion="train"):
    v = base_eda(particion)
    t = pd.crosstab(v[var].astype(str), v.estrato_num)
    t = t.loc[t.sum(1).sort_values(ascending=False).index]
    prop = t.div(t.sum(1), axis=0)
    V, _ = cramer_v(t)
    fig = go.Figure()
    for k in CLASES:
        fig.add_trace(go.Bar(y=prop.index, x=prop[k], name=f"Estrato {k}", orientation="h",
                             marker_color=COLOR_ESTRATO[k], marker_line_color="white", marker_line_width=2,
                             customdata=t[k], hovertemplate="%{y}<br>Estrato " + str(k) +
                             ": %{x:.1%} (%{customdata:,})<extra></extra>"))
    etiquetas = [f"{i}  (n={n:,})".replace(",", " ") for i, n in zip(prop.index, t.sum(1))]
    fig.update_yaxes(autorange="reversed", tickvals=list(prop.index), ticktext=etiquetas, showgrid=False)
    fig.update_xaxes(tickformat=".0%", title="proporción dentro de la categoría")
    fig.update_layout(barmode="stack", title=f"{CATEGORICAS[var]} vs estrato — V de Cramér = {V:.2f}")
    return _estilo(fig, alto=120 + 55 * len(prop))


def fig_correlacion(particion="train"):
    v = base_eda(particion)
    cols = list(NUMERICAS) + ["estrato_num"]
    m = v[cols].sample(min(60_000, len(v)), random_state=42).corr(method="spearman")
    et = [NUMERICAS.get(c, "Estrato") for c in cols]
    fig = go.Figure(go.Heatmap(z=m.values, x=et, y=et, zmin=-1, zmax=1, colorscale=DIVERGENTE,
                               text=np.round(m.values, 2), texttemplate="%{text}", textfont_size=12,
                               hovertemplate="%{y} vs %{x}: ρ = %{z:.2f}<extra></extra>",
                               colorbar=dict(title="ρ")))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(title="Correlación de Spearman (muestra de 60 000 viviendas de train)")
    return _estilo(fig, alto=560, leyenda=False)


def fig_faltantes(particion="train"):
    v = base_eda(particion)
    cols = list(NUMERICAS) + list(CATEGORICAS)
    pct = (v[cols].isna().mean() * 100).sort_values()
    fig = go.Figure(go.Bar(x=pct.values, y=[NUMERICAS.get(c, CATEGORICAS.get(c)) for c in pct.index],
                           orientation="h", marker_color="#3987e5", text=[f"{p:.2f} %" for p in pct.values],
                           textposition="outside", hovertemplate="%{y}: %{x:.3f} %<extra></extra>"))
    fig.update_xaxes(title="% de viviendas sin dato", range=[0, max(pct.max() * 1.3, 0.5)])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="Valores faltantes por variable")
    return _estilo(fig, alto=430, leyenda=False)


def _grilla(df, col, celda=0.005, min_n=5):
    g = df.dropna(subset=["lat", "lon"]).assign(glat=lambda d: (d.lat / celda).round() * celda,
                                                 glon=lambda d: (d.lon / celda).round() * celda)
    a = g.groupby(["glat", "glon"]).agg(valor=(col, "mean"), n=(col, "size")).reset_index()
    return a[a.n >= min_n]


def fig_mapa(var="estrato_num", particion="todas"):
    v = base_eda(particion)
    a = _grilla(v, var)
    titulo = "Estrato medio" if var == "estrato_num" else f"{NUMERICAS[var]} (media)"
    if var == "estrato_num":
        rango, escala = (1, 6), SECUENCIAL
    else:
        rango, escala = tuple(a.valor.quantile([0.02, 0.98])), SECUENCIAL
    fig = go.Figure(go.Scattermap(
        lat=a.glat, lon=a.glon, mode="markers",
        marker=dict(size=np.clip(np.sqrt(a.n) * 1.1, 5, 16), color=a.valor, colorscale=escala,
                    cmin=rango[0], cmax=rango[1], opacity=0.85, colorbar=dict(title=titulo.split(" (")[0])),
        customdata=np.c_[a.valor, a.n],
        hovertemplate=titulo + ": %{customdata[0]:.2f}<br>viviendas en la celda: %{customdata[1]:,}<extra></extra>"))
    fig.update_layout(map=dict(style=ESTILO_MAPA, center=dict(lat=10.97, lon=-74.81), zoom=11.3),
                      title=f"{titulo} por celda de ~550 m", height=600, margin=dict(l=0, r=0, t=50, b=0),
                      font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=15))
    return fig


def fig_gradiente_norte_sur(particion="todas"):
    v = base_eda(particion).dropna(subset=["lat"])
    v = v.assign(franja=pd.qcut(v.lat, 10, labels=[f"F{i}" for i in range(1, 11)]))
    t = pd.crosstab(v.franja, v.estrato_num, normalize="index")
    fig = go.Figure()
    for k in CLASES:
        fig.add_trace(go.Bar(x=t.index.astype(str), y=t[k], name=f"Estrato {k}", marker_color=COLOR_ESTRATO[k],
                             marker_line_color="white", marker_line_width=2,
                             hovertemplate="%{x}: %{y:.1%}<extra>Estrato " + str(k) + "</extra>"))
    media = v.groupby("franja", observed=True).estrato_num.mean()
    fig.update_layout(barmode="stack",
                      title=f"Composición por franjas de latitud (sur → norte): estrato medio de {media.iloc[0]:.1f} "
                            f"a {media.max():.1f} (franja {media.idxmax()}), {media.iloc[-1]:.1f} en la más al norte")
    fig.update_yaxes(tickformat=".0%", title="proporción")
    fig.update_xaxes(title="franja (F1 = más al sur, F10 = más al norte; 10 % de viviendas cada una)")
    return _estilo(fig, alto=430)


def fig_correlograma():
    d = diagnosticos()["correlograma"]
    fig = go.Figure(go.Scatter(x=d["centro_km"], y=d["moran_I"], mode="lines+markers",
                               line=dict(color="#2a78d6", width=2), marker=dict(size=8),
                               hovertemplate="%{x:.2f} km: I = %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.3, line_dash="dot", line_color="#8a8984",
                  annotation_text="I = 0.3 (umbral de dependencia débil)", annotation_position="top right")
    fig.add_vline(x=1.0, line_dash="dash", line_color="#eb6834",
                  annotation_text="buffer elegido: 1 km", annotation_position="bottom right")
    fig.update_xaxes(title="distancia entre edificios (km)")
    fig.update_yaxes(title="I de Moran del estrato")
    fig.update_layout(title=f"Correlograma: ¿hasta dónde se parecen los vecinos? (Moran global = {diagnosticos()['moran_estrato']['I']:.3f})")
    return _estilo(fig, alto=420, leyenda=False)


def fig_tamano_edificios():
    v = viviendas()
    t = v.groupby("edificio_id").size()
    bins = [0, 1, 5, 20, 100, 500, 4000]
    et = ["1 (casa)", "2–5", "6–20", "21–100", "101–500", "> 500"]
    c = pd.cut(t, bins, labels=et)
    edif = c.value_counts().reindex(et) / len(t)
    unid = t.groupby(c, observed=False).sum().reindex(et) / t.sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=et, y=edif.values, name="% de edificios", marker_color="#86b6ef",
                         hovertemplate="%{x}: %{y:.1%} de los edificios<extra></extra>"))
    fig.add_trace(go.Bar(x=et, y=unid.values, name="% de viviendas", marker_color="#1c5cab",
                         hovertemplate="%{x}: %{y:.1%} de las viviendas<extra></extra>"))
    fig.update_layout(barmode="group", title=f"Unidades por edificio: mediana {int(t.median())}, máximo {t.max():,}".replace(",", " ").replace("  ", ", "))
    fig.update_yaxes(tickformat=".0%", title="proporción")
    fig.update_xaxes(title="viviendas en el edificio")
    return _estilo(fig, alto=420)


# ---------------------------------------------------------------------------
# Pestaña 3: modelos base
# ---------------------------------------------------------------------------
def tabla_metricas():
    r = resultados()
    t = pd.DataFrame(r["resultados_test"]).loc[MODELOS]
    cv = pd.DataFrame(r["tabla_cv"]).loc[MODELOS]
    out = pd.DataFrame({
        "Modelo": [NOMBRE_CORTO[m] for m in MODELOS],
        "F1 macro CV (± desv.)": [f"{a:.3f} ± {b:.3f}" for a, b in zip(cv["f1_macro (media)"], cv["F1 macro (desv. entre folds)"])],
        "F1 macro test": t["f1_macro"].values, "Accuracy": t["accuracy"].values,
        "Acc. balanceada": t["balanced_accuracy"].values, "MAE ordinal": t["MAE_ordinal"].values,
        "Acc. ±1": t["accuracy_±1"].values, "Kappa cuadr.": t["kappa_cuadrático"].values,
        "AUC OvR": t["AUC_OvR_macro"].values, "Log loss": t["log_loss"].values})
    return out


METRICAS_COMPARABLES = {"f1_macro": "F1 macro", "accuracy": "Accuracy", "balanced_accuracy": "Accuracy balanceada",
                        "MAE_ordinal": "MAE ordinal (menor es mejor)", "accuracy_±1": "Accuracy ±1 estrato",
                        "kappa_cuadrático": "Kappa cuadrático", "AUC_OvR_macro": "AUC OvR macro"}


def fig_comparacion(metrica="f1_macro"):
    r = resultados()
    t = r["resultados_test"][metrica]
    fig = go.Figure()
    vals = [t[m] for m in MODELOS]
    error = None
    if metrica == "f1_macro":
        cv = r["tabla_cv"]
        fig.add_trace(go.Bar(x=[NOMBRE_CORTO[m] for m in MODELOS], y=[cv["f1_macro (media)"][m] for m in MODELOS],
                             name="CV espacial (media ± desv.)", marker_color="#e6e5e1",
                             marker_pattern=dict(shape="/", fgcolor="#8a8984", size=7), marker_line_color="#8a8984",
                             marker_line_width=1,
                             error_y=dict(type="data", array=[cv["F1 macro (desv. entre folds)"][m] for m in MODELOS],
                                          color="#52514e", thickness=1.5),
                             hovertemplate="%{x}<br>CV: %{y:.3f}<extra></extra>"))
        ic = r["ic_modelo_principal"]
        principal = MODELOS.index("Logística B_fisicas+ubicacion")
        error = dict(type="data", symmetric=False,
                     array=[0] * principal + [ic["IC95_sup"]["f1_macro"] - vals[principal]] + [0],
                     arrayminus=[0] * principal + [vals[principal] - ic["IC95_inf"]["f1_macro"]] + [0],
                     color="#0b0b0b", thickness=1.5)
    fig.add_trace(go.Bar(x=[NOMBRE_CORTO[m] for m in MODELOS], y=vals, name="Test (7 bloques no vistos)",
                         marker_color=[COLOR_MODELO[m] for m in MODELOS], error_y=error,
                         text=[f"{x:.3f}" for x in vals], textposition="outside",
                         hovertemplate="%{x}<br>test: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group", title=f"{METRICAS_COMPARABLES[metrica]}: modelos frente a líneas base")
    fig.update_yaxes(title=METRICAS_COMPARABLES[metrica], rangemode="tozero")
    return _estilo(fig, alto=460)


def fig_optimismo():
    r = resultados()
    o = r["optimismo_validacion_aleatoria"]
    sens = diagnosticos()["sensibilidad_buffer"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Validación aleatoria vs espacial", "Efecto del buffer"),
                        horizontal_spacing=0.12)
    nombres = ["Aleatoria (filas mezcladas)", "Espacial (bloques + buffer)"]
    claves = ["aleatoria (StratifiedKFold)", "espacial (bloques + buffer)"]
    fig.add_trace(go.Bar(x=nombres, y=[o["F1 macro CV"][k] for k in claves], marker_color=["#e34948", "#2a78d6"],
                         error_y=dict(array=[o["desv."][k] for k in claves], color="#52514e"),
                         text=[f"{o['F1 macro CV'][k]:.2f}" for k in claves], textposition="outside",
                         hovertemplate="%{x}: %{y:.3f}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=[s["buffer_km"] for s in sens], y=[s["f1"] for s in sens], mode="lines+markers",
                             line=dict(color="#2a78d6", width=2), marker=dict(size=9),
                             error_y=dict(array=[s["desv"] for s in sens], color="#86b6ef"),
                             hovertemplate="buffer %{x} km: F1 %{y:.3f}<extra></extra>"), 1, 2)
    fig.update_yaxes(title_text="F1 macro (CV)", row=1, col=1, range=[0, 0.75])
    fig.update_xaxes(title_text="buffer (km)", row=1, col=2)
    fig.update_layout(title="La validación aleatoria duplica el F1: la estructura espacial no se puede ignorar")
    return _estilo(fig, alto=430, leyenda=False)


def _pred(modelo):
    p = predicciones()
    proba = p[[f"p{k}|{modelo}" for k in CLASES]].to_numpy()
    return p["estrato"].to_numpy(), p[f"pred|{modelo}"].to_numpy(), proba


def fig_confusion(modelo="Logística B_fisicas+ubicacion", normalizar=True):
    y, pr, _ = _pred(modelo)
    m = confusion_matrix(y, pr, labels=CLASES)
    z = m / m.sum(1, keepdims=True) if normalizar else m
    texto = np.vectorize(lambda a: f"{a:.0%}")(z) if normalizar else m
    fig = go.Figure(go.Heatmap(z=z, x=[f"E{k}" for k in CLASES], y=[f"E{k}" for k in CLASES],
                               colorscale=SECUENCIAL, text=texto, texttemplate="%{text}", textfont_size=14,
                               customdata=m, hovertemplate="real %{y} → predicho %{x}: %{customdata:,} viviendas<extra></extra>",
                               colorbar=dict(title="% de la fila" if normalizar else "viviendas")))
    fig.update_yaxes(autorange="reversed", title="estrato real", showgrid=False)
    fig.update_xaxes(title="estrato predicho", side="bottom")
    fig.update_layout(title=f"Matriz de confusión — {NOMBRE_CORTO[modelo]}")
    return _estilo(fig, alto=480, leyenda=False)


def tabla_por_clase(modelo="Logística B_fisicas+ubicacion"):
    y, pr, _ = _pred(modelo)
    p, r, f, s = precision_recall_fscore_support(y, pr, labels=CLASES, zero_division=0)
    return pd.DataFrame({"Estrato": CLASES, "Precisión": p, "Recall": r, "F1": f, "Viviendas en test": s})


def fig_por_clase(modelo="Logística B_fisicas+ubicacion"):
    t = tabla_por_clase(modelo)
    fig = go.Figure()
    for col, color in [("Precisión", "#86b6ef"), ("Recall", "#2a78d6"), ("F1", "#104281")]:
        fig.add_trace(go.Bar(x=[f"E{k}" for k in t.Estrato], y=t[col], name=col, marker_color=color,
                             hovertemplate="%{x}: " + col + " %{y:.2f}<extra></extra>"))
    fig.update_layout(barmode="group", title=f"Desempeño por estrato — {NOMBRE_CORTO[modelo]}")
    fig.update_yaxes(range=[0, 1], title="valor")
    return _estilo(fig, alto=420)


def fig_roc(modelo="Logística B_fisicas+ubicacion"):
    y, _, proba = _pred(modelo)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#c3c2b7", dash="dot"),
                             showlegend=False, hoverinfo="skip"))
    for j, k in enumerate(CLASES):
        if (y == k).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y == k, proba[:, j])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=COLOR_ESTRATO[k], width=2),
                                 name=f"E{k} (AUC {auc(fpr, tpr):.2f})",
                                 hovertemplate=f"E{k}: FPR %{{x:.2f}}, TPR %{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title="tasa de falsos positivos", range=[0, 1])
    fig.update_yaxes(title="tasa de verdaderos positivos", range=[0, 1.02])
    fig.update_layout(title=f"Curvas ROC uno-contra-resto — {NOMBRE_CORTO[modelo]}")
    return _estilo(fig, alto=460)


def fig_calibracion(modelo="Logística B_fisicas+ubicacion"):
    y, _, proba = _pred(modelo)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#c3c2b7", dash="dot"),
                             name="calibración perfecta", hoverinfo="skip"))
    for j, k in enumerate(CLASES):
        if (y == k).sum() < 50:
            continue
        fr, mp = calibration_curve(y == k, proba[:, j], n_bins=10, strategy="quantile")
        fig.add_trace(go.Scatter(x=mp, y=fr, mode="lines+markers", name=f"E{k}", marker=dict(size=8),
                                 line=dict(color=COLOR_ESTRATO[k], width=2),
                                 hovertemplate=f"E{k}: predicho %{{x:.2f}}, observado %{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title="probabilidad predicha", range=[0, 1])
    fig.update_yaxes(title="frecuencia observada", range=[0, 1])
    fig.update_layout(title=f"Calibración por estrato — {NOMBRE_CORTO[modelo]}")
    return _estilo(fig, alto=460)


TIPO_VIVIENDA = {"tv_1": "VIS", "tv_2": "VIP", "tv_3": "no VIS", "tv_4": "no aplica"}


def _nombre_variable(f):
    """Nombre legible de una columna del Pipeline (numérica, indicador de faltante u one-hot)."""
    f = f.split("__", 1)[-1]
    nums = {**NUMERICAS, "x_km": "Posición este-oeste (x)", "y_km": "Posición norte-sur (y)"}
    if f in nums:
        return nums[f]
    if f.startswith("missingindicator_"):
        return f"{nums.get(f[17:], f[17:])}: dato faltante"
    for c, et in [("condicion_predio_", "Régimen: "), ("uso_", "Uso: "), ("tipo_vivienda_", "Tipo de vivienda: ")]:
        if f.startswith(c):
            v = f[len(c):]
            v = TIPO_VIVIENDA.get(v, v)
            return et + ("otras (infrecuentes)" if v == "infrequent_sklearn" else v.replace("_", " "))
    return f.replace("_", " ")


def fig_coeficientes(conjunto="B_fisicas+ubicacion", top=15):
    c = coeficientes()
    c = c[c.modelo == conjunto].pivot(index="variable", columns="estrato", values="coef")
    orden = (c[6] - c[1]).abs().sort_values(ascending=False).index[:top]
    c = c.loc[orden]
    lim = float(np.abs(c.values).max())
    fig = go.Figure(go.Heatmap(z=c.values, x=[f"E{k}" for k in c.columns], y=[_nombre_variable(v) for v in c.index],
                               colorscale=DIVERGENTE, zmin=-lim, zmax=lim, text=np.round(c.values, 2),
                               texttemplate="%{text}", textfont_size=12,
                               hovertemplate="%{y} → %{x}: β = %{z:.3f}<extra></extra>", colorbar=dict(title="β")))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    nombre = "Logística B (físicas + ubicación)" if conjunto.startswith("B") else "Logística A (físicas)"
    fig.update_layout(title=f"Coeficientes por estrato — {nombre} ({top} variables con mayor |β6 − β1|)")
    return _estilo(fig, alto=170 + 32 * top, leyenda=False)


def fig_curva_aprendizaje():
    d = diagnosticos()["curva_aprendizaje"]
    n = np.array(d["n"])
    fig = go.Figure()
    for clave, nombre, color in [("train", "entrenamiento", "#e34948"), ("val", "validación espacial", "#2a78d6")]:
        m, s = np.array(d[f"{clave}_media"]), np.array(d[f"{clave}_desv"])
        fig.add_trace(go.Scatter(x=np.r_[n, n[::-1]], y=np.r_[m + s, (m - s)[::-1]], fill="toself",
                                 fillcolor=color, opacity=0.12, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=n, y=m, mode="lines+markers", name=nombre, line=dict(color=color, width=2),
                                 marker=dict(size=8), hovertemplate="n = %{x:,}: F1 %{y:.3f}<extra>" + nombre + "</extra>"))
    fig.update_xaxes(type="log", title="viviendas de entrenamiento (escala log)",
                     tickvals=[5_000, 10_000, 20_000, 50_000, 100_000], ticktext=["5 mil", "10 mil", "20 mil", "50 mil", "100 mil"])
    fig.update_yaxes(title="F1 macro", range=[0, 0.75])
    fig.update_layout(title="Curva de aprendizaje — logística B")
    return _estilo(fig, alto=430)


def fig_mapa_residuos():
    a = _leer(_ESTADO["v"], "residuos_mapa.pkl.gz")
    lim = float(a.residuo.abs().quantile(0.98))
    m = diagnosticos()["moran_residuos"]["B_fisicas+ubicacion"]
    fig = go.Figure(go.Scattermap(
        lat=a.lat, lon=a.lon, mode="markers",
        marker=dict(size=np.clip(np.sqrt(a.n) * 2, 5, 14), color=a.residuo, colorscale=DIVERGENTE,
                    cmin=-lim, cmax=lim, colorbar=dict(title="residuo")),
        customdata=np.c_[a.residuo, a.n],
        hovertemplate="residuo medio %{customdata[0]:+.2f}<br>viviendas: %{customdata[1]:,}<extra></extra>"))
    fig.update_layout(map=dict(style=ESTILO_MAPA, center=dict(lat=a.lat.mean(), lon=a.lon.mean()), zoom=11.8),
                      title=f"Residuo ordinal en test (I de Moran = {m['I']:.2f})",
                      height=560, margin=dict(l=0, r=0, t=50, b=0), font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=15))
    return fig


# ---------------------------------------------------------------------------
# Simulador
# ---------------------------------------------------------------------------
def modelo_principal():
    return _leer(_ESTADO["v"], "modelo_logistica_B.joblib")


def rangos_simulador():
    return _leer(_ESTADO["v"], "simulador_rangos.json")


def predecir(entrada):
    """entrada: dict con variables físicas, categóricas y lat/lon -> probabilidades por estrato.
    Las coordenadas se proyectan con la MISMA función del Entregable 1 (src/limpieza.py)."""
    from src import limpieza
    lat, lon = entrada.pop("lat"), entrada.pop("lon")
    x, y = limpieza.proyectar_km(np.array([lat]), np.array([lon]))
    fila = dict(entrada, x_km=float(x[0]), y_km=float(y[0]), dist_centro_km=float(np.hypot(x[0], y[0])))
    X = pd.DataFrame([fila])
    mdl = modelo_principal()
    return dict(zip(mdl.classes_.tolist(), mdl.predict_proba(X)[0].tolist()))


def fig_simulador(probas):
    fig = go.Figure(go.Bar(x=[f"E{k}" for k in CLASES], y=[probas[k] for k in CLASES],
                           marker_color=[COLOR_ESTRATO[k] for k in CLASES],
                           text=[f"{probas[k]:.0%}" for k in CLASES], textposition="outside",
                           hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.1], tickformat=".0%", title="probabilidad")
    k = max(probas, key=probas.get)
    esperado = sum(c * p for c, p in probas.items())
    fig.update_layout(title=f"Predicción: estrato {k} (valor esperado {esperado:.1f})")
    return _estilo(fig, alto=360, leyenda=False)


def entregable1():
    return _leer(_ESTADO["v"], "entregable1.json")


def fig_estratos_coordenadas():
    e = pd.DataFrame(entregable1()["estratos"])
    pt, pc = e.todas / e.todas.sum(), e.con_coord / e.con_coord.sum()
    x = [f"Estrato {k}" for k in e.estrato]
    fig = go.Figure([
        go.Bar(name=f"Todas las viviendas ({e.todas.sum():,})".replace(",", " "), x=x, y=pt, marker_color="#c3c2b7",
               text=[f"{p:.1%}" for p in pt], textposition="outside", customdata=e.todas,
               hovertemplate="%{x}: %{y:.1%} (%{customdata:,})<extra>todas</extra>"),
        go.Bar(name=f"Con coordenadas ({e.con_coord.sum():,})".replace(",", " "), x=x, y=pc, marker_color="#2a78d6",
               text=[f"{p:.1%}" for p in pc], textposition="outside", customdata=e.con_coord,
               hovertemplate="%{x}: %{y:.1%} (%{customdata:,})<extra>con coordenadas</extra>")])
    fig.update_yaxes(tickformat=".0%", title="proporción de viviendas", range=[0, 0.38])
    fig.update_layout(barmode="group", title="% por estrato: antes y después de la revisión de coordenadas",
                      legend=dict(orientation="h", y=1.08, x=0))
    return _estilo(fig, alto=430)


def fig_cobertura_coordenadas():
    e = pd.DataFrame(entregable1()["estratos"])
    cob = e.con_coord / e.todas
    fig = go.Figure(go.Bar(x=[f"Estrato {k}" for k in e.estrato], y=cob,
                           marker_color=[COLOR_ESTRATO[k] for k in e.estrato],
                           text=[f"{c:.1%}" for c in cob], textposition="outside",
                           customdata=e.todas - e.con_coord,
                           hovertemplate="%{x}: %{y:.1%} con coordenadas (%{customdata:,} sin)<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.12], title="viviendas con coordenadas")
    fig.update_layout(title="¿Qué parte de cada estrato conserva coordenadas?")
    return _estilo(fig, alto=430, leyenda=False)


def fig_viviendas_por_tamano_grupo():
    t = pd.DataFrame(entregable1()["viviendas_por_tamano"])
    p = t.viviendas / t.viviendas.sum()
    fig = go.Figure(go.Bar(x=t.tamano, y=p, marker_color="#1c5cab", text=[f"{v:.1%}" for v in p],
                           textposition="outside", customdata=t.viviendas,
                           hovertemplate="grupos de %{x} unidades: %{customdata:,} viviendas (%{y:.1%})<extra></extra>"))
    fig.update_xaxes(title="unidades en el grupo (predio matriz)")
    fig.update_yaxes(tickformat=".0%", title="proporción de viviendas", range=[0, p.max() * 1.2])
    fig.update_layout(title="¿En qué tamaño de grupo vive cada vivienda?")
    return _estilo(fig, alto=420, leyenda=False)


def fig_kish(icc, m_pond, n=332_718):
    """Tamaño efectivo n/deff en función del ICC, para el tamaño medio ponderado dado."""
    r = np.linspace(0, 1, 201)
    nef = n / (1 + (m_pond - 1) * r)
    deff = 1 + (m_pond - 1) * icc
    fig = go.Figure(go.Scatter(x=r, y=nef, mode="lines", line=dict(color="#2a78d6", width=3),
                               hovertemplate="ICC %{x:.2f}: n_ef ≈ %{y:,.0f}<extra></extra>"))
    fig.add_scatter(x=[icc], y=[n / deff], mode="markers", marker=dict(size=14, color="#e34948"), hoverinfo="skip")
    fig.add_annotation(x=icc, y=np.log10(n / deff), text=f"ICC {icc:.3f} → n_ef ≈ {n / deff:,.0f}".replace(",", " "),
                       ax=-110, ay=-70, showarrow=True, arrowcolor="#e34948", bgcolor="#ffffff",
                       bordercolor="#e34948", font=dict(size=14))
    marcas = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 332_718]
    fig.update_yaxes(type="log", title="tamaño efectivo (escala log)", tickvals=marcas,
                     ticktext=[f"{v:,}".replace(",", " ") for v in marcas], minor=dict(showgrid=False))
    fig.update_xaxes(title="ICC (parecido del estrato dentro del grupo)", range=[0, 1.02])
    fig.update_layout(title=f"Efecto de diseño de Kish con m̃ = {m_pond:.1f}: deff = {deff:.1f}")
    return _estilo(fig, alto=420, leyenda=False)
