"""Dashboard del Entregable 1 — Estrato socioeconómico de las viviendas de Barranquilla.

Autores: María Carolina Cantillo Orozco (200179105) y
         Juan Camilo Oñoro Araujo (200177329)
Machine Learning — Maestría, Universidad del Norte. Profesor Lihki Rubio.

Muestra el Entregable 1 en sus dos versiones, con un selector:
- v1: la versión entregada (las viviendas sin coordenadas quedan fuera del modelo).
- v2: las viviendas sin coordenadas se ubican con el centroide de su manzana (capítulo 1.7 del libro v2).
Cuatro pestañas: contexto, base de datos y calidad, análisis exploratorio y modelo base. Las figuras salen de
``figuras.py`` y los datos de ``datos/v1`` y ``datos/v2``. Para Render: ``gunicorn app:server``.
"""
import dash_bootstrap_components as dbc
import numpy as np
from dash import Dash, Input, Output, dcc, html

import figuras as F
import paginas_v1
import paginas_v2
from comun import AUTORES, tabla

app = Dash(__name__, title="Estrato socioeconómico — Entregable 1", suppress_callback_exceptions=True)
server = app.server

LIBROS = {"v1": "https://mcarocantillo.github.io/entregable1-ml-barranquilla/",
          "v2": "https://mcarocantillo.github.io/entregable1-ml-barranquilla/v2/"}
PAGINAS = {"v1": paginas_v1, "v2": paginas_v2}
_CACHE = {}


def construir(version):
    """Las cuatro pestañas de una versión (se construyen una vez y se reutilizan)."""
    if version not in _CACHE:
        with F.usar(version):
            _CACHE[version] = PAGINAS[version].construir(LIBROS[version])
    return _CACHE[version]


NOTA_VERSION = {
    "v1": "Versión 1, la entregada: las viviendas sin coordenadas (16.2 %, casi todas informales) quedan fuera del modelo.",
    "v2": "Versión 2: las viviendas sin coordenadas se ubican con el centroide de su manzana (96.9 % de la base "
          "entra al modelo). Cambian la partición, el análisis y el modelo.",
}

app.layout = html.Div([
    html.Header(dbc.Container([
        html.H1("Estrato socioeconómico de las viviendas de Barranquilla", className="titulo"),
        html.Div(["Clasificación con datos catastrales — Entregable 1, Machine Learning (Maestría, Universidad del "
                  "Norte). Profesor Lihki Rubio. ", html.A(id="enlace-libro", children="Jupyter Book del Entregable 1",
                                                           href=LIBROS["v2"], target="_blank",
                                                           style={"color": "#fff", "textDecoration": "underline"})],
                 className="subtitulo"),
        html.Div(AUTORES, className="autores"),
        dbc.RadioItems(id="version", value="v2", inline=True, className="mt-2 selector-version",
                       options=[{"label": etiqueta, "value": v} for v, etiqueta in F.VERSIONES.items()],
                       inputStyle={"marginRight": "6px"}, labelStyle={"marginRight": "18px", "color": "#fff"}),
    ], fluid=True), className="cabecera"),
    dbc.Container([
        dbc.Alert(id="nota-version", color="info", className="mt-3 mb-0"),
        dcc.Loading(html.Div(id="contenido"), type="default"),
    ], fluid=True),
    html.Footer(dbc.Container(html.Small(
        "Datos: catastro abierto de la Alcaldía de Barranquilla (descarga del 15-sep-2026). Coordenadas redondeadas a "
        "~110 m y sin número predial. Semilla 42. " + AUTORES), fluid=True), className="pie"),
])


@app.callback(Output("contenido", "children"), Output("nota-version", "children"), Output("enlace-libro", "href"),
              Input("version", "value"))
def mostrar_version(version):
    p = construir(version)
    pestanas = dbc.Tabs([
        dbc.Tab(p["t1"], label="1. Contexto del problema", tab_id="t1"),
        dbc.Tab(p["t2"], label="2. Base de datos y calidad", tab_id="t2"),
        dbc.Tab(p["t3"], label="3. Análisis exploratorio", tab_id="t3"),
        dbc.Tab(p["t4"], label="4. Modelo base", tab_id="t4"),
    ], active_tab="t1", className="mt-3")
    return pestanas, NOTA_VERSION[version], LIBROS[version]


# ===========================================================================
# Callbacks de los controles (todos reciben la versión activa)
# ===========================================================================
@app.callback(Output("eda-dist", "figure"), Output("eda-ns", "figure"), Output("eda-corr", "figure"),
              Output("eda-falt", "figure"), Output("eda-resumen", "children"),
              Input("eda-particion", "value"), Input("version", "value"))
def actualizar_eda_general(part, version):
    with F.usar(version):
        return (F.fig_distribucion_estrato(part), F.fig_gradiente_norte_sur(part), F.fig_correlacion(part),
                F.fig_faltantes(part), tabla(F.tabla_resumen_numericas(part), 2))


@app.callback(Output("eda-num-fig", "figure"), Output("eda-num-texto", "children"),
              Input("eda-num", "value"), Input("eda-log", "value"), Input("eda-particion", "value"),
              Input("version", "value"))
def actualizar_numerica(var, log, part, version):
    with F.usar(version):
        ef = F.efecto_numerica(var, part)
        fig = F.fig_numerica(var, "log" in (log or []), part)
    fuerza = "fuerte" if ef["eta2_H"] >= 0.14 else "moderada" if ef["eta2_H"] >= 0.06 else "débil"
    texto = html.P([f"Kruskal-Wallis: H = {ef['H']:,.0f}, p < 0.001 (n = {ef['n']:,}). ".replace(",", " "),
                    "Con tantas viviendas cualquier diferencia es significativa; lo que importa es el ",
                    html.B("tamaño del efecto"), f": η² = {ef['eta2_H']:.3f}, asociación {fuerza} con el estrato."],
                   className="text-muted")
    return fig, texto


@app.callback(Output("eda-cat-fig", "figure"), Input("eda-cat", "value"), Input("eda-particion", "value"),
              Input("version", "value"))
def actualizar_categorica(var, part, version):
    with F.usar(version):
        return F.fig_categorica(var, part)


@app.callback(Output("eda-mapa", "figure"), Input("eda-mapa-var", "value"), Input("eda-particion", "value"),
              Input("version", "value"))
def actualizar_mapa(var, part, version):
    with F.usar(version):
        return F.fig_mapa(var, part)


@app.callback(Output("mod-comp", "figure"), Input("mod-metrica", "value"), Input("version", "value"))
def actualizar_comparacion(metrica, version):
    with F.usar(version):
        return F.fig_comparacion(metrica)


@app.callback(Output("mod-conf", "figure"), Output("mod-clase", "figure"), Output("mod-roc", "figure"),
              Output("mod-cal", "figure"), Input("mod-modelo", "value"), Input("mod-normalizar", "value"),
              Input("version", "value"))
def actualizar_diagnostico(modelo, norm, version):
    with F.usar(version):
        return (F.fig_confusion(modelo, "si" in (norm or [])), F.fig_por_clase(modelo), F.fig_roc(modelo),
                F.fig_calibracion(modelo))


@app.callback(Output("mod-coef-fig", "figure"), Input("mod-coef", "value"), Input("version", "value"))
def actualizar_coeficientes(conj, version):
    with F.usar(version):
        return F.fig_coeficientes(conj)


NUM_SIM = ["area_construida", "area_catastral_terreno", "total_habitaciones", "total_banios", "total_plantas",
           "planta_ubicacion", "altura", "antiguedad"]
CAT_SIM = ["condicion_predio", "uso", "tipo_vivienda"]


@app.callback(Output("sim-fig", "figure"), Output("sim-mapa", "figure"),
              [Input(f"sim-{v}", "value") for v in NUM_SIM + CAT_SIM] + [Input("sim-lat", "value"),
                                                                         Input("sim-lon", "value"),
                                                                         Input("version", "value")])
def simular(*valores):
    version = valores[-1]
    valores = valores[:-1]
    entrada = dict(zip(NUM_SIM + CAT_SIM, valores[:-2]))
    for v in NUM_SIM:
        entrada[v] = np.nan if entrada[v] is None else float(entrada[v])
    lat, lon = valores[-2:]
    with F.usar(version):
        probas = F.predecir(dict(entrada, lat=lat, lon=lon))
        mapa = F.fig_mapa("estrato_num", "todas")
        fig = F.fig_simulador(probas)
    mapa.add_scattermap(lat=[lat], lon=[lon], mode="markers", marker=dict(size=18, color="#e34948"),
                        hovertemplate="vivienda simulada<extra></extra>", showlegend=False)
    mapa.update_layout(height=380, title="Ubicación simulada (punto rojo)", showlegend=False,
                       map=dict(center=dict(lat=lat, lon=lon), zoom=11))
    return fig, mapa


@app.callback(Output("kish-fig", "figure"), Input("kish-icc", "value"), Input("version", "value"))
def actualizar_kish(icc, version):
    gx = construir(version)["GX"]
    with F.usar(version):
        return F.fig_kish(gx["icc"] if icc is None else icc, gx["m_pond"])


# La versión por defecto se construye al arrancar, para que la primera visita no espere.
construir("v2")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
