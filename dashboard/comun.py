"""Componentes comunes del dashboard del Entregable 1 (recuadros, tarjetas, tablas)."""
import dash_bootstrap_components as dbc
from dash import dcc, html

AUTORES = "María Carolina Cantillo Orozco (200179105) · Juan Camilo Oñoro Araujo (200177329)"
GRAF = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


def fmt(n):
    return f"{n:,}".replace(",", " ")


def interpretacion(*parrafos, titulo="Interpretación"):
    """Recuadro de lectura que acompaña a cada figura."""
    return dbc.Alert([html.H6(titulo, className="fw-bold mb-2")] + [html.P(p, className="mb-2") for p in parrafos],
                     color="light", className="interpretacion")


def tarjeta_kpi(valor, etiqueta, nota=None):
    return dbc.Card(dbc.CardBody([html.Div(valor, className="kpi-valor"), html.Div(etiqueta, className="kpi-etiqueta"),
                                  html.Div(nota, className="kpi-nota") if nota else None]), className="h-100 kpi")


def grafico(fig, id_=None):
    props = {"figure": fig, "config": GRAF}
    return dcc.Graph(id=id_, **props) if id_ else dcc.Graph(**props)


def seccion(titulo, *hijos, sub=None):
    return html.Section([html.H3(titulo, className="mt-4 mb-1"), html.P(sub, className="text-muted") if sub else None,
                         *hijos], className="mb-4")


def tabla(df, decimales=3):
    df = df.copy()
    for c in df.select_dtypes("number"):
        df[c] = df[c].map(lambda x: f"{x:,.{decimales}f}".replace(",", " "))
    return dbc.Table.from_dataframe(df, striped=True, bordered=False, hover=True, size="sm", className="tabla")
