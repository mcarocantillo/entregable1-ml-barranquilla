# -*- coding: utf-8 -*-
"""Estilo común de figuras (fuentes grandes para legibilidad en el libro).

Centralizar el estilo garantiza que todas las figuras del libro sean
coherentes (mismos tamaños de letra, resolución y colores por estrato), de
modo que el lector pueda comparar gráficos de distintos capítulos sin
reinterpretar la codificación visual.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as C

# Paleta secuencial para el estrato (ordinal: de claro/bajo a oscuro/alto)
# Se usa viridis porque es perceptualmente uniforme, legible en escala de
# grises y apta para daltonismo; un color fijo por estrato evita que el mismo
# estrato cambie de color entre figuras.
PALETA_ESTRATO = dict(zip(C.CLASES, sns.color_palette("viridis", len(C.CLASES))))


def estilo():
    """Aplica el tema global de seaborn/matplotlib usado en todo el libro.

    Se llama una vez al inicio de cada notebook; afecta a todas las figuras
    creadas después.

    Parámetros
    ----------
    (ninguno)

    Devuelve
    --------
    None
    """
    sns.set_theme(style="whitegrid", context="notebook")  # cuadrícula suave para leer valores
    mpl.rcParams.update({
        "figure.dpi": 110,   # resolución en pantalla (notebook / HTML)
        "savefig.dpi": 150,  # resolución mayor para los PNG guardados en figuras/
        # Tamaños de letra por encima del valor por defecto: las figuras se
        # leen reducidas dentro del libro y en el PDF del informe
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 15,
        "axes.titleweight": "bold",
    })


def guardar(fig, nombre):
    """Guarda una figura como PNG en la carpeta de figuras del libro.

    Parámetros
    ----------
    fig : matplotlib.figure.Figure
        Figura a guardar.
    nombre : str
        Nombre del archivo sin extensión.

    Devuelve
    --------
    pathlib.Path
        Ruta del PNG escrito (config.CARPETA_FIGURAS / "<nombre>.png").
    """
    ruta = C.CARPETA_FIGURAS / f"{nombre}.png"
    fig.savefig(ruta, bbox_inches="tight")  # recorta márgenes en blanco sobrantes
    return ruta


def ejes_mapa(ax, titulo=None):
    """Formatea unos ejes para mostrar un mapa en coordenadas proyectadas (km).

    Parámetros
    ----------
    ax : matplotlib.axes.Axes
        Ejes donde se dibujó el mapa (x_km, y_km).
    titulo : str, opcional
        Título de los ejes; si es None no se pone título.

    Devuelve
    --------
    None

    Por qué
    -------
    La relación de aspecto 1:1 hace que un km mida lo mismo en horizontal y
    en vertical; sin ella el mapa se deformaría y las distancias y los
    bloques espaciales se verían engañosos.
    """
    ax.set_aspect("equal")
    ax.set_xlabel("x (km, este)")
    ax.set_ylabel("y (km, norte)")
    if titulo:
        ax.set_title(titulo)
