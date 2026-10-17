"""Páginas del dashboard del Entregable 1 — versión 1 (la entregada).

Generado por construir_dashboard_e1.py a partir del dashboard del Entregable 2 (app.py original).
Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)
"""
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html

import figuras as F
from comun import GRAF, fmt, grafico, interpretacion, seccion, tabla, tarjeta_kpi


def construir(LIBRO_E1):
    """Construye las cuatro pestañas con los datos de la versión activa (F.usar)."""
    K = F.kpis()
    R = F.resultados()
    D = F.diagnosticos()
    IC = R["ic_modelo_principal"]

    # ===========================================================================
    # Pestaña 1 — Contexto del problema
    # ===========================================================================
    pasos = [("1", "Descarga", "Catastro abierto (ArcGIS REST), 15-sep-2026"),
             ("2", "Limpieza", "Reglas por fila, antes de partir"),
             ("3", "Partición espacial", "Bloques de 2 km; test reservado"),
             ("4", "EDA en train", "Uni, bi, multivariado y espacial"),
             ("5", "Pipeline", "Imputar, log, escalar, one-hot"),
             ("6", "CV espacial", "Bloques + buffer de 1 km"),
             ("7", "Test", "Una sola evaluación")]
    flujo = html.Div([html.Div([html.Div(n, className="paso-num"), html.Div(t, className="paso-titulo"),
                                html.Div(d, className="paso-desc")], className="paso") for n, t, d in pasos],
                     className="flujo")

    tab_contexto = dbc.Container([
        seccion("¿Se puede predecir el estrato de una vivienda con sus datos catastrales?",
                dbc.Row([
                    dbc.Col([
                        html.P(["En Colombia, el ", html.B("estrato socioeconómico"), " (1 a 6) clasifica los inmuebles "
                                "residenciales para cobrar tarifas diferenciadas de servicios públicos: los estratos 1 a 3 "
                                "reciben subsidios y los estratos 5 y 6 pagan una contribución (Ley 142 de 1994). Lo asigna "
                                "cada alcaldía por manzana, con una metodología que mira la vivienda y su entorno."]),
                        html.P(["Este proyecto plantea un problema de ", html.B("clasificación multiclase ordinal"), ": "
                                "predecir el estrato de cada unidad de vivienda de Barranquilla a partir de sus "
                                "características físicas en el catastro (área, baños, habitaciones, pisos, antigüedad, "
                                "régimen de propiedad) y de su ubicación. Un modelo así serviría para detectar viviendas "
                                "con un estrato inconsistente con sus características, apoyar la actualización de la "
                                "estratificación y estimarlo donde falta."]),
                        html.P(["Este entregable fija la ", html.B("referencia"), ": un modelo base lineal "
                                "(regresión logística multinomial) comparado con líneas base triviales y con una regla "
                                "puramente geográfica, evaluado con un protocolo que respeta la estructura espacial de los "
                                "datos. Los modelos más flexibles quedan para el Entregable 2."]),
                    ], md=7),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H5("Ficha del problema", className="fw-bold"),
                        html.Ul([
                            html.Li([html.B("Objetivo: "), "estrato (1–6), ordinal y desbalanceado"]),
                            html.Li([html.B("Unidad de observación: "), "unidad de vivienda del catastro"]),
                            html.Li([html.B("Tipo de datos: "), "transversal con componente espacial (foto del catastro)"]),
                            html.Li([html.B("Fuente: "), "datos abiertos de catastro, Alcaldía de Barranquilla"]),
                            html.Li([html.B("Ruta del enunciado: "), "A (clasificación) + componente espacial"]),
                            html.Li([html.B("Métrica principal: "), "F1 macro (todas las clases pesan igual)"]),
                        ], className="mb-0"),
                    ])), md=5),
                ])),
        dbc.Row([
            dbc.Col(tarjeta_kpi(fmt(K["viviendas"]), "viviendas analizadas"), md=3),
            dbc.Col(tarjeta_kpi(fmt(K["edificios"]), "predios matriz", "edificios, conjuntos o casas"), md=3),
            dbc.Col(tarjeta_kpi(f"{K['moran']:.2f}", "I de Moran del estrato", "autocorrelación espacial muy fuerte"), md=3),
            dbc.Col(tarjeta_kpi(f"{K['f1_test']:.2f}", "F1 macro del modelo en test",
                                f"IC 95 %: [{IC['IC95_inf']['f1_macro']:.2f}, {IC['IC95_sup']['f1_macro']:.2f}]"), md=3),
        ], className="g-3 my-2"),
        seccion("Cómo se trabajó", flujo,
                interpretacion("El orden importa: las reglas que miran una sola fila (por ejemplo, descartar áreas "
                               "imposibles) se aplican antes de partir los datos; todo lo que se aprende de los datos "
                               "(medianas, cuantiles, escalas) se calcula solo con entrenamiento y dentro de un Pipeline. "
                               "El conjunto de prueba son zonas completas de la ciudad que el modelo nunca vio.",
                               titulo="Por qué este orden")),
        dbc.Row([
            dbc.Col(grafico(F.fig_embudo()), md=7),
            dbc.Col(grafico(F.fig_distribucion_estrato()), md=5),
        ]),
        interpretacion(
            "La limpieza excluye usos no habitacionales, unidades de menos de 10 m² (parqueaderos o depósitos), registros "
            "que describen un edificio entero en una fila y filas sin estrato residencial. Quedan 332 718 viviendas "
            "(87 % de lo descargado).",
            "El estrato está desbalanceado: el estrato 1 es la clase más frecuente y el 6 la más escasa. Por eso la métrica "
            "principal es el F1 macro y no la accuracy, que premiaría predecir solo las clases grandes."),
    ], fluid=True)

    # ===========================================================================
    # Pestaña 2 — Base de datos y calidad (capítulo 1 del Entregable 1)
    # ===========================================================================
    E1 = F.entregable1()
    GX, G0, MJ = E1["grupo"]["con_excepcion"], E1["grupo"]["sin_excepcion"], E1["mejoras"]
    estr = {e["estrato"]: e for e in E1["estratos"]}


    def pct(a, b):
        return f"{100 * a / b:.1f} %"


    unidades = [("382 597", "filas descargadas", "catastro abierto, 15-sep-2026"),
                (fmt(E1["viviendas"]), "unidades de vivienda", "cada fila: una casa o un apartamento"),
                (fmt(E1["predios"]), "predios", "números prediales (NPN) de 30 dígitos distintos"),
                (fmt(E1["predios_matriz"]), "predios matriz", "edificio, conjunto o casa: 22 primeros dígitos del NPN"),
                (fmt(GX["grupos"]), "grupos para el ICC", "predios matriz, separando las mejoras en terreno ajeno")]

    tabla_imp = [{"Regla": r["variable"], "Rango o condición": r["rango plausible"], "Filas": fmt(int(r["n imposibles"])),
                  "% de filas": f'{float(r["%"]):.3f} %'} for r in E1["tabla_imposibles"]]

    tab_datos = dbc.Container([
        dbc.Alert(["Esta pestaña resume el ", html.B("capítulo 1 del Entregable 1"), " (base de datos y calidad). El "
                   "detalle, con el código, está en el ", html.A("Jupyter Book del Entregable 1", href=LIBRO_E1,
                                                                   target="_blank"), "."],
                  color="info", className="mt-3"),

        seccion("1. ¿Qué se cuenta? Viviendas, predios y predios matriz",
                dbc.Row([dbc.Col(tarjeta_kpi(v, e, n), className="mb-2") for v, e, n in unidades], className="g-3"),
                interpretacion(
                    "Las mismas viviendas se pueden contar de tres maneras. Una unidad de vivienda es una fila. Un predio es "
                    "un número predial nacional (NPN) de 30 dígitos: hay algo menos de predios que de viviendas porque unas "
                    "pocas viviendas comparten número. Un predio matriz es el lote o edificio al que pertenecen: los "
                    "22 primeros dígitos del NPN (los dígitos 23–24 identifican la torre y los siguientes el apartamento).",
                    "Ejemplo: un conjunto con 200 apartamentos son 200 viviendas y 200 predios, pero un solo predio matriz. "
                    "Una casa sola es 1 vivienda, 1 predio y 1 predio matriz.",
                    f"Excepción, condición de propiedad 5 («mejoras en terreno ajeno»): son construcciones levantadas sobre "
                    f"un lote que no es del dueño de la vivienda, casi siempre viviendas informales. El catastro registra "
                    f"muchas viviendas independientes bajo el mismo lote matriz, así que para medir la dependencia cada "
                    f"mejora es su propio grupo. Son {fmt(MJ['viviendas'])} viviendas ({pct(MJ['viviendas'], E1['viviendas'])}) "
                    f"en {fmt(MJ['prefijos'])} lotes, que pasan a ser {fmt(MJ['grupos'])} grupos propios.",
                    titulo="Cómo leer estos números")),

        seccion("2. Estratos antes y después de la revisión de coordenadas",
                dbc.Row([dbc.Col(grafico(F.fig_estratos_coordenadas()), md=7),
                         dbc.Col(grafico(F.fig_cobertura_coordenadas()), md=5)]),
                interpretacion(
                    f"{fmt(E1['sin_coordenadas'])} viviendas ({pct(E1['sin_coordenadas'], E1['viviendas'])}) no tienen "
                    "coordenadas: todas son faltantes, ninguna cae fuera de Barranquilla. Como el modelo usa la ubicación, "
                    "solo trabaja con las que sí las tienen.",
                    f"Las coordenadas no faltan al azar: falta un tercio del estrato 1 (cobertura "
                    f"{pct(estr[1]['con_coord'], estr[1]['todas'])}) y casi una cuarta parte del 2 "
                    f"({pct(estr[2]['con_coord'], estr[2]['todas'])}), mientras que los estratos 3 a 6 las conservan casi "
                    "todas. Por eso el estrato 1 baja de 32.8 % a 26.3 %, el 2 también pierde peso y los demás lo ganan: "
                    "el estrato 3 pasa a ser casi tan frecuente como el 1.",
                    "Consecuencia: el modelo describe la ciudad formal. Para las viviendas sin coordenadas (sobre todo "
                    "informales de estratos 1 y 2) sus resultados no aplican sin más.")),

        seccion("3. Calidad de los datos",
                dbc.Row([
                    dbc.Col(html.Div(tabla(pd.DataFrame(tabla_imp), 3), className="tabla-scroll"), md=7),
                    dbc.Col(interpretacion(
                        "Un valor imposible está fuera de lo físicamente posible para cualquier vivienda (un piso 0 o 99, "
                        "un año de construcción 1512). Un valor incoherente es posible por sí solo, pero no con el resto "
                        "de la fila (menos de 6 m² por habitación o más baños que habitaciones + 3).",
                        f"En ambos casos el valor pasa a faltante, queda marcado y la fila se conserva: "
                        f"{E1['filas_con_imposible']} filas con algún valor imposible "
                        f"({100 * E1['filas_con_imposible'] / E1['viviendas']:.2f} %) y {E1['filas_con_incoherente']} con alguna "
                        "incoherencia. Distinto es lo que no es una vivienda (parqueaderos de menos de 10 m², registros que "
                        "describen un edificio entero): esas filas sí se excluyen en el embudo.",
                        "Los faltantes de las variables físicas son menores del 0.2 % y vienen casi todos de estas "
                        "reglas; se imputan con la mediana de entrenamiento dentro del Pipeline.",
                        titulo="Imposible, incoherente o fuera de alcance"), md=5)])),

        seccion("4. Dependencia dentro del grupo y tamaño efectivo",
                dbc.Row([
                    dbc.Col(tarjeta_kpi(f"{GX['icc']:.3f}", "ICC(1) del estrato", "dentro del predio matriz"), md=3),
                    dbc.Col(tarjeta_kpi(f"{GX['m_pond']:.1f}", "tamaño medio ponderado m̃", "Σm² / N"), md=3),
                    dbc.Col(tarjeta_kpi(f"{GX['deff']:.1f}", "efecto de diseño de Kish",
                                        f"errores estándar ≈ {GX['deff'] ** 0.5:.0f} veces mayores"), md=3),
                    dbc.Col(tarjeta_kpi(f"≈ {fmt(int(round(GX['n_ef'], -2)))}", "tamaño efectivo",
                                        f"frente a {fmt(E1['viviendas'])} filas"), md=3),
                ], className="g-3 mb-2"),
                dbc.Row([dbc.Col(grafico(F.fig_viviendas_por_tamano_grupo()), md=6),
                         dbc.Col([grafico(F.fig_kish(GX["icc"], GX["m_pond"]), "kish-fig"),
                                  dbc.Label("Prueba otro ICC:"),
                                  dcc.Slider(id="kish-icc", min=0, max=1, step=0.001, value=round(GX["icc"], 3),
                                             marks={0: "0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1: "1"})], md=6)]),
                interpretacion(
                    f"Paso 1, el grupo: el {pct(GX['grupos_1'], GX['grupos'])} de los grupos son casas de una sola "
                    f"unidad, pero el {pct(GX['filas_multi'], E1['viviendas'])} de las viviendas vive en grupos de varias "
                    f"unidades, y solo {GX['multi_no_uniformes']} de esos {fmt(GX['multi'])} grupos mezclan estratos. El "
                    f"grupo más grande tiene {fmt(GX['max'])} unidades en {GX['torres_mayor']} torres.",
                    f"Paso 2, ICC(1) por ANOVA de una vía: mide qué parte de la variación del estrato está entre grupos. "
                    f"Vale {GX['icc']:.3f}: dentro de un conjunto el estrato casi nunca cambia.",
                    f"Paso 3, tamaño medio ponderado: el promedio simple es {GX['media']:.2f} unidades por grupo, pero "
                    f"visto desde una vivienda típica el grupo tiene m̃ = Σm²/N = {GX['m_pond']:.1f} unidades.",
                    f"Paso 4, efecto de diseño de Kish: deff = 1 + (m̃ − 1)·ICC = 1 + {GX['m_pond'] - 1:.1f} × "
                    f"{GX['icc']:.3f} ≈ {GX['deff']:.1f}. Con ICC = 0 sería 1 (filas independientes); con ICC = 1 sería "
                    "m̃ (cada grupo vale por una sola observación).",
                    f"Paso 5, tamaño efectivo: n_ef = N / deff ≈ {fmt(int(round(GX['n_ef'])))}. Las {fmt(E1['viviendas'])} "
                    f"filas aportan tanta información independiente como unas {fmt(int(round(GX['n_ef'], -2)))} viviendas sueltas. "
                    f"Sin la excepción de las mejoras el resultado sería peor y engañoso: ICC {G0['icc']:.3f}, grupo máximo "
                    f"{fmt(G0['max'])}, deff {G0['deff']:.1f}, n_ef ≈ {fmt(int(round(G0['n_ef'], -1)))}.",
                    "Es una descripción, no una decisión de modelado: lo que se hace con ella es mantener juntos los "
                    "edificios y las zonas al partir los datos (bloques espaciales) y medir la incertidumbre remuestreando "
                    "bloques, no filas.",
                    titulo="Interpretación, en cinco pasos")),
    ], fluid=True)

    # ===========================================================================
    # Pestaña 3 — EDA
    # ===========================================================================
    selector_particion = dbc.RadioItems(
        id="eda-particion", value="train", inline=True,
        options=[{"label": "Solo entrenamiento (como en el EDA)", "value": "train"},
                 {"label": "Todas las viviendas", "value": "todas"}])

    tab_eda = dbc.Container([
        dbc.Alert(["El EDA se hace ", html.B("solo con el conjunto de entrenamiento"), ", para que ninguna decisión del "
                   "modelo se tome mirando el test. Puedes cambiar a todas las viviendas para describir la ciudad completa."],
                  color="info", className="mt-3"),
        html.Div(selector_particion, className="mb-3 filtros"),

        seccion("1. Variable objetivo",
                dbc.Row([dbc.Col(grafico(F.fig_distribucion_estrato("train"), "eda-dist"), md=5),
                         dbc.Col(grafico(F.fig_gradiente_norte_sur("train"), "eda-ns"), md=7)]),
                interpretacion(
                    "En entrenamiento los estratos 1 y 3 son los más frecuentes (alrededor de una cuarta parte cada uno) y el 6 "
                    "apenas pasa del 5 %: la razón entre la clase mayor y la menor es 4.6:1. Con este desbalance, un modelo que "
                    "siempre predijera la clase más frecuente acertaría cerca de una de cada cuatro viviendas sin aprender nada.",
                    "La ciudad está segregada de sur a norte: las franjas del sur son casi todas de estratos 1 y 2, y el "
                    "estrato medio sube hasta cerca de 5 en la penúltima franja (F9), donde dominan los estratos 4 a 6. En la "
                    "franja más al norte vuelve a bajar (3.3), porque allí también hay barrios de estratos bajos: el gradiente "
                    "no es una línea recta. La ubicación será la información más fuerte del modelo, y también la razón por la "
                    "que la validación debe ser espacial.")),

        seccion("2. Variables numéricas",
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id="eda-num", options=[{"label": v, "value": k} for k, v in F.NUMERICAS.items()],
                                         value="total_banios", clearable=False), md=5),
                    dbc.Col(dbc.Checklist(id="eda-log", options=[{"label": "escala log(1 + x)", "value": "log"}],
                                          value=["log"], switch=True), md=3),
                ], className="filtros mb-2"),
                grafico(F.fig_numerica("total_banios"), "eda-num-fig"),
                html.Div(id="eda-num-texto"),
                html.H5("Resumen estadístico (train)", className="mt-3"),
                html.Div(tabla(F.tabla_resumen_numericas(), 2), id="eda-resumen"),
                interpretacion(
                    "Casi todas las variables físicas son muy asimétricas a la derecha (asimetría > 1) y tienen colas largas: "
                    "por eso se transforman con log(1 + x) dentro del Pipeline. Los valores faltantes son muy pocos (< 0.2 %) "
                    "y se imputan con la mediana de entrenamiento.",
                    "El número de baños es la variable física más asociada al estrato (η² ≈ 0.31), seguida del piso de "
                    "ubicación y del área construida. La relación es monótona: a mayor estrato, más baños y más área, pero "
                    "con mucho traslape entre estratos vecinos, que es justo donde el modelo se equivocará.")),

        seccion("3. Variables categóricas",
                dcc.Dropdown(id="eda-cat", options=[{"label": v, "value": k} for k, v in F.CATEGORICAS.items()],
                             value="condicion_predio", clearable=False, className="filtros mb-2", style={"maxWidth": 480}),
                grafico(F.fig_categorica("condicion_predio"), "eda-cat-fig"),
                interpretacion(
                    "El régimen de propiedad es la categórica más informativa: los predios informales son casi todos de "
                    "estratos 1 y 2, y las unidades en propiedad horizontal (apartamentos) se concentran en estratos medios y "
                    "altos. El uso y el tipo de vivienda aportan menos y son redundantes con el régimen.",
                    "Las categorías con menos del 1 % de las viviendas se agrupan en una sola dentro del Pipeline "
                    "(one-hot con categorías infrecuentes), para no crear columnas casi vacías.")),

        seccion("4. Correlaciones y valores faltantes",
                dbc.Row([dbc.Col(grafico(F.fig_correlacion(), "eda-corr"), md=7),
                         dbc.Col(grafico(F.fig_faltantes(), "eda-falt"), md=5)]),
                interpretacion(
                    "Se usa Spearman porque las relaciones son monótonas pero no lineales y hay colas largas. Las variables "
                    "de tamaño (área, baños, habitaciones) están correlacionadas entre sí, pero ninguna pareja es tan alta "
                    "como para excluir una variable (los VIF del Entregable 1 quedaron por debajo del umbral).",
                    "El terreno se correlaciona negativamente con el piso de ubicación: en los edificios, cada apartamento "
                    "registra solo su cuota del lote, que es más pequeña cuanto más alto es el edificio.")),

        seccion("5. Componente espacial",
                dbc.Row([dbc.Col(dcc.Dropdown(id="eda-mapa-var",
                                              options=[{"label": "Estrato medio", "value": "estrato_num"}] +
                                                      [{"label": v, "value": k} for k, v in F.NUMERICAS.items()
                                                       if k != "dist_centro_km"],
                                              value="estrato_num", clearable=False), md=5)], className="filtros mb-2"),
                grafico(F.fig_mapa("estrato_num", "train"), "eda-mapa"),
                dbc.Row([dbc.Col(grafico(F.fig_correlograma()), md=6), dbc.Col(grafico(F.fig_tamano_edificios()), md=6)]),
                interpretacion(
                    f"El estrato tiene una autocorrelación espacial extremadamente fuerte (I de Moran = {K['moran']:.2f} entre "
                    "edificios vecinos): viviendas cercanas casi siempre comparten estrato. El correlograma muestra que ese "
                    "parecido cae por debajo de 0.3 a unos 2.9 km; con eso se fijó un buffer de 1 km entre entrenamiento y "
                    "validación.",
                    "Además, el estrato casi no varía dentro de un predio matriz (ICC = 0.992; tamaño efectivo ≈ 2 300, ver pestaña 2) y unas pocas torres concentran "
                    "muchas viviendas. Las filas no son datos independientes: por eso la partición mantiene juntos los "
                    "edificios y las zonas, y la incertidumbre se mide remuestreando bloques, no filas.",
                    "Con «solo entrenamiento» el mapa muestra huecos rectangulares: son los bloques de 2 km reservados para "
                    "test, que el EDA no mira.",
                    "Cerca del 16 % de las viviendas (predios informales, casi todos de estratos 1–2) no tienen coordenadas y "
                    "quedan fuera del modelo: las conclusiones aplican a la ciudad formal.")),
    ], fluid=True)

    # ===========================================================================
    # Pestaña 3 — Modelos base
    # ===========================================================================
    opciones_modelo = [{"label": F.NOMBRE_CORTO[m], "value": m} for m in F.modelos_disponibles()]
    rangos = F.rangos_simulador()
    etq_tv = {"tv_1": "Vivienda de Interés Social", "tv_2": "Vivienda de Interés Prioritario",
              "tv_3": "Vivienda que no es de interés social", "tv_4": "No aplica"}


    def control_numero(var, etiqueta):
        p1, p50, p99 = rangos[var]
        return dbc.Col([dbc.Label(etiqueta), dbc.Input(id=f"sim-{var}", type="number", value=round(p50, 1),
                                                       min=0, step=1 if p99 < 50 else 5)], md=3)


    simulador = dbc.Card(dbc.CardBody([
        html.H5("Simulador: ¿qué estrato predice el modelo?", className="fw-bold"),
        html.P("Cambia las características de una vivienda y su ubicación. Los valores iniciales son las medianas de "
               "entrenamiento. Es una herramienta para entender el modelo, no para asignar estratos reales.",
               className="text-muted"),
        dbc.Row([control_numero(v, F.NUMERICAS[v]) for v in ["area_construida", "area_catastral_terreno",
                                                              "total_habitaciones", "total_banios"]], className="g-2"),
        dbc.Row([control_numero(v, F.NUMERICAS[v]) for v in ["total_plantas", "planta_ubicacion", "altura",
                                                              "antiguedad"]], className="g-2 mt-1"),
        dbc.Row([dbc.Col([dbc.Label(F.CATEGORICAS[c]), dcc.Dropdown(
            id=f"sim-{c}", value=rangos["moda_categorias"][c], clearable=False,
            options=[{"label": etq_tv.get(o, o.replace("_", " ")), "value": o} for o in rangos["categorias"][c]])], md=4)
            for c in ["condicion_predio", "uso", "tipo_vivienda"]], className="g-2 mt-1"),
        dbc.Row([dbc.Col([dbc.Label("Latitud (sur ↔ norte)"),
                          dcc.Slider(id="sim-lat", min=10.90, max=11.04, step=0.002, value=10.97,
                                     marks={10.9: "10.90", 10.97: "10.97", 11.04: "11.04"})], md=6),
                 dbc.Col([dbc.Label("Longitud (oeste ↔ este)"),
                          dcc.Slider(id="sim-lon", min=-74.90, max=-74.76, step=0.002, value=-74.81,
                                     marks={-74.9: "−74.90", -74.83: "−74.83", -74.76: "−74.76"})], md=6)],
                className="g-2 mt-2"),
        dbc.Row([dbc.Col(dcc.Graph(id="sim-fig", config=GRAF), md=6),
                 dbc.Col(dcc.Graph(id="sim-mapa", config=GRAF), md=6)]),
    ]), className="mt-2")


    tab_modelos = dbc.Container([
        dbc.Row([
            dbc.Col(tarjeta_kpi(f"{IC['valor en test']['f1_macro']:.2f}", "F1 macro en test",
                                f"IC 95 % por bloques: [{IC['IC95_inf']['f1_macro']:.2f}, {IC['IC95_sup']['f1_macro']:.2f}]"), md=3),
            dbc.Col(tarjeta_kpi(f"{IC['valor en test']['accuracy']:.2f}", "accuracy en test", "por debajo de la alerta de 0.80"), md=3),
            dbc.Col(tarjeta_kpi(f"{R['resultados_test']['accuracy_±1']['Logística B_fisicas+ubicacion']:.2f}",
                                "accuracy ±1 estrato", "casi todos los errores son entre vecinos"), md=3),
            dbc.Col(tarjeta_kpi(f"{IC['valor en test']['kappa_cuadrático']:.2f}", "kappa cuadrático",
                                "acuerdo ordinal alto"), md=3),
        ], className="g-3 mt-2"),

        seccion("1. Modelo base y líneas base",
                html.P(["Modelo base: ", html.B("regresión logística multinomial"), " con penalización L2, dentro de un "
                        "Pipeline (winsorización → imputación por mediana → log(1 + x) → estandarización; categóricas a "
                        "one-hot). Se comparan dos conjuntos de variables: ", html.B("A"), " solo físicas y ", html.B("B"),
                        " físicas + ubicación (x, y y distancia al centro). Los hiperparámetros (C y pesos de clase) se "
                        "eligieron con validación cruzada espacial y la regla de una desviación estándar: A con C = 0.01 y "
                        "pesos balanceados; B con C = 0.01 sin pesos. Líneas base: tres DummyClassifier y la «moda por zona» "
                        "(la clase más frecuente en la celda de 2 km), que es la versión para clasificación de la media por "
                        "zona."]),
                dcc.Dropdown(id="mod-metrica", value="f1_macro", clearable=False, style={"maxWidth": 420},
                             options=[{"label": v, "value": k} for k, v in F.METRICAS_COMPARABLES.items()],
                             className="filtros mb-2"),
                grafico(F.fig_comparacion("f1_macro"), "mod-comp"),
                html.Div(tabla(F.tabla_metricas()), className="tabla-scroll"),
                interpretacion(
                    "La logística B es el modelo principal: tiene el mayor F1 en la validación cruzada espacial (0.30) y en "
                    "test (0.45). Supera con claridad a las Dummy: la diferencia de F1 frente a la mayoritaria es +0.38 "
                    "(IC 95 % [0.11, 0.39]) y frente a la estratificada +0.28 (IC [0.03, 0.31]).",
                    "Frente a la moda por zona también gana en todas las métricas de test (F1 0.45 frente a 0.33; MAE 0.46 "
                    "frente a 0.94), pero la diferencia de F1 no es estadísticamente significativa (+0.12, IC [−0.01, 0.19]): "
                    "con solo 7 bloques de test la incertidumbre es grande. Lo mismo pasa entre B y A.",
                    "El F1 de test (0.45) es mayor que el de la CV (0.30) por dos razones probables: los 7 bloques de test "
                    "resultaron favorables (con muy poco estrato 6) y el modelo final se entrena con 138 239 viviendas, entre "
                    "1.9 y 2.6 veces las de cada fold. La estimación más representativa del desempeño en zonas nuevas es la "
                    "de la CV espacial. La accuracy (0.60) no activa la alerta de posible fuga (≥ 0.80).")),

        seccion("2. ¿Por qué validar por bloques espaciales?",
                grafico(F.fig_optimismo()),
                interpretacion(
                    "Con validación aleatoria, viviendas del mismo edificio y de la misma cuadra caen a la vez en "
                    "entrenamiento y validación, y el modelo «reconoce» la zona: el F1 sube a 0.60, el doble que con "
                    "bloques. Ese número sería engañoso para predecir en zonas nuevas.",
                    "El buffer también importa: sin él el F1 de CV es 0.40, con 1 km baja a 0.30 y con el alcance completo "
                    "del correlograma (2.9 km) cae a 0.18, porque se pierde mucho entrenamiento. Se eligió 1 km como "
                    "compromiso.")),

        seccion("3. Diagnóstico por modelo",
                dbc.Row([dbc.Col(dcc.Dropdown(id="mod-modelo", options=opciones_modelo,
                                              value="Logística B_fisicas+ubicacion", clearable=False), md=5),
                         dbc.Col(dbc.Checklist(id="mod-normalizar", value=["si"], switch=True,
                                               options=[{"label": "matriz en % por fila", "value": "si"}]), md=3)],
                        className="filtros mb-2"),
                dbc.Row([dbc.Col(grafico(F.fig_confusion(), "mod-conf"), md=6),
                         dbc.Col(grafico(F.fig_por_clase(), "mod-clase"), md=6)]),
                dbc.Row([dbc.Col(grafico(F.fig_roc(), "mod-roc"), md=6),
                         dbc.Col(grafico(F.fig_calibracion(), "mod-cal"), md=6)]),
                interpretacion(
                    "En la logística B los errores son casi siempre entre estratos vecinos (accuracy ±1 = 0.94). Hay dos "
                    "desplazamientos hacia el centro de la escala: el estrato 2 casi nunca se predice (recall 0.06; la "
                    "mayoría se clasifica como 3) y más de la mitad de los estratos 5 y 6 se predicen como 4.",
                    "Las curvas ROC muestran que el modelo sí ordena bien el estrato 2 (AUC alto), pero casi nunca gana el "
                    "argmax frente al 3: el problema es de calibración y de umbral, no de falta de señal. La calibración "
                    "confirma que el estrato 3 está sobreestimado y el 2 subestimado.",
                    "Selecciona otro modelo en la lista para comparar: la Dummy mayoritaria solo predice estrato 1 (la clase más frecuente en el entrenamiento que queda tras el buffer) y la moda "
                    "por zona acierta las zonas grandes pero falla en los bordes entre estratos.")),

        seccion("4. Coeficientes",
                dbc.RadioItems(id="mod-coef", value="B_fisicas+ubicacion", inline=True, className="filtros mb-2",
                               options=[{"label": "Logística B", "value": "B_fisicas+ubicacion"},
                                        {"label": "Logística A", "value": "A_fisicas"}]),
                grafico(F.fig_coeficientes(), "mod-coef-fig"),
                interpretacion(
                    "Cada celda es el coeficiente de la variable (estandarizada) en la ecuación de ese estrato: azul empuja "
                    "hacia el estrato, rojo lo aleja. Se muestran las variables con mayor diferencia entre el estrato 6 y el 1.",
                    "La posición norte-sur domina: moverse al norte aumenta la probabilidad de estratos altos y reduce la de "
                    "estratos bajos. Entre las físicas, más baños y más área empujan hacia estratos altos, y los predios "
                    "informales hacia el estrato 1.",
                    "Cautela: las variables están correlacionadas entre sí, así que un coeficiente aislado no es un efecto "
                    "causal, y la multinomial no usa el orden de los estratos.")),

        seccion("5. Curva de aprendizaje y residuos espaciales",
                dbc.Row([dbc.Col(grafico(F.fig_curva_aprendizaje()), md=5), dbc.Col(grafico(F.fig_mapa_residuos()), md=7)]),
                interpretacion(
                    "La curva de validación casi no sube al multiplicar los datos por 20, de 2 700 a 54 000 viviendas (F1 de "
                    "0.27 a 0.30): es poco probable que más filas cambien mucho este modelo. Lo que falta es flexibilidad (no linealidades, interacciones) y variables del "
                    "entorno. La brecha entre entrenamiento y validación refleja sobre todo el cambio de zona, no "
                    "sobreajuste a filas.",
                    "En el mapa, cada punto es el residuo medio (estrato real − estrato esperado) de las viviendas de test en "
                    "una celda de ~110 m: azul significa que el modelo subestima el estrato, rojo que lo sobreestima.",
                    f"Los residuos siguen fuertemente autocorrelacionados en el espacio (Moran = "
                    f"{D['moran_residuos']['B_fisicas+ubicacion']['I']:.2f}): el modelo capta el gradiente norte-sur, pero no "
                    "los barrios. Hay zonas enteras donde subestima (azul) o sobreestima (rojo). Esa fue la motivación para "
                    "probar, en el Entregable 2, modelos con efectos locales y variables del entorno.")),

        seccion("6. Simulador", simulador),
    ], fluid=True)

    # ===========================================================================
    # Layout
    # ===========================================================================
    return dict(t1=tab_contexto, t2=tab_datos, t3=tab_eda, t4=tab_modelos, GX=GX)
