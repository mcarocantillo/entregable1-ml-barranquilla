# -*- coding: utf-8 -*-
"""
Diagnóstico de los rangos plausibles (valores imposibles) del catastro.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Pregunta: ¿los rangos de config.LIMITES_PLAUSIBLES son demasiado amplios para
una unidad de vivienda de Barranquilla? El script NO cambia nada del libro;
solo produce evidencia para decidir y documentar los límites.

Se apoya en las notas de clase del profesor Lihki Rubio:
  - 9.10.4.1.2: outliers por el criterio de Tukey (fuera de
    [Q1 - 1.5·IQR, Q3 + 1.5·IQR]) e inspección visual (boxplots).
  - 9.10.4.1.4: % de faltantes por variable y plan de tratamiento (imputar
    con mediana si el % es bajo; eliminar si es muy alto, > 70 %).
y en las instrucciones del proyecto (sección 1, "valores imposibles o inconsistentes"; orden de
trabajo: las decisiones aprendidas de los datos se toman solo con train).

Distinción clave que guía el diagnóstico:
  - IMPOSIBLE (error de captura o código): se convierte en faltante.
  - EXTREMO PERO POSIBLE (p. ej. una casa grande de estrato 6): se conserva y
    lo controla la winsorización p0.1/p99.9 del Pipeline.
Por eso no basta con mirar el percentil: hay que ver QUIÉN tiene el valor
(condición del predio, uso, estrato) y si es coherente con el resto de la fila.

Uso (Google Colab o local):
    python diagnostico_rangos.py RUTA_AL_CSV [RUTA_A_dataset_modelado.parquet]

Si se da el parquet del cap. 2, el diagnóstico se restringe a TRAIN (orden de
trabajo de las instrucciones del proyecto). Sin parquet usa todas las filas y lo advierte.
Escribe el reporte en diagnostico_rangos_reporte.txt (y lo imprime).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Parámetros (copiados de src/config.py para que el script sea autónomo) ---
USOS_NO_HABITABLES = ["Residencial_Garajes_En_PH", "Residencial_Garajes_Cubiertos",
                      "Residencial_Depositos_Lockers"]
LIMITES_ACTUALES = {
    "total_habitaciones": (0, 30), "total_banios": (0, 20), "total_plantas": (0, 40),
    "planta_ubicacion": (1, 60), "area_construida": (1, 20_000),
    "area_catastral_terreno": (0, 50_000), "anio_construccion": (1900, 2026),
}
# Umbrales "residenciales razonables" para UNA unidad de vivienda: no son
# límites de imposibilidad, son preguntas ("¿quién supera esto?").
UMBRALES_REVISION = {
    "total_habitaciones": 10, "total_banios": 8, "total_plantas": 35,
    "planta_ubicacion": 35, "area_construida": 1_000, "area_catastral_terreno": 10_000,
}
VARS = list(LIMITES_ACTUALES)

salida = []


def p(*a):
    """Imprime y guarda en el reporte."""
    t = " ".join(str(x) for x in a)
    print(t)
    salida.append(t)


def titulo(t):
    p("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def tabla(df):
    p(df.to_string())


def conteo(serie, n=None):
    """value_counts como dict legible (claves int/str, sin np.int64)."""
    vc = serie.value_counts(dropna=False)
    vc = vc.sort_index() if n is None else vc.head(n)
    return {(int(k) if isinstance(k, (int, np.integer, float, np.floating)) and pd.notna(k)
             and float(k).is_integer() else str(k)): int(v) for k, v in vc.items()}


def normalizar(s):
    s = s.astype("string").str.strip().str.replace(r"\s+", "_", regex=True)
    return s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


# --- 1. Carga y mismos pasos 1-3 del embudo del cap. 1 ------------------------
if len(sys.argv) < 2:
    candidatos = list(Path(".").glob("predios_residenciales_barranquilla*.csv")) + \
                 list(Path("/content").glob("predios_residenciales_barranquilla*.csv"))
    if not candidatos:
        sys.exit("Indica la ruta del CSV: python diagnostico_rangos.py RUTA.csv [dataset_modelado.parquet]")
    ruta_csv = candidatos[0]
else:
    ruta_csv = Path(sys.argv[1])
ruta_parquet = Path(sys.argv[2]) if len(sys.argv) > 2 else None

df = pd.read_csv(ruta_csv, dtype={"numero_predial_nacional": "string"}, low_memory=False)
df.insert(0, "id_fila", np.arange(len(df)))  # mismo id_fila que src/limpieza.py
for c in ["uso", "condicion_predio", "estrato"]:
    df[c] = normalizar(df[c])
df = df[~df["uso"].isin(USOS_NO_HABITABLES)]
area = pd.to_numeric(df["area_construida"], errors="coerce")
df = df[area.notna() & (area > 0)].copy()
df["estrato_num"] = pd.to_numeric(df["estrato"].str.extract(r"_(\d)$")[0], errors="coerce")
df = df[df["estrato_num"].between(1, 6)].copy()
for c in VARS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

titulo("1. DATOS ANALIZADOS")
p(f"CSV: {ruta_csv.name} | filas tras pasos 1-3 del embudo: {len(df):,}")
if ruta_parquet and ruta_parquet.exists():
    part = pd.read_parquet(ruta_parquet, columns=["id_fila", "particion"])
    df = df.merge(part, on="id_fila", how="left")
    p(f"Partición del cap. 2: {df['particion'].value_counts(dropna=False).to_dict()}")
    df = df[df["particion"] == "train"].copy()
    p(f"-> Se analiza SOLO TRAIN: {len(df):,} filas (orden de trabajo de las instrucciones del proyecto).")
    p("   Las filas sin coordenadas no están en la partición y quedan fuera.")
else:
    p("AVISO: sin dataset_modelado.parquet se analizan TODAS las filas (train + test).")
    p("       Úsalo solo como exploración; la decisión final debe verificarse en train.")

# --- 2. Resumen por variable: percentiles, Tukey (profesor Lihki 9.10.4.1.2) ----------
titulo("2. RESUMEN POR VARIABLE (valores crudos, antes de anular imposibles)")
filas = []
for c in VARS:
    x = df[c]
    q1, q3 = x.quantile([.25, .75])
    iqr = q3 - q1
    sup_tukey = q3 + 1.5 * iqr
    # Tukey sobre log1p: con colas largas (área, terreno) Tukey en escala cruda
    # marca miles de valores legítimos; en escala log es más informativo.
    lx = np.log1p(x.clip(lower=0))
    lq1, lq3 = lx.quantile([.25, .75])
    sup_tukey_log = np.expm1(lq3 + 1.5 * (lq3 - lq1))
    mn, mx = LIMITES_ACTUALES[c]
    filas.append({
        "variable": c, "n": int(x.notna().sum()), "% faltante": round(100 * x.isna().mean(), 2),
        "min": x.min(), "p50": x.median(), "p90": x.quantile(.9), "p99": x.quantile(.99),
        "p99.9": x.quantile(.999), "max": x.max(),
        "Tukey sup": round(sup_tukey, 1), "n > Tukey": int((x > sup_tukey).sum()),
        "Tukey sup (log)": round(sup_tukey_log, 1), "n > Tukey log": int((x > sup_tukey_log).sum()),
        "límite actual": f"[{mn}, {mx}]", "n fuera límite": int((x.notna() & ~x.between(mn, mx)).sum()),
        "umbral revisión": UMBRALES_REVISION.get(c, "-"),
        "n > umbral": int((x > UMBRALES_REVISION[c]).sum()) if c in UMBRALES_REVISION else "-",
    })
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
tabla(pd.DataFrame(filas).set_index("variable").T)
p("\nLectura: 'n > Tukey' son outliers estadísticos (profesor Lihki), NO imposibles. Si Tukey marca")
p("miles de filas en una variable de cola larga, el criterio no sirve como regla de limpieza.")

# --- 3. ¿Quién supera los umbrales de revisión? -----------------------------
titulo("3. ¿QUIÉN SUPERA LOS UMBRALES DE REVISIÓN? (condición del predio, uso, estrato)")
for c, u in UMBRALES_REVISION.items():
    m = df[c] > u
    if m.sum() == 0:
        p(f"\n{c} > {u}: ninguna fila.")
        continue
    p(f"\n--- {c} > {u}: {int(m.sum()):,} filas ({100 * m.mean():.3f} %) ---")
    sub = df[m]
    p("condición del predio:", conteo(sub["condicion_predio"], n=10))
    p("uso (top 5):", conteo(sub["uso"], n=5))
    p("estrato:", conteo(sub["estrato_num"]),
      "| estrato en todo el conjunto (%):",
      {int(k): round(float(100 * v), 1) for k, v in df["estrato_num"].value_counts(normalize=True).sort_index().items()})
    # Coherencia con el área: 30 habitaciones en 60 m2 es error; en 3 000 m2, un edificio.
    p("área construida de esas filas: mediana",
      round(sub["area_construida"].median(), 1), "| p10", round(sub["area_construida"].quantile(.1), 1),
      "| p90", round(sub["area_construida"].quantile(.9), 1))

# --- 4. Razones entre variables: incoherencias dentro de la fila -------------
titulo("4. INCOHERENCIAS DENTRO DE LA FILA (razones entre variables)")
h = df["total_habitaciones"].where(df["total_habitaciones"] > 0)
b = df["total_banios"]
a = df["area_construida"]
reglas_razon = {
    "área por habitación < 6 m2 (habitación imposible de tan pequeña)": (a / h) < 6,
    "área por habitación > 300 m2": (a / h) > 300,
    "baños > habitaciones + 3": b > df["total_habitaciones"] + 3,
    "planta_ubicacion > total_plantas (piso por encima del edificio)":
        df["planta_ubicacion"] > df["total_plantas"],
    "área > 1 000 m2 con <= 3 habitaciones": (a > 1000) & (df["total_habitaciones"] <= 3),
}
filas = []
for nombre, m in reglas_razon.items():
    m = m.fillna(False)
    filas.append({"regla": nombre, "n": int(m.sum()), "%": round(100 * m.mean(), 3),
                  "condición más común": df.loc[m, "condicion_predio"].mode().iat[0] if m.any() else "-"})
tabla(pd.DataFrame(filas).set_index("regla"))
p("\nDistribución de área por habitación (m2):",
  (a / h).describe(percentiles=[.001, .01, .5, .99, .999]).round(1).to_dict())

# --- 5. Año de construcción ---------------------------------------------------
titulo("5. AÑO DE CONSTRUCCIÓN")
y = df["anio_construccion"]
p("valores < 1950 (valor: filas):", dict(list(conteo(y[y < 1950]).items())[:40]))
p("valores > 2026:", conteo(y[y > 2026]))
p(f"año real mínimo >= 1800: {y[y >= 1800].min()} | filas 1800-1899: {int(y.between(1800, 1899).sum())}")
p("Lectura: si no hay años reales entre 1800 y 1899, el límite 1900 no corta construcciones")
p("históricas reales; los valores como 1512 o 2500 son códigos centinela.")

# --- 6. Altura del edificio ---------------------------------------------------
titulo("6. ALTURA: total_plantas y planta_ubicacion")
p("total_plantas, 15 valores más altos (valor: filas):",
  dict(list(conteo(df["total_plantas"].dropna()).items())[-15:]))
p("planta_ubicacion, 15 valores más altos (valor: filas):",
  dict(list(conteo(df["planta_ubicacion"].dropna()).items())[-15:]))
p("Lectura: el techo debe ser el edificio más alto real de la ciudad (verificar con fuente")
p("externa); valores aislados por encima de él son códigos o errores.")

# --- 7. Casos extremos para inspección manual ---------------------------------
titulo("7. CASOS EXTREMOS PARA INSPECCIÓN (NPN truncado por privacidad)")
cols = ["condicion_predio", "uso", "estrato_num", "total_habitaciones", "total_banios",
        "total_plantas", "planta_ubicacion", "area_construida", "area_catastral_terreno"]
for c in ["total_habitaciones", "total_banios", "area_construida"]:
    top = df.nlargest(10, c)[["numero_predial_nacional"] + cols].copy()
    top["numero_predial_nacional"] = top["numero_predial_nacional"].str[:12] + "…"
    p(f"\nTop 10 por {c}:")
    tabla(top.reset_index(drop=True))

# --- 8. Efecto de cada opción en el % de faltantes (profesor Lihki 9.10.4.1.4) --------
titulo("8. ¿CUÁNTO FALTANTE CREARÍA CADA OPCIÓN? (criterio de imputación del profesor Lihki)")
filas = []
for c, (mn, mx) in LIMITES_ACTUALES.items():
    falt0 = df[c].isna().mean()
    fuera_act = df[c].notna() & ~df[c].between(mn, mx)
    fila = {"variable": c, "% faltante original": round(100 * falt0, 3),
            "% tras límite actual": round(100 * (falt0 + fuera_act.mean()), 3)}
    if c in UMBRALES_REVISION:
        fuera_rev = df[c].notna() & ~df[c].between(mn, UMBRALES_REVISION[c])
        fila["% si el techo fuera el umbral de revisión"] = round(100 * (falt0 + fuera_rev.mean()), 3)
    filas.append(fila)
tabla(pd.DataFrame(filas).set_index("variable"))
p("\nLectura (notas 9.10.4.1.4): con % bajos la imputación por mediana dentro del Pipeline")
p("es adecuada; ninguna opción se acerca al 30 % ni al 70 % que justificarían otro tratamiento.")

# --- 9. Segunda ronda: unidades diminutas, ceros y registros "edificio" ----------
titulo("9. SEGUNDA RONDA: UNIDADES DIMINUTAS, CEROS Y REGISTROS QUE SON EDIFICIOS")
a = df["area_construida"]
p("Área construida pequeña (filas con área < umbral):",
  {f"< {u} m2": int((a < u).sum()) for u in [5, 10, 15, 20, 25, 30]})
peq = df[a < 20]
p("\nFilas con área < 20 m2:", len(peq))
p("  condición:", conteo(peq["condicion_predio"], n=10))
p("  uso:", conteo(peq["uso"], n=8))
p("  planta_ubicacion (top):", conteo(peq["planta_ubicacion"], n=10))
p("  habitaciones:", conteo(peq["total_habitaciones"], n=8), "| baños:", conteo(peq["total_banios"], n=8))
p("  estrato:", conteo(peq["estrato_num"]))
pu99 = df[df["planta_ubicacion"] >= 90]
p("\nFilas con planta_ubicacion >= 90 (códigos):", len(pu99),
  "| área mediana:", round(pu99["area_construida"].median(), 1),
  "| habitaciones:", conteo(pu99["total_habitaciones"], n=6))
alto = df[df["planta_ubicacion"].between(30, 89)]
p("Filas con planta_ubicacion 30-89:", len(alto), "| área mediana:", round(alto["area_construida"].median(), 1),
  "| uso:", conteo(alto["uso"], n=4), "| estrato:", conteo(alto["estrato_num"]))
p("\nCeros: habitaciones = 0:", int((df["total_habitaciones"] == 0).sum()),
  "| baños = 0:", int((df["total_banios"] == 0).sum()),
  "| ambos = 0:", int(((df["total_habitaciones"] == 0) & (df["total_banios"] == 0)).sum()))
z = df[df["total_habitaciones"] == 0]
p("  habitaciones = 0 por condición:", conteo(z["condicion_predio"], n=8),
  "| área mediana:", round(z["area_construida"].median(), 1))
p("\nPH_Matriz en train:", int((df["condicion_predio"] == "PH_Matriz").sum()),
  "| área mediana:", round(df.loc[df["condicion_predio"] == "PH_Matriz", "area_construida"].median(), 1))
ed = df[(df["total_habitaciones"] > 20) | (df["area_construida"] > 2000)]
p("Registros tipo 'edificio' (habitaciones > 20 o área > 2 000 m2):", len(ed),
  "| condición:", conteo(ed["condicion_predio"], n=6), "| uso:", conteo(ed["uso"], n=4))
p("Años con muchas filas iguales (posible 'amontonamiento'), top 10:", conteo(df["anio_construccion"], n=10))

Path("diagnostico_rangos_reporte.txt").write_text("\n".join(salida), encoding="utf-8")
print("\nReporte guardado en diagnostico_rangos_reporte.txt")
