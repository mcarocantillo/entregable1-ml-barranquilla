# -*- coding: utf-8 -*-
"""
Genera un CSV SINTÉTICO con el MISMO esquema que
predios_residenciales_barranquilla.csv. Sirve ÚNICAMENTE para comprobar que
todos los notebooks del libro corren de principio a fin (sin errores y en un
tiempo razonable con ~380 mil filas) cuando no se tiene acceso a los datos
reales. Sus números NO significan nada: no deben usarse en el informe.

Imita a propósito los rasgos que ya conocemos de los datos reales: gradiente
norte-sur del estrato, edificios PH con muchas unidades que comparten
centroide, ~14 % de predios informales sin coordenadas, códigos centinela
(1512, 2500, 20) en el año, registros NPH de edificios completos con cientos
de habitaciones, garajes/depósitos, estratos No_Aplica/Otro y filas sin predio.

Uso:  python tests/generar_datos_sinteticos.py  [ruta_salida.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
ETIQ = {1: "Bajo_Bajo_1", 2: "Bajo_2", 3: "Medio_Bajo_3", 4: "Medio_4", 5: "Medio_Alto_5", 6: "Alto_6"}
PROP = np.array([27.8, 19.8, 20.4, 15.3, 6.0, 6.4]) / 100


def campo_socioeconomico(lat, lon, centros, pesos):
    s = 2.2 * (lat - 10.96) / 0.06 - 0.8 * (lon + 74.80) / 0.05
    for (cl, co), w in zip(centros, pesos):
        s += w * np.exp(-(((lat - cl) / 0.012) ** 2 + ((lon - co) / 0.012) ** 2))
    return s


def estrato_desde_campo(s):
    cortes = np.quantile(s, np.cumsum(PROP)[:-1])
    return 1 + np.searchsorted(cortes, s)


def main(salida):
    centros = list(zip(rng.uniform(10.91, 11.03, 25), rng.uniform(-74.86, -74.76, 25)))
    pesos = rng.normal(0, 1.2, 25)

    # --- Casas NPH (una unidad, a veces dos construcciones por lote) --------
    n_casas = 150_000
    lat = rng.uniform(10.905, 11.035, n_casas)
    lon = rng.uniform(-74.865, -74.765, n_casas)
    s = campo_socioeconomico(lat, lon, centros, pesos) + rng.normal(0, 0.6, n_casas)
    est = estrato_desde_campo(s)
    casas = pd.DataFrame({"lat": lat, "lon": lon, "e": est, "tipo": "casa"})
    dobles = casas.sample(frac=0.12, random_state=1)
    casas = pd.concat([casas, dobles], ignore_index=True)
    casas["edif"] = np.r_[np.arange(n_casas), dobles.index.to_numpy()]
    casas["unidad"] = casas.groupby("edif").cumcount()

    # --- Edificios PH (muchas unidades con el centroide de la matriz) -------
    n_edif = 7_000
    lat = rng.uniform(10.93, 11.035, n_edif)
    lon = rng.uniform(-74.86, -74.77, n_edif)
    s = campo_socioeconomico(lat, lon, centros, pesos) + 1.5 + rng.normal(0, 0.4, n_edif)
    est_e = np.clip(estrato_desde_campo(s) + rng.integers(0, 2, n_edif), 1, 6)
    unidades = np.clip(rng.poisson(6 + 5 * est_e), 2, 120)
    pisos = np.clip((unidades / rng.integers(2, 6, n_edif)).astype(int), 2, 35)
    idx = np.repeat(np.arange(n_edif), unidades)
    ph = pd.DataFrame({"lat": lat[idx], "lon": lon[idx], "e": est_e[idx], "tipo": "ph",
                       "edif": n_casas + idx, "pisos_edif": pisos[idx]})
    ph["unidad"] = ph.groupby("edif").cumcount()

    # --- Informales (sin terreno ni matriz -> sin coordenadas) --------------
    n_inf = 55_000
    inf = pd.DataFrame({"lat": np.nan, "lon": np.nan,
                        "e": rng.choice([1, 2, 3], n_inf, p=[.7, .25, .05]), "tipo": "informal",
                        "edif": n_casas + n_edif + np.arange(n_inf), "unidad": 0})

    df = pd.concat([casas, ph, inf], ignore_index=True)
    n = len(df)
    e = df["e"].to_numpy()
    es_ph = (df["tipo"] == "ph").to_numpy()
    es_inf = (df["tipo"] == "informal").to_numpy()

    area = np.exp(rng.normal(3.75 + 0.28 * e - 0.25 * es_ph - 0.3 * es_inf, 0.42))
    hab = np.clip(np.round(area / 32 + rng.normal(0, 0.8, n)), 1, 9)
    ban = np.clip(np.round(0.5 + 0.35 * e + rng.normal(0, 0.6, n)), 1, 7)
    plantas = np.where(es_ph, 1, np.clip(rng.poisson(0.4 + 0.15 * e), 1, 3))
    piso_ubic = np.where(es_ph, 1 + (df["unidad"].to_numpy() % np.nan_to_num(df.get("pisos_edif", 2)).astype(int).clip(1)), 1)
    altura = np.where(es_ph, np.nan_to_num(df["pisos_edif"].to_numpy(), nan=2) * 3.0, plantas * 3.0)
    anio = np.clip(np.round(rng.normal(1978 + 4 * e + 12 * es_ph, 16)), 1900, 2025)
    terreno = np.where(es_ph, 0.0, area * rng.uniform(0.7, 1.6, n))

    zona = rng.integers(1, 3, n)
    sector = rng.integers(1, 99, n)
    barrio = rng.integers(1, 99, n)
    manz = rng.integers(1, 9999, n)
    terr = df["edif"].to_numpy() % 9999
    cond_dig = np.where(es_ph, 9, 0)
    npn_edif = [f"08001{z:02d}{s_:02d}00{b:02d}{m:04d}{t:04d}{c}" for z, s_, b, m, t, c in
                zip(zona, sector, barrio, manz, terr, cond_dig)]
    edif_map = pd.Series(npn_edif).groupby(df["edif"].to_numpy()).first()
    pref = edif_map.reindex(df["edif"].to_numpy()).to_numpy()
    sufijo = np.where(es_ph, [f"01{p % 100:02d}{u:04d}" for p, u in zip(piso_ubic.astype(int), df["unidad"])],
                      [f"0000{u:04d}" for u in df["unidad"]])
    npn = np.char.add(pref.astype(str), sufijo.astype(str))
    npn = np.where(es_inf, [f"0800100030000000{i:06d}{'0' * 8}" for i in range(n)], npn)

    uso = np.where(es_ph, "Residencial_Apartamentos_4_y_mas_pisos_en_PH", "Residencial_Vivienda_Hasta_3_Pisos")
    rar = rng.random(n)
    uso = np.where((~es_ph) & (rar < 0.01), "Residencial_Barracas", uso)
    uso = np.where((~es_ph) & (rar > 0.97), "Residencial_Apartamentos_4_y_mas_pisos", uso)
    tv = np.where(es_ph, 3, 4).astype(float)
    tv = np.where(rar < 0.004, 1, tv)
    cond = np.where(es_ph, "PH_Unidad_Predial", np.where(es_inf, "Informal", "NPH"))
    dest = rng.choice(["Habitacional", "Comercial", "Mixto"], n, p=[.96, .025, .015])
    tplanta = rng.choice(["Piso", "Sotano", "Mezanine"], n, p=[.985, .01, .005])
    fuente = np.where(es_ph, "propagado_matriz", np.where(es_inf, "sin_dato", "directo"))

    out = pd.DataFrame({
        "numero_predial_nacional": npn, "estrato": [ETIQ[int(v)] for v in e],
        "tipo_vivienda": tv, "destinacion_economica": dest, "condicion_predio": cond,
        "area_catastral_terreno": np.round(terreno, 2), "area_construida": np.round(area, 2),
        "uso": uso, "tipo_planta": tplanta, "planta_ubicacion": piso_ubic.astype(float),
        "altura": altura, "total_habitaciones": hab, "total_banios": ban,
        "total_plantas": plantas.astype(float), "anio_construccion": anio,
        "centroide_lon": df["lon"].round(6), "centroide_lat": df["lat"].round(6),
        "centroide_fuente": fuente,
    })

    # --- Imperfecciones conocidas de los datos reales ----------------------
    i = rng.choice(n, 200, replace=False); out.loc[i, "anio_construccion"] = 1512
    i = rng.choice(n, 3, replace=False); out.loc[i, "anio_construccion"] = 2500
    out.loc[rng.choice(n, 1), "anio_construccion"] = 20
    nph_ap = rng.choice(np.where(uso == "Residencial_Apartamentos_4_y_mas_pisos")[0], 60, replace=False)
    out.loc[nph_ap, "total_habitaciones"] = rng.integers(31, 700, 60)
    out.loc[nph_ap, "total_banios"] = rng.integers(21, 460, 60)
    out.loc[nph_ap, "area_construida"] = rng.uniform(3000, 52000, 60)
    out.loc[rng.choice(n, 60, replace=False), "area_catastral_terreno"] = rng.uniform(5e4, 1.7e6, 60)
    out.loc[rng.choice(n, 6, replace=False), "total_plantas"] = rng.integers(41, 71, 6)
    out.loc[rng.choice(np.where(es_ph)[0], 300, replace=False), "planta_ubicacion"] = 99
    # faltantes (más en informales: mecanismo MAR)
    p_nan = np.where(es_inf, 0.10, 0.01)
    out.loc[rng.random(n) < p_nan, "anio_construccion"] = np.nan
    out.loc[rng.random(n) < 0.004, "total_banios"] = np.nan
    out.loc[rng.random(n) < 0.01, "planta_ubicacion"] = np.nan
    out.loc[rng.random(n) < 0.002, "area_construida"] = 0.0
    # nulos en categóricas (en filas con estrato válido): el caso que rompe
    # SimpleImputer si llegan como pd.NA
    out.loc[rng.random(n) < 0.01, "tipo_vivienda"] = np.nan
    out.loc[rng.random(n) < 0.005, "destinacion_economica"] = np.nan
    out.loc[rng.random(n) < 0.003, "uso"] = np.nan
    # estrato no residencial y garajes/depósitos
    out.loc[rng.random(n) < 0.035, "estrato"] = rng.choice(["No_Aplica", "Otro", "No Aplica"])
    gar = rng.choice(np.where(es_ph)[0], 6000, replace=False)
    out.loc[gar, "uso"] = rng.choice(["Residencial_Garajes_En_PH", "Residencial_Depositos_Lockers"], 6000)
    # filas sin predio asociado
    sin = pd.DataFrame({c: np.nan for c in out.columns}, index=range(3183))
    sin["uso"] = "Residencial_Vivienda_Hasta_3_Pisos"
    sin["area_construida"] = rng.uniform(40, 200, 3183)
    out = pd.concat([out, sin], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)

    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(salida, index=False)
    print(f"CSV sintético: {len(out)} filas -> {salida}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/predios_sinteticos.csv")
