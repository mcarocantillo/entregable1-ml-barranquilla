# Resumen ejecutivo del EDA

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

Este resumen sintetiza los capítulos 3 a 9, siguiendo la estructura que piden
las notas de clase para el cierre del EDA (9.10.4.1.5): calidad de los datos,
variables más prometedoras, problemas detectados y decisiones preliminares.
Todas las cifras son de **entrenamiento** (el test no se usó en el EDA), salvo
que se indique lo contrario.

## Calidad de los datos

| Aspecto | Hallazgo | Cap. |
|---|---|---|
| Tamaño | 332 718 unidades de vivienda en 168 044 edificios (dataset completo); ~9 500 filas por columna tras one-hot | 1 |
| Valores faltantes | Mínimos: máximo 0.12 % (`total_habitaciones`). No vienen de celdas vacías, sino de las reglas de limpieza (códigos centinela e incoherencias). Mecanismo **MAR** en habitaciones, piso y antigüedad | 3 |
| Valores imposibles | Años centinela (2500, 1512) y pisos codificados (97–99) pasan a faltante; unidades < 10 m² y registros agregados se excluyen | 1 |
| Outliers univariados | Tukey marca 3–6 % en áreas y conteos, pero son viviendas grandes reales, no errores; la regla es inservible en plantas y altura (IQR = 0) | 3 |
| Outliers multivariados | 0.57 % marcados por Mahalanobis robusta e Isolation Forest a la vez; son combinaciones raras (0 habitaciones, 0 baños) y están sobrerrepresentados en estratos 5–6 | 7 |
| Cobertura | El 16.2 % de las filas no tiene coordenadas; el 99.5 % de ellas son predios informales y el 96.5 % son estratos 1–2 (V de Cramér = 0.42) | 3, 8 |

## Variables más prometedoras

| Variable | Asociación con el estrato | Cap. |
|---|---|---|
| `y_km` (eje norte-sur) | η² = 0.50 (efecto grande); AUC univariado en zonas no vistas = 0.76 | 6, 9 |
| `total_banios` | η² = 0.31; AUC = 0.69 | 6, 9 |
| `dist_centro_km`, `x_km` | η² = 0.32 y 0.26 | 6 |
| `planta_ubicacion` | η² = 0.21 | 6 |
| `area_construida` | η² = 0.18; mediana de 54 m² en estrato 1 frente a 152 m² en estrato 6 | 6 |
| `condicion_predio`, `uso`, `tipo_vivienda` | V de Cramér = 0.33, 0.24 y 0.22 | 6 |

La **ubicación domina**: el estrato tiene una autocorrelación espacial
extremadamente fuerte (Moran's I = 0.913) y su media sube de 1.55 en el sur a
4.08 en el norte (cap. 4 y 8). En PCA, el eje más asociado con el estrato es
el de **tipología** (apartamento moderno en altura frente a casa con lote), no
el de tamaño (cap. 7).

## Problemas detectados

- **Desbalance moderado (4.6 : 1).** La clase minoritaria (estrato 6) tiene
  12 390 casos en train: no hace falta sobremuestreo, pero el accuracy no
  sirve como métrica (cap. 4).
- **Dependencia entre filas.** El estrato casi no varía dentro de un edificio
  (ICC = 0.991): el tamaño efectivo es ≈ 1 870, no 332 718 (cap. 1).
- **Dependencia espacial.** Moran's I = 0.913 y alcance del correlograma de
  2.9 km: una validación aleatoria sería muy optimista (cap. 8).
- **Asimetría fuerte.** Todas las variables físicas tienen |asimetría| > 1
  (terreno: 51.7) (cap. 5).
- **Relaciones no monótonas y que cambian por zona.** El terreno sube y luego
  baja con el estrato, y la correlación área–estrato va de 0.07 a 0.48 según
  la macrozona (cap. 6 y 8.5).
- **Redundancia entre categóricas.** `uso` y `condicion_predio` codifican
  ambas el régimen de propiedad horizontal. Entre numéricas **no hay
  multicolinealidad preocupante**: todos los VIF < 3 (5.4 al añadir las
  espaciales) (cap. 6).
- **Variables sin información.** `destinacion_economica` y `tipo_planta` son
  degeneradas; `altura` es casi constante (vale 3 en el 99.7 % de las filas)
  (cap. 5).
- **Fuga de datos: no se encontró.** Ninguna variable se acerca a la alerta de
  AUC univariado de 0.95 (máximo 0.76), y train y test no comparten
  edificios, predios ni coordenadas (cap. 9).

## Decisiones preliminares para el preprocesamiento y la validación

Todas quedan registradas en `datos/procesados/decisiones_eda.json`, y el
capítulo 10 construye el `Pipeline` leyendo ese archivo.

| Decisión | Hallazgo que la motiva | Cap. |
|---|---|---|
| Partición por **bloques espaciales de 2 km**, con **buffer de 1 km** | Moran's I = 0.913; alcance de 2.9 km; ICC = 0.991 | 2, 8 |
| Métrica principal **F1 macro**, más MAE ordinal, accuracy ±1 y kappa cuadrático | Desbalance 4.6 : 1; estrato ordinal | 4 |
| Sin sobremuestreo; comparar `class_weight=None` frente a `"balanced"` | La clase minoritaria tiene 12 390 casos | 4 |
| Imputación: mediana + indicador de faltante (numéricas); categoría `faltante` (categóricas) | Faltantes < 1 %, mecanismo MAR | 3 |
| `log1p` en 7 variables; `altura` lineal | \|asimetría\| > 1 que `log1p` reduce | 5 |
| Winsorización a los percentiles 0.1 y 99.9 de train; no se eliminan outliers | Las viviendas grandes son reales y pesan más en los estratos altos | 3, 7 |
| Categorías con < 1 % agrupadas en `infrequent` | Categorías raras en `uso` y `condicion_predio` | 5 |
| Variables del modelo: 8 numéricas, 3 categóricas (V ≥ 0.1) y 3 espaciales | Tamaños de efecto, VIF < 10 y auditoría de fuga | 6, 8, 9 |
| Se excluyen `destinacion_economica`, `tipo_planta`, identificadores y `centroide_fuente` | Degeneradas, identificadores o metadatos | 5, 9 |
