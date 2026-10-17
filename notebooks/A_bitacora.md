# Apéndice A. Bitácora metodológica (decisiones y hallazgos)

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

Registro cronológico de decisiones, hallazgos y su justificación. Complementa
el archivo `datos/procesados/decisiones_eda.json`, que registra las decisiones
automáticas de cada capítulo.

## A.1 Contexto

- Curso de Machine Learning (maestría), profesor Lihki Rubio. El clustering de
  tipologías es la **tesis**; este entregable es **supervisado**
  (clasificación multiclase del estrato).
- Fuente: servicio ArcGIS REST de datos abiertos de catastro de Barranquilla.

## A.2 Variable objetivo

- `estrato` (texto de dominio ArcGIS) → `estrato_num` (1–6). Filas
  `No_Aplica`/`Otro`/nulas se **excluyen** (no se imputa el objetivo: se
  inventaría la respuesta).

## A.3 Componente geoespacial

- Predio (capa 500) no tiene geometría; el terreno (315) sí. ~60 % de las
  unidades (PH) no tienen terreno propio → heredan el centroide de su edificio
  matriz (prefijo de 22 dígitos del NPN).
- Predios `Informal` sin terreno ni matriz llegan sin coordenadas (16.2 % de
  las unidades del dataset de trabajo). Un primer intento de ubicarlos por
  manzana (capa 320) se descartó porque algunos prefijos de manzana son
  genéricos. La versión actual (A.12) los ubica con el centroide de su
  manzana calculado con las viviendas ya ubicadas, solo en manzanas
  compactas; quedan sin ubicar 3.1 % de las unidades.

## A.4 Calidad de datos

- **Duplicados (resuelto):** ~29 % de filas repetidas por uniones M:N
  duplicadas → dedup de pares en tablas de relación + invariante "una fila por
  característica (objectid)". El 7 % de NPN repetidos restante son unidades
  reales (proyectos en serie, varias construcciones por lote): se conservan y
  se agrupan al partir.
- **Valores imposibles:** rangos de dominio. `anio_construccion = 1512` aparece
  exactamente 200 veces → **código centinela** de "año desconocido"; no hay
  datos reales entre el año 20 y 1899 → límite inferior 1900 justificado con
  datos. En el crudo también aparece `2500` ×809 (mismo patrón; la mayoría
  en garajes o filas sin estrato, que salen antes); en el dataset de trabajo
  quedan 129 años imposibles (0.04 %).
- **`planta_ubicacion = 99` (ejecución real, 28-sep):** el máximo y el p99.9
  son exactamente 99, mientras que el edificio más alto tiene 35 plantas → es
  un código, no un piso. Primero se fijó el rango [1, 40]; tras el diagnóstico
  de rangos (A.10) el techo pasó a 60 porque hay pisos reales hasta el 41. En
  la ejecución final se anulan 351 valores de piso (0.11 %) y 484 filas tienen
  algún valor imposible.
- Casos extremos en habitaciones/baños/área: eran mayoritariamente `NPH` +
  apartamentos, es decir, **edificios completos en una sola fila**. Tras el
  diagnóstico de rangos (A.10) ya no se anulan valor por valor: se excluyen del
  alcance como registros agregados (paso 2c del embudo). Precedentes: De Cock
  (2011) y valoración catastral masiva (boxplots + Moran local).
- **Cambio metodológico (28-sep, noche):** antes se imputaba con la mediana de
  TODO el dataset dentro de `limpieza.py`. Las instrucciones del proyecto exigen que toda decisión que
  aprende de los datos se calcule **solo con train**. Ahora `limpieza.py` solo
  convierte los valores imposibles en NaN (+ bandera) y la mediana se aprende
  dentro del `Pipeline` en cada fold. Mismo criterio del profesor Lihki (imputar con
  mediana cuando el % es bajo), aplicado en el lugar correcto.

## A.5 Orden de trabajo y partición

- Reglas fila a fila (sin estadísticas del dataset) → antes del split.
- **Split:** partición estratificada por grupos sobre bloques de 2 km × 2 km
  (1 de 5 folds = test ≈ 20 %), con la función propia determinista
  `folds_estratificados_por_grupo` (mismo algoritmo que `StratifiedGroupKFold`,
  ver A.10). Cubre a la vez desbalance (estratifica), entidades repetidas
  (edificio completo en un bloque) y autocorrelación espacial (bloques).
- Filas sin coordenadas: fuera del modelado (sesgo documentado en el cap. 3).
- **Buffer:** se decide con el correlograma de train (distancia a la que el
  Moran's I del estrato baja de 0.3, recortada a [0.25, 1] km) y se aplica
  tanto en la CV interna como entre train y test. Si el alcance supera 1 km la
  separación es parcial (compromiso para no perder demasiado entrenamiento);
  el capítulo 10 reporta la sensibilidad del F1 al buffer.
- **Transparencia:** el tamaño de bloque (2 km) y los rangos plausibles se
  fijaron con corridas exploratorias previas sobre el dataset completo, antes
  de formalizar la partición. Son reglas de diseño/dominio que no ajustan
  parámetros con las etiquetas de test, pero se declara por honestidad
  metodológica.
- **Excepción declarada al "solo train":** el análisis de patrón de puntos y
  cobertura (cap. 8.3) usa las ubicaciones de train + test (sin etiquetas),
  para que los bloques de prueba no aparezcan como huecos artificiales.

## A.6 EDA (solo train)

- Faltantes: < 30 % en todas las variables → mediana + indicador de faltante
  (mecanismo evaluado con V de Cramér, SMD y AUC de predecir el nulo).
- Outliers estadísticos: no se eliminan (sesgarían estratos altos) → log1p +
  winsorización p0.1/p99.9 aprendida en train.
- Transformación log1p para variables no negativas con |asimetría| > 1 que
  la transformación reduce.
- Categorías < 1 % → `infrequent` (OneHotEncoder, frecuencias de train).
- Multicolinealidad: VIF > 10 → se retira la variable de menor η² con el
  estrato.
- Bidimensional: Kruskal-Wallis + η²_H, χ² + V de Cramér (corregida),
  información mutua, Pearson vs Spearman (regla del profesor Lihki), corrección de Holm
  y BH.
- Multivariado: PCA, Mahalanobis robusta (MCD) + Isolation Forest (1 %),
  K-Means con silueta, t-SNE solo visual.
- Espacial: Clark-Evans, L de Ripley, DBSCAN haversine, Moran global/LISA/Gi*
  a nivel de edificio (kNN k = 8), correlograma/semivariograma,
  heterogeneidad por macrozonas, MAUP.

## A.7 Fuga de datos

- Excluidos: NPN (identificador y proxy de manzana), prefijo de edificio,
  `centroide_fuente` (metadato), estrato en texto, banderas.
- No se construye el rezago espacial del estrato (usaría la respuesta de los
  vecinos).
- AUC univariado ≥ 0.95 con validación espacial → alerta y exclusión.
- **Corrección metodológica (ejecución real, 28-sep):** el AUC univariado se
  calculaba sobre las predicciones fuera-de-fold *agrupadas* y daba AUC < 0.5
  incluso para una variable constante (`tipo_planta`). Con bloques
  espaciales, las proporciones de clase del fold de entrenamiento quedan
  anticorrelacionadas con las del fold de validación, y al agrupar folds
  aparece ese artefacto (Forman y Scholz, 2010). Ahora el AUC se calcula
  dentro de cada fold y se promedia.

## A.8 Modelo base

- Pipeline: winsorización → imputación (mediana + indicador) → log1p →
  estandarización; categóricas: 'faltante' + one-hot con `infrequent`.
- Regresión logística multinomial L2; `GridSearchCV` sobre `C` y
  `class_weight` con F1 macro y folds espaciales con buffer.
- Dos conjuntos de variables: A (físicas) y B (físicas + ubicación). El
  principal se elige por la CV, nunca por test.
- Líneas base: Dummy (mayoritaria, estratificada, uniforme) y moda por zona,
  con tamaño de zona elegido por la misma CV espacial (en la ejecución final
  resultó 2 km; en ejecuciones anteriores, 4 km).
- Test evaluado una sola vez; IC por **bootstrap de bloques**; diferencias
  pareadas; Moran de residuos ordinales; curva de aprendizaje; calibración.

## A.9 Historial de ejecuciones con los datos reales (28 a 30 de septiembre)

Las cifras de esta sección corresponden a ejecuciones anteriores y se
conservan como registro del proceso. Las vigentes están en A.12.

- **Embudo:** 382 597 → −36 497 usos no habitables → −11 sin área → −12 218
  sin estrato residencial → **333 871** unidades (321 156 predios, 168 224
  edificios). 279 739 con coordenadas (83.8 %).
- **Dependencia:** ICC = 0.991; m̃ = 179 (torre máxima 3 048 unidades);
  deff = 177.6; n efectivo ≈ 1 900.
- **Partición:** test 60 111 filas en 7 bloques (21.5 %); train 219 628 en
  34 bloques. Divergencia JS = 0.009; estrato 6: 1.3 % en test frente a 5.7 %
  en train.
- **Sesgo de cobertura:** filas sin coordenadas: 66 % estrato 1, 30 %
  estrato 2, 99.5 % informales (V = 0.42).
- **EDA:** η² de y_km 0.50, baños 0.31, piso 0.21, área 0.18. VIF < 3.
  Moran's I del estrato 0.915; alcance del correlograma 2.9 km → buffer
  1 km (excluye 36–38 % de train en promedio; 27–49 % según el fold). MAUP: I = 0.48 entre celdas de
  2 km. DBSCAN con ε = 300 m: un cluster con el 94 % de los puntos → se añade
  una rejilla de ε.
- **Modelo (primera ejecución, ahora superada por la tercera):** ambas logísticas eligen C = 0.01 y `balanced`. CV espacial:
  F1 A 0.303, B 0.314, moda por zona 0.205. Test (B): F1 macro 0.435 [IC
  0.14–0.46], accuracy 0.568, AUC 0.88, accuracy ±1 0.94, kappa 0.82. Moda
  por zona: accuracy 0.607, kappa 0.83; ΔF1 B − moda = +0.09, IC [−0.06,
  0.13], no significativo. Validación aleatoria: F1 0.62 (optimismo +0.30).
  Moran de residuos 0.67; ECE 0.16; recall del estrato 2 = 0.03. La alerta de
  accuracy ≥ 0.80 no se activó.

- **Segunda ejecución (28-sep, tarde) y regla 1-SE:** con el piso 99 como
  faltante, los capítulos 1–9 cambian solo en decimales, y DBSCAN elige
  ε = 150 m (33 clusters). En el cap. 10, la "mejor" combinación del conjunto
  B saltó de C = 0.01 balanceado (F1 CV 0.315) a C = 10 sin pesos (0.317):
  una diferencia de 0.002 frente a una desviación entre folds de ~0.09. Ese
  modelo casi sin regularizar tenía coeficientes de hasta 81 por DE y peor
  calibración (ECE 0.22). Se adopta la **regla de una desviación estándar**
  (Breiman et al., 1984; Hastie et al., 2009): se elige la combinación más
  regularizada a menos de 1 SE de la mejor.

- **Tercera ejecución del cap. 10 (30-sep, en Google Colab):** el equipo
  local bloqueó las librerías compiladas de Python (política de control de
  aplicaciones de Windows), así que el capítulo se ejecutó en Colab con la
  partición ya guardada (idéntica: 138 779 filas de train tras el buffer, 60 111
  de test). Los folds de la CV cambiaron (excluyen de 9.7 % a 70.3 % del
  entrenamiento según el fold; media 32.3 %), y con ellos las cifras de CV:
  F1 A 0.237 (C = 0.01, balanceado), B 0.230 (C = 0.01, sin pesos; la regla
  1-SE descartó el máximo C = 1 con 0.244), moda por zona 0.209 (4 km), Dummy
  0.04–0.12. Por la regla (mayor F1 en CV) el **modelo principal es A**.
  Test: A F1 0.346 [IC 0.19–0.36], accuracy 0.416, AUC 0.79, MAE 0.83,
  kappa 0.65, ECE 0.11; B F1 0.450, accuracy 0.599, AUC 0.89, MAE 0.46,
  kappa 0.82; moda por zona F1 0.346, accuracy 0.607, MAE 0.46, kappa 0.83.
  Δ F1 A − moda = −0.0002 [−0.12, 0.05]; A − B = −0.10 [−0.15, 0.05].
  Validación aleatoria 0.484 (optimismo +0.25). Moran de residuos A 0.35, B
  0.71. Lección metodológica: la CV espacial es sensible a la asignación de
  bloques a folds; diferencias de ≈0.01 entre modelos no se interpretan.

- **Alcance del clustering:** el K-Means del cap. 7 es exploratorio (instrucciones
  del proyecto, 2.4, "si procede") y DBSCAN (cap. 8) describe el patrón de puntos
  (instrucciones del proyecto, 2.7, obligatorio). Ninguno alimenta al modelo. El clustering de
  tipologías constructivas es objeto de otro trabajo (tesis).

## A.10 Revisión de rangos y ejecución final (30-sep-2026)

- **Diagnóstico de rangos** (`diagnostico_rangos.py`, sobre el train de la
  partición anterior; reporte completo en `diagnostico_rangos_reporte.txt`, en
  la raíz del libro). Motivación:
  ¿los rangos de imposibilidad eran demasiado amplios? Hallazgos:
  - la regla de Tukey (notas de clase 9.10.4.1.2) marcaba miles de viviendas
    reales (10 580 en habitaciones, 13 009 en área) → se usa para describir,
    no para limpiar;
  - los extremos de habitaciones, baños y área eran **edificios completos en
    una fila** (p. ej. 684 habitaciones y 52 168 m²), no errores de digitación;
  - había unidades de 2 a 10 m² (656 filas < 10 m²), muchas con piso "99":
    probables parqueaderos o depósitos;
  - errores visibles solo en conjunto (2 habitaciones y 82 baños en 82 m²);
  - pisos reales hasta el 41 (torres de estrato 5–6): el techo de 40 anulaba
    datos reales;
  - 2 014 viviendas con 0 habitaciones y 7 145 con 0 baños, y el año 1986 en el
    12 % de train (amontonamiento) → se documentan como limitaciones.
- **Decisiones (src/config.py, src/limpieza.py):** paso 2b, excluir área
  < 10 m² (umbral conservador para no sacar viviendas mínimas de estratos 1–2);
  paso 2c, excluir registros agregados (> 20 habitaciones, > 15 baños o
  > 2 000 m²); paso 5, incoherencias → NaN + `flag_incoherente_*` (baños >
  habitaciones + 3; < 6 m² por habitación); techo del piso 40 → 60. Con los %
  de faltantes resultantes (< 0.2 %), la imputación por mediana dentro del
  Pipeline sigue justificada (notas de clase 9.10.4.1.4).
- **Problema de reproducibilidad de la partición:** al ejecutar en Colab
  (scikit-learn 1.6.1), `StratifiedGroupKFold(shuffle=True)` produjo un test
  con 46 % de estrato 3 y casi sin estratos 4–6 (Jensen-Shannon 0.12, frente a
  0.009). **Decisión:** función propia `folds_estratificados_por_grupo`
  (mismo algoritmo voraz, con `numpy.random.default_rng` y semilla fija) para
  el test y para los folds de la CV. Reproduce la partición buena (JS 0.009)
  en cualquier versión.
- **Ejecución final** (Linux en la nube, Python 3.11, scikit-learn 1.8.0, CSV
  real): embudo 382 597 → −36 497 usos no habitables → −11 sin área → −1 044
  área < 10 m² → −234 registros agregados → −12 093 sin estrato residencial →
  **332 718** unidades (278 824 con coordenadas). Test 59 954 filas (21.5 %,
  7 bloques); train 218 870 → 138 239 tras el buffer de 1 km. Moran 0.913;
  K-Means k = 4 (V = 0.31).
- **Modelo:** el principal por CV espacial es **B** (físicas + ubicación;
  C = 0.01 sin pesos; la regla 1-SE descartó C = 0.1). CV: B 0.299 ± 0.086,
  A 0.283, moda por zona (2 km) 0.193. Test: B F1 0.450 [IC 0.16–0.46],
  accuracy 0.598, MAE 0.46, κ 0.81, AUC 0.89, ECE 0.14; moda por zona F1 0.328
  (Δ +0.12, IC [−0.01, 0.19], no significativo); A F1 0.349. Validación
  aleatoria 0.602 (optimismo +0.30). Moran de residuos 0.70 (B). Estrato 2:
  recall 0.06 (83 % → 3).
- **Caché de dominios ArcGIS:** el servidor no respondió desde el entorno de
  la ejecución final; `datos/dominios_arcgis.json` se reconstruyó con los
  alias y el dominio de `tipo_vivienda` de la descarga original (28-sep).

## A.11 Corrección del grupo del ICC (9-oct-2026)

- **Problema:** el ICC y el efecto de diseño agrupaban por los 22 primeros dígitos del NPN. Con condición de
  propiedad 5 (*mejoras en terreno ajeno*) ese prefijo reúne viviendas informales independientes: el mayor
  «edificio» (3 045 unidades, sector 11, manzana 0615, sin coordenadas) era un lote del suroccidente.
- **Corrección:** cada mejora cuenta como su propio grupo; el resto sigue agrupado por predio matriz.
- **Antes:** m̃ = 179.36; ICC = 0.991; deff = 177.76; n efectivo ≈ 1 872 (máximo 3 045).
- **Después:** 182 524 grupos; m̃ = 145.42; ICC = 0.992; deff = 144.27; n efectivo ≈ 2 306 (máximo 1 860,
  un conjunto de 93 torres). Solo con coordenadas: n efectivo ≈ 1 623. Las conclusiones no cambian.
- Cifras verificadas reproduciendo la limpieza en Python puro (se obtuvieron de nuevo las cifras anteriores
  con la definición vieja). La partición y el modelo no usan este agrupamiento.

## A.12 Dos universos de datos y reorganización del libro (16-oct-2026)

- **Pregunta:** varias estadísticas del capítulo 1 (clases, tamaño, ICC) se
  calculaban con las 332 718 filas limpias, mientras que todo lo demás usa las
  278 824 con coordenadas. ¿Con cuál universo reportar?
- **Hallazgo:** las 53 894 filas sin coordenadas son casi todas mejoras en
  terreno ajeno (condición 5) de estratos 1 y 2: la cobertura de coordenadas
  es 67.2 % en el estrato 1 y ≥ 97.7 % en los estratos 3–6. Quitarlas baja el
  estrato 1 de 32.8 % a 26.3 %. No faltan al azar.
- **Decisión:** se reportan los dos universos. El capítulo 1 describe la base
  con las 332 718 filas y muestra al lado las cifras con coordenadas (clases,
  duplicados, faltantes, n/p, ICC); desde el capítulo 2 se modela con las
  278 824. Nueva sección 1.7.
- **Alternativa evaluada (aplicada después, ver A.13):** el 92.7 % de las filas sin
  coordenadas (49 981) comparte manzana (17 primeros dígitos del NPN) con
  viviendas ubicadas, con una dispersión típica de manzana de ~30 m. Con el
  centroide de la manzana el universo de modelado pasaría al 98.8 %. Se deja
  para el Entregable 2 porque cambia la partición y todos los resultados.
- **Reorganización:** nueva Guía de lectura (conceptos con ejemplos y
  recorrido de los datos); el capítulo 1 se reordenó para explicar la
  estructura del NPN y del predio matriz antes de la limpieza, y cada regla
  junto al embudo; cada capítulo abre con un cuadro "Qué hace este capítulo"
  y explica sus métodos con ejemplos antes de usarlos.
- **Reejecución:** el libro completo se volvió a ejecutar (Python 3.13,
  scikit-learn 1.9). La partición y las cifras del EDA y del modelo se
  reproducen; solo cambian en la cuarta cifra decimal algunos Moran (p. ej.
  0.2183 → 0.2179 en área construida).


## A.13 Ubicación por manzana de las viviendas sin coordenadas (16-oct-2026)

- **Motivo:** el 92.7 % de las 53 894 filas sin coordenadas (casi todas de
  estratos 1 y 2) comparte manzana (17 primeros dígitos del NPN) con
  viviendas ubicadas. Ubicarlas reduce el sesgo de cobertura, que era la
  principal reserva del veredicto (11.4).
- **Por qué no se hizo antes sin más:** algunos prefijos de manzana son
  genéricos. La manzana `0000` reúne hasta 820 edificios repartidos en 7 km,
  y hay manzanas cuyos puntos ubicados están a más de 600 m del centroide.
  Ponderando por filas recuperables, ~10 % habría caído en esos prefijos.
- **Regla:** centroide de los puntos distintos ya ubicados de la manzana,
  solo si el código no es `0000` y todos sus puntos están a ≤ 250 m del
  centroide (`src/config.py`, `limpieza.recuperar_por_manzana`, paso 6 del
  embudo). Usa coordenadas de otras filas, nunca el estrato.
- **Resultado:** 43 594 filas ubicadas (80.9 % de las sin coordenadas);
  universo de modelado 83.8 % → 96.9 %; quedan 10 300 (3.1 %): 3 913 en
  manzanas sin viviendas ubicadas, 3 502 en manzanas dispersas y 2 885 con
  manzana `0000`. Error de posición dejando fuera cada edificio ubicado:
  mediana 31 m, p90 68 m, p99 122 m.
- **Análisis espacial:** las viviendas de una manzana recuperada comparten
  punto; en Moran, LISA, correlograma y Moran de residuos cuentan como un
  punto por manzana (`espacial.deduplicar_por_edificio`).
- **Efecto en la partición:** la grilla es la misma, pero cambian los conteos
  por bloque y la asignación estratificada: solo 3 de los 7 bloques de test
  coinciden con los de A.10. Test 65 145 filas (20.2 %); train 257 273 →
  144 586 tras el buffer de 1 km.
- **EDA:** desbalance 6.2:1 (estrato 1 mayoritario); V de Cramér de cobertura
  0.42 → 0.22; Moran 0.918; alcance del correlograma 2.6 km (buffer sigue en
  1 km); K-Means k = 5, con un clúster de viviendas informales sin terreno;
  13.5 % de train con terreno 0 (mejoras en terreno ajeno).
- **Modelo:** B con C = 0.01 y `class_weight="balanced"` (regla 1-SE en A y
  en B). CV: B 0.322 ± 0.085, A 0.286, moda por zona (4 km) 0.187. Test: B F1
  0.492 [IC 0.28–0.51], accuracy 0.611, MAE 0.44, κ 0.84, AUC 0.90,
  ECE 0.03; B − A +0.12 [0.03, 0.15]; B − moda por zona +0.14 [−0.12, 0.17].
  Validación aleatoria 0.620. Moran de residuos 0.56 (B). El estrato que más
  falla es el 4 (recall 0.28; 48 % → 3); el 2 se recupera (recall 0.60).
- El tamaño de zona elegido para la moda por zona se guarda ahora en
  `resultados_modelo.json` (`tam_zona_moda_km`) para que el Entregable 2 use
  el mismo.

## A.14 Pendientes

- (Hecho el 30-sep) Ejecutar el capítulo 10 con la regla 1-SE y actualizar
  textos, conclusiones y resumen.
- Opcional: calcular el bootstrap pareado B − moda por zona (hoy solo se prueba
  el modelo principal contra las bases).
- Verificar licencia y fecha de descarga.
- Verificar con el diccionario del catastro: terreno de unidades PH (cuota
  del lote) y si 0 habitaciones es un valor válido.
