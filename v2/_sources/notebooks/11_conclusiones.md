# 11. Conclusiones, limitaciones y próximos pasos

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

## 11.1 Hallazgos principales

1. **Base de datos.** El catastro abierto de Barranquilla permitió construir
   una base de **332 718 unidades de vivienda** con estrato residencial válido
   (87.0 % de las 382 597 filas descargadas), en 320 792 predios y 168 044
   predios matriz (edificios, conjuntos o lotes). De ellas, **322 418 (96.9 %)
   tienen coordenadas** (278 824 propias y 43 594 ubicadas por su manzana) y
   son el universo con el que se parte y se modela; las 10 300 restantes son
   casi todas mejoras en terreno ajeno de estrato 1, y el capítulo 1 reporta
   las cifras clave en los dos universos (sección 1.7). Durante la construcción se corrigió un error de uniones
   muchos-a-muchos que duplicaba ~29 % de las filas. La limpieza distingue
   tres tipos de problema. (i) **Códigos centinela**: año de construcción
   2500 y 1512, y pisos 97–99; se tratan como faltantes. (ii) **Filas que no
   describen una vivienda**: unidades de menos de 10 m², probables
   parqueaderos o depósitos (1 044 filas), y **registros agregados** que
   describen un edificio completo en una sola fila (234 filas); se excluyen
   del alcance. (iii) **Incoherencias dentro de la fila**: más baños que
   habitaciones + 3, o menos de 6 m² por habitación (396 filas); se anula
   el valor y se conserva la fila. La regla de Tukey de las notas de clase se
   usa para **describir** outliers (cap. 3), no para limpiar: en variables de
   cola larga marca miles de viviendas reales.
2. **Ubicación por manzana.** El 16.2 % de las viviendas (casi todas mejoras
   informales en terreno ajeno, de estratos 1 y 2) llega sin coordenadas. Su
   número predial dice en qué manzana están, y el 92.7 % comparte manzana con
   viviendas ubicadas. Se ubicaron 43 594 de ellas con el **centroide de su
   manzana**, solo en manzanas compactas (código distinto de `0000` y todos
   sus puntos a menos de 250 m del centroide). El error de esa aproximación,
   medido con los edificios ya ubicados, es de **31 m** en la mediana (p90 de
   68 m), muy poco frente a los bloques de 2 km y al buffer de 1 km. Con ello,
   el **96.9 %** de las viviendas entra al modelado (antes, el 83.8 %), y la V
   de Cramér entre "tener coordenadas" y el estrato baja de 0.42 a 0.22
   (sección 1.7.2 y 3.3).
3. **Estructura de dependencia.** El estrato casi no varía dentro de un
   edificio o conjunto (ICC = 0.992), y hay conjuntos de hasta 1 860 unidades: la
   información sobre el estrato está gobernada por edificios, conjuntos y zonas, no por
   filas (tamaño efectivo ≈ 2 300 con todas las filas y ≈ 2 170 con las que
   tienen coordenadas). Esto obliga a partir los datos por grupos espaciales y
   a medir la incertidumbre remuestreando bloques.
4. **Objetivo.** El estrato está moderadamente desbalanceado (6.2:1 en
   train; el estrato 1 es la clase mayoritaria), es ordinal y tiene una
   **autocorrelación espacial extremadamente fuerte** (Moran's I = 0.918 a
   nivel de edificio; casi la mitad de los edificios en clusters LISA y
   prácticamente ningún outlier espacial). La ciudad está segregada: estrato
   medio 1.5 en la franja sur frente a 3.9 en la norte.
5. **Predictoras.** La variable más asociada al estrato es la **posición
   norte-sur** (η² = 0.46). Entre las físicas destacan el número de **baños**
   (η² = 0.33), el piso de ubicación (0.25) y el **área construida** (0.17),
   con relaciones monótonas pero no lineales (justifican log1p). El régimen de
   propiedad (`condicion_predio`, V = 0.33) y el `uso` (0.26) aportan
   información, pero son redundantes entre sí. `destinacion_economica` y
   `tipo_planta` son degeneradas y se excluyeron. La relación entre variables
   físicas y estrato **cambia por zona** (Spearman área–estrato de 0.12 a
   0.49 según la macrozona).
6. **Modelo base.** El modelo principal, elegido por F1 macro en la
   validación cruzada espacial, es la regresión logística multinomial con
   variables físicas **y** de ubicación (modelo B; C = 0.01,
   `class_weight="balanced"`). Alcanza **F1 macro 0.32 ± 0.08 en validación
   cruzada espacial** y **0.49 en test** (IC 95 % por bloques [0.28, 0.51]; 7
   bloques no vistos), frente a 0.087 de la clase mayoritaria (Δ = +0.405, IC
   [0.10, 0.44]). Los errores son casi siempre entre estratos adyacentes
   (accuracy ±1 = 0.96, MAE ordinal = 0.44, kappa cuadrático = 0.84). El
   estrato que más falla es el **4** (recall 0.28): el 48 % se clasifica
   como 3 y el 22 % como 5. Los estratos 5 y 6 se confunden entre sí y con
   el 4 (el 6 acierta el 21 %).
7. **La ubicación manda, pero no basta una regla geográfica.** Una línea base
   puramente geográfica, la **clase más frecuente en una zona de 4 km** (el
   tamaño elegido por CV), queda por debajo del modelo en todas las métricas
   de test (F1 0.35 frente a 0.49; MAE 0.52 frente a 0.44; kappa 0.77 frente
   a 0.84), aunque la diferencia de F1 no es significativa (+0.14, IC
   [−0.12, 0.17]). B supera al modelo con solo variables físicas (A) en test
   (F1 0.49 frente a 0.37; IC de la diferencia [0.03, 0.15], que excluye 0),
   pero en la CV espacial la ventaja es menor y queda dentro del ruido (0.322
   frente a 0.286). Lo que aportan las coordenadas depende de la zona: en uno
   de los folds de la CV, desordenarlas **mejora** el F1.
8. **Validación.** Una validación aleatoria habría estimado un F1 macro de
   **0.62**, frente a **0.32** con bloques y buffer: casi el doble. El buffer
   también importa: el F1 de CV baja de 0.41 sin buffer a 0.32 con 1 km, una
   caída del orden de la variabilidad entre folds. En este problema, la
   validación espacial no es opcional. Además, la partición y los folds
   dependían de la versión de scikit-learn (con `StratifiedGroupKFold`, un
   entorno produjo un test casi sin estratos 4–6). Por eso se construyen con
   una función propia determinista (`folds_estratificados_por_grupo`), y las
   diferencias de ≈0.01–0.04 entre modelos en CV no se interpretan.
9. **Residuos y calibración.** Los residuos conservan una autocorrelación
   espacial fuerte (Moran = 0.56 en B y 0.47 en A): el modelo capta la
   tendencia de gran escala, pero no los barrios. En conjunto, las
   probabilidades están bien calibradas (ECE = 0.03), pero no por clase: con
   los pesos de clase, el estrato 1 y el 4 quedan subestimados y el 2, el 3,
   el 5 y el 6 sobreestimados, lo que explica que el 4 pierda casi siempre
   frente al 3 pese a ordenar bien (AUC 0.93). La curva de aprendizaje es
   plana (F1 de 0.325 a 0.322 al multiplicar por 20 los datos): más filas no
   ayudarán, hacen falta un modelo más flexible y variables del entorno.

## 11.2 Limitaciones

- **Cobertura:** el 3.1 % de las filas (10 300 viviendas, 98 % informales y
  84 % de estrato 1) sigue sin coordenadas y queda fuera del modelo: sus
  manzanas no tienen viviendas ubicadas, son dispersas o tienen el código
  genérico `0000`. Quitarlas baja el estrato 1 de 32.8 % a 31.1 % (sección
  1.7.3). Las conclusiones sobre esos asentamientos deben leerse con cautela.
- **Ubicación aproximada de las viviendas informales:** las 43 594 viviendas
  ubicadas por su manzana están en el centroide de la manzana, no en su
  propio lote (error mediano estimado de 31 m), y todas las de una manzana
  comparten punto. Los análisis espaciales las cuentan como un punto por
  manzana, pero cualquier variable que dependa de la posición exacta dentro
  de la manzana no es fiable para ellas.
- **Pocos bloques de test:** el test quedó con 7 bloques, con poco estrato 6
  (1.2 % frente a 4.8 % en train). Los intervalos son muy anchos y la cifra de
  test es una realización favorable; la estimación más representativa es la
  de la CV espacial.
- **CV sensible a los folds:** con pocos bloques, la asignación de bloques a
  folds cambia la cifra de CV en varias centésimas. Por eso se fijó una
  partición determinista y se reportan desviaciones e intervalos. La
  ubicación por manzana cambió los conteos por bloque y, con ellos, la
  partición: solo 3 de los 7 bloques de test coinciden con los de la versión
  anterior del libro, así que sus cifras de test no se comparan una a una.
- **Buffer parcial:** el alcance del correlograma (2.6 km) supera el buffer
  usado (1 km). La validación espacial sigue siendo algo optimista.
- **Estrato como etiqueta administrativa:** se asigna por manzana o lado de
  manzana con una metodología que incluye el entorno urbano; el modelo solo ve
  la vivienda y su ubicación aproximada.
- **Calidad del catastro:** las características físicas pueden estar
  desactualizadas. El año de construcción aparece amontonado en ciertos
  valores (1986 concentra el 12 % de train en el diagnóstico de rangos), y hay
  viviendas con 0 habitaciones o 0 baños que no se pudieron verificar contra
  el diccionario oficial; se conservaron.
- **Significado de `anio_construccion` (revisión posterior, octubre de 2026):**
  el campo parece registrar el año de la última incorporación o actualización
  de la construcción en el catastro, no su fecha original. Solo 1.442 casas de
  toda la ciudad tienen años anteriores a 1975, lo que no es compatible con
  barrios antiguos como El Prado, Barrio Abajo o el Centro. Además, 15.376
  apartamentos de estratos 1 y 2, en 24 lotes de bloques de 5 pisos con
  tipología de vivienda de interés prioritario, registran años entre 1932 y
  1972, con un solo año por lote. Como el filtro de plausibilidad aceptaba
  años desde 1900, esas unidades entraron al modelo con 54 a 94 años de
  `antiguedad`. La variable debe leerse como «años desde el registro
  catastral» y no como edad física del edificio. Queda pendiente confirmar su
  definición con la Gerencia de Gestión Catastral. El detalle está en el
  [sitio de datos de la tesis](https://mcarocantillo.github.io/tesis-datos-barranquilla/11_coherencia.html).
- **Modelo lineal:** no captura interacciones ni heterogeneidad espacial; la
  multinomial ignora el orden de las clases, y aunque la calibración global
  es buena (ECE = 0.03), las probabilidades por clase están desplazadas por
  los pesos de clase.
- **Riesgos éticos:** reidentificación por NPN + coordenadas y uso
  discriminatorio del estrato (sección 1.11).

## 11.3 Próximos pasos (Entregable 2)

- Modelos no lineales (árboles, *gradient boosting*) y **regresión logística
  ordinal**, que respeta el orden de las clases y debería ayudar con los
  estratos intermedios y altos (4 y 5), hoy absorbidos por sus vecinos.
- Recalibrar probabilidades por clase (Platt o isotónica, dentro de la CV
  espacial) y ajustar umbrales por clase: el estrato 4 ordena bien (AUC 0.93)
  pero casi nunca gana el argmax.
- Variables de **vecindad física**: rezago espacial de área, baños y
  antigüedad de los vecinos. Son legítimas porque no usan el estrato. Con la
  ubicación por manzana, estas variables también cubren los barrios
  informales.
- Revisar `altura` (casi constante) y la antigüedad, que no aportan en la
  importancia por permutación.
- Más folds o bloques más pequeños para un test menos dependiente de pocas
  zonas, y comparar tamaños de bloque y buffer como hiperparámetros de la
  evaluación (notas de clase 9.1.11).
- Verificar con el diccionario del catastro los ceros de habitaciones y
  baños y el amontonamiento de años.
- Ubicar el 3.1 % que sigue sin coordenadas con la cartografía oficial de
  manzanas o con geocodificación de direcciones, y, si se consigue la
  geometría del lote de las mejoras informales, reemplazar el centroide de
  la manzana por la posición real.

## 11.4 Veredicto: ¿la base de datos sirve para el proyecto?

Para cerrar, se revisa si la base cumple lo que pide el proyecto, contrastando
cada requisito con un score calculado en los capítulos anteriores. Cada
criterio indica de dónde sale: las **instrucciones del proyecto (Entregable 1)** (secciones 1,
2.3, 2.5 y 3, y la Figura 1) o las **notas de clase** del profesor Lihki Rubio
(sección 9). Donde ninguna fija un número, se usa el umbral que el propio EDA
declaró antes de modelar.

| # | Criterio | Fuente | Umbral | Score obtenido | Cap. | ¿Cumple? |
|---|---|---|---|---|---|---|
| 1 | Tamaño de la muestra | Instrucciones 1 | ≥ 20 000 observaciones | 332 718 viviendas (322 418 con coordenadas) | 1 | Sí |
| 2 | Relación n/p (filas por cada columna del modelo; ver 1.8) | Instrucciones 1 | reportar; n ≫ p | ~9 500 filas por columna (~9 200 con coordenadas) | 1 | Sí |
| 3 | Casos de la clase minoritaria | Instrucciones 1 y 2.1 | suficientes sin sobremuestreo | 12 414 (estrato 6, train); desbalance 6.2 : 1 | 4 | Sí |
| 4 | Entidades independientes | Instrucciones 1 | reportar el tamaño efectivo | 168 044 predios matriz (165 954 con coordenadas); ICC = 0.992; tamaño efectivo ≈ 2 300 (≈ 2 170 con coordenadas) | 1 | Sí, con partición por bloques |
| 5 | Valores faltantes | Notas 9.10.4.1.2 | < 30 % por variable | máximo 0.12 %; mecanismo MAR | 3 | Sí |
| 6 | Sesgo de muestreo y representatividad | Instrucciones 1 | sin sesgo de cobertura | 3.1 % sin coordenadas tras ubicar por manzana (16.2 % antes), casi todo informal y de estrato 1; V = 0.22 con el estrato (0.42 antes) | 1, 3, 8 | Con reservas |
| 7 | Asociación predictoras–objetivo (tamaño de efecto) | Instrucciones 2.3 | al menos un efecto grande (η² > 0.14, umbral del cap. 6) | η² = 0.46 (`y_km`), 0.33 (baños), 0.25 (piso) | 6 | Sí |
| 8 | Multicolinealidad | Instrucciones 2.3 | VIF ≤ 10 (regla del cap. 6) | VIF < 2 en las físicas; 6.1 con las espaciales | 6 | Sí |
| 9 | Fuga de datos | Instrucciones 2.5 | AUC univariado lejos de 1 (alerta en 0.95); 0 entidades compartidas | AUC máximo 0.75; 0 edificios, predios o coordenadas compartidos | 9 | Sí |
| 10 | Comparación con la línea base trivial | Instrucciones 3; notas 9.10.4.6.2 | IC 95 % bootstrap de la diferencia excluye 0 | Δ F1 macro = +0.405 frente al `DummyClassifier`, IC [0.10, 0.44], p = 0.012 | 10 | Sí |
| 11 | AUC | Instrucciones 3; notas 9.4 | mayor que el azar (0.5) | AUC one-vs-rest macro = 0.90 (test) | 10 | Sí |
| 12 | ¿El problema es trivial? | Instrucciones, Figura 1 y nota crítica | accuracy < 80–90 % | accuracy = 0.43 en CV espacial y 0.61 en test | 10 | Sí (no es trivial) |
| 13 | Tamaño de muestra suficiente | Instrucciones 3 (curva de aprendizaje) | curva plana al aumentar los datos | F1 de 0.325 a 0.322 al multiplicar por 20 los datos | 10 | Sí |
| 14 | Residuos sin dependencia espacial | Instrucciones 3 | Moran de residuos cercano a 0 | Moran = 0.56 | 10 | Con reservas |

**Veredicto: sí, la base de datos es adecuada para el proyecto, con dos
reservas.** Cumple 12 de los 14 criterios.

- **Base de datos (criterios 1–6).** Es grande, tiene casos de sobra en todas
  las clases y casi no tiene faltantes. Su tamaño efectivo es mucho menor que
  el número de filas, pero eso no la invalida: obliga a partir por bloques y
  a remuestrear bloques, que es lo que hace el libro.
- **EDA (criterios 7–9).** Las predictoras contienen información real sobre
  el estrato (efectos grandes), no son redundantes entre sí y ninguna es un
  proxy del objetivo.
- **Modelo base (criterios 10–13).** El modelo supera con significancia a la
  línea base trivial. La pregunta de control de la Figura 1 de las instrucciones del proyecto
  ("¿accuracy ≥ 80–90 %?") da **no**: el problema no es trivial, no hay
  señales de fuga y el dataset es suficientemente desafiante para el curso.
  La curva de aprendizaje muestra que la cantidad de datos no es el límite.

Las dos reservas son:

- **Representatividad (criterio 6).** Tras ubicar por manzana a la mayoría
  de las viviendas informales, la base describe casi toda la ciudad, pero el
  3.1 % que sigue sin coordenadas es casi todo informal y de estrato 1, y las
  viviendas ubicadas por su manzana tienen una posición aproximada. Las
  conclusiones sobre los asentamientos que no se pudieron ubicar deben
  leerse con cautela.
- **Residuos con dependencia espacial (criterio 14).** Según las instrucciones del proyecto, esto
  indica información estructural sin capturar. Es una limitación del modelo
  lineal más que de la base: el modelo capta la tendencia norte-sur, pero no
  los barrios. Lo mismo explica que el F1 macro sea modesto (0.32 en CV
  espacial). Variables de vecindad y modelos más flexibles quedan para el
  Entregable 2 (11.3), con la misma base.

En resumen, la base no necesita más filas ni otra fuente para seguir con el
proyecto; necesita mejores variables derivadas (vecindad física) y modelos
más flexibles.
