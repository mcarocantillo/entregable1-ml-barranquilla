# 11. Conclusiones, limitaciones y próximos pasos

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

## 11.1 Hallazgos principales

1. **Base de datos.** El catastro abierto de Barranquilla permitió construir
   una base de **332 718 unidades de vivienda** con estrato residencial válido
   (87.0 % de las 382 597 filas descargadas), en 320 792 predios y 168 044
   edificios. Durante la construcción se corrigió un error de uniones
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
2. **Estructura de dependencia.** El estrato casi no varía dentro de un
   edificio (ICC = 0.991), y hay torres de hasta 3 045 unidades: la
   información sobre el estrato está gobernada por edificios y zonas, no por
   filas (tamaño efectivo ≈ 1 870). Esto obliga a partir los datos por grupos
   espaciales y a medir la incertidumbre remuestreando bloques.
3. **Objetivo.** El estrato está moderadamente desbalanceado (4.6:1 en
   train), es ordinal y tiene una **autocorrelación espacial extremadamente
   fuerte** (Moran's I = 0.913 a nivel de edificio; casi la mitad de los
   edificios en clusters LISA y prácticamente ningún outlier espacial). La
   ciudad está segregada: estrato medio 1.5 en la franja sur frente a 4.1 en
   la norte.
4. **Predictoras.** La variable más asociada al estrato es la **posición
   norte-sur** (η² = 0.50). Entre las físicas destacan el número de **baños**
   (η² = 0.31), el piso de ubicación (0.21) y el **área construida** (0.18),
   con relaciones monótonas pero no lineales (justifican log1p). El régimen de
   propiedad (`condicion_predio`, V = 0.33) y el `uso` (0.24) aportan
   información, pero son redundantes entre sí. `destinacion_economica` y
   `tipo_planta` son degeneradas y se excluyeron. La relación entre variables
   físicas y estrato **cambia por zona** (Spearman área–estrato de 0.07 a
   0.48 según la macrozona).
5. **Modelo base.** El modelo principal, elegido por F1 macro en la
   validación cruzada espacial, es la regresión logística multinomial con
   variables físicas **y** de ubicación (modelo B; C = 0.01, sin pesos de
   clase). Alcanza **F1 macro 0.30 ± 0.09 en validación cruzada espacial** y
   **0.45 en test** (IC 95 % por bloques [0.16, 0.46]; 7 bloques no vistos),
   frente a 0.075 de la clase mayoritaria (Δ = +0.375, IC [0.11, 0.39]). Los
   errores son casi siempre entre estratos adyacentes (accuracy ±1 = 0.94,
   MAE ordinal = 0.46, kappa cuadrático = 0.81). Dos desplazamientos hacia el
   centro de la escala explican la mayoría de los errores: el **estrato 2
   casi no se predice** (recall 0.06; el 83 % se clasifica como 3) y el 56 %
   de los estratos 5 y 6 se predice como 4.
6. **La ubicación manda, pero no basta una regla geográfica.** Una línea base
   puramente geográfica, la **clase más frecuente en una zona de 2 km**, queda
   por debajo del modelo en todas las métricas de test (F1 0.33 frente a
   0.45; MAE 0.94 frente a 0.46; kappa 0.43 frente a 0.81), aunque la
   diferencia de F1 no es significativa (+0.12, IC [−0.01, 0.19]). La
   ubicación es lo que más aporta al modelo: las tres variables espaciales
   encabezan la importancia por permutación, y B supera al modelo con solo
   variables físicas (A) en test (F1 0.45 frente a 0.35; MAE 0.46 frente a
   0.82). Esa ventaja tampoco es significativa en F1 (IC [−0.05, 0.14]) y en
   la CV espacial es mínima (0.299 frente a 0.283).
7. **Validación.** Una validación aleatoria habría estimado un F1 macro de
   **0.60**, frente a **0.30** con bloques y buffer: el doble. El buffer
   también importa: el F1 de CV baja de 0.40 sin buffer a 0.30 con 1 km, una
   caída mayor que la variabilidad entre folds. En este problema, la
   validación espacial no es opcional. Además, la partición y los folds
   dependían de la versión de scikit-learn (con `StratifiedGroupKFold`, un
   entorno produjo un test casi sin estratos 4–6). Por eso se construyen con
   una función propia determinista (`folds_estratificados_por_grupo`), y las
   diferencias de ≈0.01–0.02 entre modelos en CV no se interpretan.
8. **Residuos y calibración.** Los residuos conservan una autocorrelación
   espacial fuerte (Moran = 0.70 en B y 0.34 en A): el modelo capta la
   tendencia de gran escala, pero no los barrios. Las probabilidades están
   regularmente calibradas (ECE = 0.14): el estrato 3 está sobreestimado y el
   2 subestimado, lo que explica que el 2 pierda casi siempre frente al 3 pese
   a ordenar bien (AUC 0.86). La curva de aprendizaje está casi plana (F1 de
   0.27 a 0.29 al multiplicar por 20 los datos): más filas no ayudarán, hacen
   falta un modelo más flexible y variables del entorno.

## 11.2 Limitaciones

- **Cobertura:** el 16.2 % de las filas (predios informales, 96.5 % en estratos
  1–2) no tiene coordenadas y queda fuera del modelo; las conclusiones
  aplican a la ciudad formal.
- **Pocos bloques de test:** el test quedó con 7 bloques, con poco estrato 6
  (1.3 % frente a 5.7 % en train). Los intervalos son muy anchos y la cifra de
  test es una realización favorable; la estimación más representativa es la
  de la CV espacial.
- **CV sensible a los folds:** con pocos bloques, la asignación de bloques a
  folds cambia la cifra de CV en varias centésimas. Por eso se fijó una
  partición determinista y se reportan desviaciones e intervalos.
- **Buffer parcial:** el alcance del correlograma (2.9 km) supera el buffer
  usado (1 km). La validación espacial sigue siendo algo optimista.
- **Estrato como etiqueta administrativa:** se asigna por manzana o lado de
  manzana con una metodología que incluye el entorno urbano; el modelo solo ve
  la vivienda y su ubicación aproximada.
- **Calidad del catastro:** las características físicas pueden estar
  desactualizadas. El año de construcción aparece amontonado en ciertos
  valores (1986 concentra el 12 % de train en el diagnóstico de rangos), y hay
  viviendas con 0 habitaciones o 0 baños que no se pudieron verificar contra
  el diccionario oficial; se conservaron.
- **Modelo lineal:** no captura interacciones ni heterogeneidad espacial; la
  multinomial ignora el orden de las clases, y las probabilidades no están
  bien calibradas (ECE = 0.14).
- **Riesgos éticos:** reidentificación por NPN + coordenadas y uso
  discriminatorio del estrato (capítulo 1.8).

## 11.3 Próximos pasos (Entregable 2)

- Modelos no lineales (árboles, *gradient boosting*) y **regresión logística
  ordinal**, que respeta el orden de las clases y debería ayudar con los
  estratos intermedios (2 y 5), hoy absorbidos por sus vecinos.
- Recalibrar probabilidades (Platt o isotónica, dentro de la CV espacial) y
  ajustar umbrales por clase: el estrato 2 ordena bien (AUC 0.86) pero casi
  nunca gana el argmax.
- Variables de **vecindad física**: rezago espacial de área, baños y
  antigüedad de los vecinos. Son legítimas porque no usan el estrato.
- Revisar `altura` (casi constante) y la antigüedad, que no aportan en la
  importancia por permutación.
- Más folds o bloques más pequeños para un test menos dependiente de pocas
  zonas, y comparar tamaños de bloque y buffer como hiperparámetros de la
  evaluación (notas de clase 9.1.11).
- Verificar con el diccionario del catastro los ceros de habitaciones y
  baños y el amontonamiento de años.
- Ubicar los predios informales con otra fuente para reducir el sesgo de
  cobertura.

## 11.4 Veredicto: ¿la base de datos sirve para el proyecto?

Para cerrar, se revisa si la base cumple lo que pide el proyecto, contrastando
cada requisito con un score calculado en los capítulos anteriores. Cada
criterio indica de dónde sale: la **rúbrica del Entregable 1** (secciones 1,
2.3, 2.5 y 3, y la Figura 1) o las **notas de clase** del profesor Lihki Rubio
(sección 9). Donde ninguna fija un número, se usa el umbral que el propio EDA
declaró antes de modelar.

| # | Criterio | Fuente | Umbral | Score obtenido | Cap. | ¿Cumple? |
|---|---|---|---|---|---|---|
| 1 | Tamaño de la muestra | Rúbrica 1 | ≥ 20 000 observaciones | 332 718 viviendas | 1 | Sí |
| 2 | Relación n/p (filas por cada columna del modelo; ver 1.6) | Rúbrica 1 | reportar; n ≫ p | ~9 500 filas por columna | 1 | Sí |
| 3 | Casos de la clase minoritaria | Rúbrica 1 y 2.1 | suficientes sin sobremuestreo | 12 390 (estrato 6, train); desbalance 4.6 : 1 | 4 | Sí |
| 4 | Entidades independientes | Rúbrica 1 | reportar el tamaño efectivo | 168 044 edificios; ICC = 0.991; tamaño efectivo ≈ 1 870 | 1 | Sí, con partición por bloques |
| 5 | Valores faltantes | Notas 9.10.4.1.2 | < 30 % por variable | máximo 0.12 %; mecanismo MAR | 3 | Sí |
| 6 | Sesgo de muestreo y representatividad | Rúbrica 1 | sin sesgo de cobertura | 16.2 % sin coordenadas, casi todo informal; V = 0.42 con el estrato | 3, 8 | Con reservas |
| 7 | Asociación predictoras–objetivo (tamaño de efecto) | Rúbrica 2.3 | al menos un efecto grande (η² > 0.14, umbral del cap. 6) | η² = 0.50 (`y_km`), 0.31 (baños), 0.21 (piso) | 6 | Sí |
| 8 | Multicolinealidad | Rúbrica 2.3 | VIF ≤ 10 (regla del cap. 6) | VIF < 3 en las físicas; 5.4 con las espaciales | 6 | Sí |
| 9 | Fuga de datos | Rúbrica 2.5 | AUC univariado lejos de 1 (alerta en 0.95); 0 entidades compartidas | AUC máximo 0.76; 0 edificios, predios o coordenadas compartidos | 9 | Sí |
| 10 | Comparación con la línea base trivial | Rúbrica 3; notas 9.10.4.6.2 | IC 95 % bootstrap de la diferencia excluye 0 | Δ F1 macro = +0.375 frente al `DummyClassifier`, IC [0.11, 0.39], p = 0.002 | 10 | Sí |
| 11 | AUC | Rúbrica 3; notas 9.4 | mayor que el azar (0.5) | AUC one-vs-rest macro = 0.89 (test) | 10 | Sí |
| 12 | ¿El problema es trivial? | Rúbrica, Figura 1 y nota crítica | accuracy < 80–90 % | accuracy = 0.46 en CV espacial y 0.60 en test | 10 | Sí (no es trivial) |
| 13 | Tamaño de muestra suficiente | Rúbrica 3 (curva de aprendizaje) | curva plana al aumentar los datos | F1 de 0.27 a 0.29 al multiplicar por 20 los datos | 10 | Sí |
| 14 | Residuos sin dependencia espacial | Rúbrica 3 | Moran de residuos cercano a 0 | Moran = 0.70 | 10 | Con reservas |

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
  línea base trivial. La pregunta de control de la Figura 1 de la rúbrica
  ("¿accuracy ≥ 80–90 %?") da **no**: el problema no es trivial, no hay
  señales de fuga y el dataset es suficientemente desafiante para el curso.
  La curva de aprendizaje muestra que la cantidad de datos no es el límite.

Las dos reservas son:

- **Representatividad (criterio 6).** La base describe la ciudad formal: las
  filas sin coordenadas son casi todas predios informales de estratos 1 y 2,
  y quedan fuera del modelo. Las conclusiones no se deben extrapolar a esos
  asentamientos.
- **Residuos con dependencia espacial (criterio 14).** Según la rúbrica, esto
  indica información estructural sin capturar. Es una limitación del modelo
  lineal más que de la base: el modelo capta la tendencia norte-sur, pero no
  los barrios. Lo mismo explica que el F1 macro sea modesto (0.30 en CV
  espacial). Variables de vecindad y modelos más flexibles quedan para el
  Entregable 2 (11.3), con la misma base.

En resumen, la base no necesita más filas ni otra fuente para seguir con el
proyecto; necesita mejores variables derivadas (vecindad física) y modelos
más flexibles.
