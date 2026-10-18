# Estrato socioeconómico y características catastrales de la vivienda en Barranquilla

**Primer entregable del proyecto de investigación — Machine Learning**
Maestría · Profesor: Lihki Rubio · Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329) · Versión 3, octubre de 2026

```{admonition} Versión 3: validación por vecinos más cercanos
:class: important
Esta es la **versión 3** del libro. Agrega el **capítulo 14**: la validación cruzada se diseña emparejando
distancias al vecino más cercano (kNNDM) en lugar de un buffer fijo de 1 km, se prueban una logística ordinal y
umbrales de decisión para el F1 macro, y el modelo principal pasa a entrenarse con todo el entrenamiento
(F1 macro de test 0.493, igual que en la versión 2, con un MAE ordinal más bajo: 0.384 frente a 0.439). La
partición en entrenamiento y prueba y los capítulos 1 a 13 no cambian. La versión 2 sigue publicada en
[/v2/](https://mcarocantillo.github.io/entregable1-ml-barranquilla/v2/).
```

```{admonition} Versión 2: viviendas ubicadas por su manzana
:class: important
Esta es la **versión 2** del libro. La diferencia con la versión 1 está en un
solo paso de la preparación de los datos: las viviendas que llegan sin
coordenadas (16.2 %, casi todas informales de estratos 1 y 2) se ubican con
el **centroide de su manzana** cuando la manzana es compacta (capítulo 1,
sección 1.7.2). Con ello el universo de modelado pasa del 83.8 % al 96.9 % de
las viviendas, cambia la partición en entrenamiento y prueba, y se recalculan
todos los análisis y el modelo. La **versión 1**, tal como se entregó, sigue
publicada en
[mcarocantillo.github.io/entregable1-ml-barranquilla](https://mcarocantillo.github.io/entregable1-ml-barranquilla/),
y su código, en la etiqueta `v1` del repositorio.
```

```{admonition} Resumen
:class: tip
A partir del servicio abierto de catastro de Barranquilla se construye una
base de **332 718 unidades de vivienda** con estrato residencial válido. Las
viviendas sin coordenadas propias (16.2 %, casi todas informales) se ubican
con el centroide de su manzana cuando la manzana es compacta, de modo que
**322 418 (96.9 %) tienen coordenadas** y forman el universo de modelado. Se
realiza un análisis exploratorio uni, bi, multivariado y espacial, y se entrena
una regresión logística multinomial para predecir el estrato socioeconómico
(1–6), comparada con líneas base triviales y espaciales bajo validación por
bloques espaciales de 2 km con buffer de 1 km. El estrato está extremadamente
autocorrelacionado en el espacio (Moran's I = 0.92) y casi no varía dentro de
un edificio (ICC = 0.99). El modelo base (variables físicas de la vivienda y
ubicación) logra un F1 macro de 0.32 en validación cruzada espacial y de 0.49
en test (IC 95 % [0.28, 0.51]), con 96 % de aciertos a ±1 estrato. Supera con
claridad a las líneas base triviales (F1 de 0.09 a 0.16) y, de forma
descriptiva, a una regla puramente geográfica (la clase más frecuente en una
zona de 4 km; F1 0.35), aunque esa diferencia no es significativa con 7
bloques de prueba. Una validación aleatoria habría casi duplicado el
desempeño estimado (F1 0.62).
```

## Cómo está organizado este libro

El orden de los capítulos sigue el **orden de trabajo** que exigen las instrucciones del proyecto
(reservar el conjunto de prueba antes de cualquier decisión basada en los
datos), y cada capítulo indica qué sección de las instrucciones del proyecto cubre:

| Capítulo | Contenido | Sección de las instrucciones |
|---|---|---|
| Guía de lectura | conceptos clave con ejemplos y recorrido de los datos (de 382 597 filas al modelo) | — |
| 1. Base de datos | problema, fuente, estructura del catastro (vivienda, predio, predio matriz), limpieza, calidad, filas sin coordenadas (dos universos), tamaño efectivo, ética | 1 |
| 2. Partición | reserva del conjunto de prueba por bloques espaciales | 2 (orden de trabajo), 3 |
| 3. Calidad en train | faltantes (patrón y mecanismo), outliers, sesgo de cobertura | 1 (calidad), 2.9 |
| 4. Variable objetivo | clases, desbalance, comportamiento espacial, implicaciones | 2.1 |
| 5. Unidimensional | estadísticos, distribuciones, normalidad, transformaciones, categorías raras | 2.2 |
| 6. Bidimensional | correlaciones, pruebas con tamaño de efecto y Holm, V de Cramér, información mutua, VIF | 2.3 |
| 7. Multivariado | PCA, outliers multivariados, clustering exploratorio | 2.4 |
| 8. Espacial | coordenadas, mapas, patrón de puntos, Moran/LISA/Gi*, correlograma, MAUP | 2.7 |
| 9. Fuga de datos | disponibilidad, identificadores, AUC univariado, entidades repetidas | 2.5 |
| Resumen ejecutivo del EDA | calidad, variables prometedoras, problemas y decisiones preliminares | 2 (notas 9.10.4.1.5) |
| 10. Modelo base | Pipeline, líneas base, CV espacial, métricas con IC, residuos, curva de aprendizaje | 2.9, 3 |
| 14. Validación por vecinos más cercanos (versión 3) | distancia al vecino conocido, validación kNNDM, logística ordinal, umbrales para el F1 macro | 3 |
| 11. Conclusiones | hallazgos, limitaciones, próximos pasos y veredicto sobre la base de datos según las instrucciones del proyecto | Figura 1, nota crítica |
| Apéndices | bitácora de decisiones, reproducibilidad, referencias | Entregables |

Las secciones 2.6 (temporal) y 2.8 (espacio-temporal) **no aplican**: los datos
son una foto del catastro sin marca de tiempo por observación (sección 1.3).

## Cómo leer este libro

Si el vocabulario de *machine learning* no es familiar, conviene empezar por
la **Guía de lectura**: explica con ejemplos los conceptos que se repiten
(entrenamiento y prueba, fuga de información, validación cruzada, bloques
espaciales, F1 macro…) y resume cuántas filas usa cada etapa. Cada capítulo
abre con un cuadro **"Qué hace este capítulo"** y cierra con sus decisiones.

## Trazabilidad EDA → modelo

Cada capítulo del EDA registra sus decisiones (con el hallazgo que las motiva)
en `datos/procesados/decisiones_eda.json`. El capítulo 10 construye el
`Pipeline` **leyendo ese archivo**, de modo que cada paso de preprocesamiento
se puede rastrear a un hallazgo concreto del EDA.

```{admonition} Sobre las cifras del texto
:class: note
Los cuadros de interpretación citan las cifras de la ejecución con los datos
reales (octubre de 2026, con la ubicación por manzana; semilla 42). Si se vuelve a ejecutar el libro con
otra descarga del catastro, hay que revisar esas cifras frente a las salidas de
cada celda.
```
