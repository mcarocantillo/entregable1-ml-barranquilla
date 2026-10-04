# Estrato socioeconómico y características catastrales de la vivienda en Barranquilla

**Primer entregable del proyecto de investigación — Machine Learning**
Maestría · Profesor: Lihki Rubio · Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329) · Septiembre de 2026

```{admonition} Resumen
:class: tip
A partir del servicio abierto de catastro de Barranquilla se construye una
base de **332 718 unidades de vivienda** con estrato residencial válido. Se
realiza un análisis exploratorio uni, bi, multivariado y espacial, y se entrena
una regresión logística multinomial para predecir el estrato socioeconómico
(1–6), comparada con líneas base triviales y espaciales bajo validación por
bloques espaciales de 2 km con buffer de 1 km. El estrato está extremadamente
autocorrelacionado en el espacio (Moran's I = 0.91) y casi no varía dentro de
un edificio (ICC = 0.99). El modelo base (variables físicas de la vivienda y
ubicación) logra un F1 macro de 0.30 en validación cruzada espacial y de 0.45
en test (IC 95 % [0.16, 0.46]), con 94 % de aciertos a ±1 estrato. Supera con
claridad a las líneas base triviales (F1 de 0.07 a 0.17) y, de forma
descriptiva, a una regla puramente geográfica (la clase más frecuente en una
zona de 2 km; F1 0.33), aunque esa diferencia no es significativa con 7
bloques de prueba. Una validación aleatoria habría duplicado el desempeño
estimado (F1 0.60).
```

## Cómo está organizado este libro

El orden de los capítulos sigue el **orden de trabajo** que exigen las instrucciones del proyecto
(reservar el conjunto de prueba antes de cualquier decisión basada en los
datos), y cada capítulo indica qué sección de las instrucciones del proyecto cubre:

| Capítulo | Contenido | Sección de las instrucciones |
|---|---|---|
| 1. Base de datos | problema, fuente, licencia, diccionario, estructura, tamaño, duplicados, valores imposibles, ética | 1 |
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
| 11. Conclusiones | hallazgos, limitaciones, próximos pasos y veredicto sobre la base de datos según las instrucciones del proyecto | Figura 1, nota crítica |
| Apéndices | bitácora de decisiones, reproducibilidad, referencias | Entregables |

Las secciones 2.6 (temporal) y 2.8 (espacio-temporal) **no aplican**: los datos
son una foto del catastro sin marca de tiempo por observación (capítulo 1.5).

## Trazabilidad EDA → modelo

Cada capítulo del EDA registra sus decisiones (con el hallazgo que las motiva)
en `datos/procesados/decisiones_eda.json`. El capítulo 10 construye el
`Pipeline` **leyendo ese archivo**, de modo que cada paso de preprocesamiento
se puede rastrear a un hallazgo concreto del EDA.

```{admonition} Sobre las cifras del texto
:class: note
Los cuadros de interpretación citan las cifras de la ejecución con los datos
reales (septiembre de 2026, semilla 42). Si se vuelve a ejecutar el libro con
otra descarga del catastro, hay que revisar esas cifras frente a las salidas de
cada celda.
```
