# Guía de lectura

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

Esta guía está pensada para leer el libro sin perderse. Explica, antes de que
aparezcan, los conceptos que se usan en varios capítulos, y muestra el
recorrido que hacen los datos desde la descarga hasta el modelo. Quien ya
conozca el vocabulario de *machine learning* puede pasar directamente al
capítulo 1.

## Cómo está escrito cada capítulo

Todos los capítulos siguen el mismo orden:

1. Un cuadro inicial **"Qué hace este capítulo"**: la pregunta que responde,
   con qué datos trabaja y qué conceptos nuevos introduce.
2. El análisis, en el orden en que se hace: primero se explica el método
   (con un ejemplo), luego se muestra el código y su salida, y después la
   interpretación.
3. Un cierre con las **decisiones** que salen del capítulo. Las decisiones se
   guardan en `datos/procesados/decisiones_eda.json`, y el modelo del
   capítulo 10 las lee de ahí.

Las celdas de código marcadas como ocultas (por ejemplo, las que solo dibujan
figuras) se pueden desplegar en la versión HTML.

## El recorrido de los datos

Los números de filas cambian a lo largo del libro porque cada etapa trabaja
con un subconjunto distinto. Esta tabla resume de dónde sale cada cifra:

| Etapa | Filas | Qué se quitó o separó | Capítulo |
|---|---|---|---|
| CSV descargado del catastro | 382 597 | — | 1 |
| Viviendas limpias con estrato 1–6 | 332 718 | garajes y depósitos, unidades de menos de 10 m², edificios completos en una fila, predios sin estrato | 1 |
| Viviendas con coordenadas (**universo de modelado**) | 322 418 | 10 300 viviendas sin ubicación, casi todas vivienda informal de estrato 1. Incluye 43 594 viviendas que llegaron sin coordenadas y se ubicaron con el centroide de su manzana | 1.7, 2 |
| Conjunto de **entrenamiento** | 257 273 | se separa el conjunto de prueba (7 zonas de la ciudad) | 2 |
| Conjunto de **prueba** | 65 145 | se guarda sin mirar hasta el final | 2, 10 |
| Entrenamiento del modelo final | 144 586 | viviendas de entrenamiento a menos de 1 km de la zona de prueba (buffer) | 8, 10 |

El capítulo 1 describe la base con las 332 718 viviendas y, al lado, con las
322 418 que tienen coordenadas. Los capítulos 3 a 9 usan solo las 257 273 de
entrenamiento. El capítulo 10 entrena con 144 586 y evalúa una sola vez con
las 65 145 de prueba.

## Conceptos clave, con ejemplos

### Variable objetivo y predictoras

La **variable objetivo** es lo que se quiere predecir: el estrato
(`estrato_num`, de 1 a 6). Las **predictoras** son la información que el
modelo usa para predecirla: el área construida, el número de baños, la
ubicación, etc. *Ejemplo:* con "120 m², 3 baños, piso 8, norte de la ciudad"
(predictoras), el modelo propone "estrato 5" (objetivo).

### Clasificación multiclase ordinal

Se trata de **clasificación** porque la respuesta es una categoría, no un
número continuo; es **multiclase** porque hay más de dos categorías (seis
estratos), y es **ordinal** porque las categorías tienen un orden.
Confundir un estrato 5 con un 6 es un error pequeño; confundir un 1 con un 6
es un error grave. Por eso, además de las métricas usuales, se reportan
métricas que penalizan más los errores lejanos (capítulo 4).

### Entrenamiento y prueba

El modelo se ajusta con un **conjunto de entrenamiento** y se evalúa con un
**conjunto de prueba** que no vio. Es como estudiar con unos ejercicios y
presentar el examen con otros: si el examen tuviera los mismos ejercicios del
estudio, la nota no diría nada sobre lo que el estudiante realmente aprendió.
La prueba se separa **antes** de mirar los datos en detalle (capítulo 2) y
solo se usa una vez, al final (capítulo 10).

### Fuga de información

Hay **fuga** cuando el modelo, o una decisión del análisis, usa información
que no tendría en el momento real de predecir. El resultado parece mejor de
lo que es. *Ejemplos:* (i) calcular la mediana para rellenar faltantes con
todas las filas, incluidas las de prueba; (ii) usar como predictora el
estrato promedio de los vecinos, que es justamente lo que no se conoce en una
zona nueva. Por eso todo lo que **aprende** de los datos (medianas, umbrales,
escalas) se calcula solo con entrenamiento, y el capítulo 9 audita cada
variable.

### Validación cruzada y *folds*

Para elegir entre modelos sin tocar la prueba, el entrenamiento se divide en
varias partes llamadas **folds** (aquí, 5). Se entrena con 4 y se valida con
la que queda, rotando hasta que cada parte haya sido validación una vez. El
promedio de las 5 validaciones estima cómo le irá al modelo con datos nuevos,
y la variación entre folds indica qué tan estable es esa estimación.

### Filas que no son independientes: el predio matriz

Los apartamentos de un mismo edificio o conjunto (el **predio matriz**,
sección 1.3) casi siempre tienen el mismo estrato. Si el apartamento 101
queda en entrenamiento y el 201 en prueba, el modelo "acierta" sin haber
aprendido nada: solo reconoce el edificio. Por eso la partición nunca
reparte un predio matriz entre entrenamiento y prueba, y por eso la muestra
vale menos de lo que sugiere su número de filas (tamaño efectivo, sección
1.9).

### Autocorrelación espacial

Es la tendencia de los lugares cercanos a parecerse. En Barranquilla el
estrato está muy autocorrelacionado: si una casa es estrato 1, sus vecinas
casi seguro también lo son. Se mide con el **índice de Moran**, que vale
cerca de 0 si no hay patrón y cerca de 1 si los vecinos son casi idénticos
(capítulo 8; aquí da 0.92).

### Bloques espaciales y buffer

Si se separara la prueba al azar, cada vivienda de prueba tendría vecinas
casi idénticas en entrenamiento y el modelo parecería mejor de lo que es.
Por eso la ciudad se divide en **bloques** de 2 km × 2 km y se separan
**bloques completos** como prueba: el modelo se evalúa en zonas que no vio.
Además, se deja un **buffer** (franja de exclusión) de 1 km: las viviendas
de entrenamiento a menos de 1 km de una zona de prueba se retiran, para que
el modelo no se apoye en vecinos inmediatos (capítulos 2, 8 y 10).

*Ejemplo:* si un bloque del barrio El Prado queda en prueba, las viviendas
del bloque vecino que están a 200 m del borde no se usan para entrenar.

### Tamaño de efecto frente a p-valor

Con más de 200 000 filas, casi cualquier diferencia, por pequeña que sea,
resulta "estadísticamente significativa" (p-valor diminuto). El p-valor dice
si una diferencia es distinguible del azar, no si es **grande**. Por eso las
conclusiones del análisis exploratorio se basan en **tamaños de efecto**
(η², V de Cramér, correlaciones), que miden qué tan fuerte es la relación
(capítulo 6).

### Desbalance de clases y F1 macro

Las clases no tienen el mismo número de casos (hay muchas más viviendas de
estrato 1 a 3 que de estrato 6). Con desbalance, el **accuracy** (porcentaje
de aciertos) engaña.

*Ejemplo:* en un grupo de 100 viviendas, 80 de estrato 1 y 20 de estrato 6,
un "modelo" que siempre dice "estrato 1" acierta el 80 %, pero nunca
reconoce un estrato 6. El **F1** de cada clase combina precisión (de las
viviendas que el modelo llamó "estrato 6", cuántas lo eran) y recall (de las
viviendas que eran estrato 6, cuántas encontró). En el ejemplo, el F1 del
estrato 1 es 0.89 y el del estrato 6 es 0. El **F1 macro** es el promedio
simple de los F1 de todas las clases: (0.89 + 0) ÷ 2 ≈ 0.44, muy por debajo
del 80 % de accuracy. Es la métrica principal del libro porque obliga al
modelo a acertar también en las clases pequeñas (capítulo 4).

### Línea base

Un modelo solo vale la pena si supera a una regla trivial. Las **líneas
base** son esas reglas: "predecir siempre el estrato más frecuente",
"predecir al azar respetando las proporciones" o "predecir el estrato más
común de la zona". Si el modelo no las supera con claridad, no aporta
(capítulo 10).

### Regresión logística multinomial y regularización

Es el modelo base del curso. Para cada vivienda calcula una probabilidad
para cada uno de los seis estratos a partir de una combinación lineal de
las predictoras, y predice el estrato con mayor probabilidad. La
**regularización** (parámetro `C`) frena coeficientes demasiado grandes para
evitar que el modelo memorice los datos de entrenamiento: un `C` pequeño
regulariza más.

### Pipeline

Es la "receta" que encadena todos los pasos de preparación (rellenar
faltantes, transformar, escalar, convertir categorías en columnas) con el
modelo. Su ventaja es que, en cada fold de la validación cruzada, todos los
pasos se aprenden **solo** con la parte de entrenamiento de ese fold, lo que
evita la fuga (capítulo 10).

### Intervalos de confianza por bootstrap de bloques

Un resultado en prueba es una sola medición, y otra selección de zonas daría
otro número. El **bootstrap** estima esa variabilidad: se vuelve a calcular
la métrica muchas veces sobre remuestras de los datos. Como las viviendas de
una misma zona no son independientes, se remuestrean **bloques completos**,
no viviendas sueltas (capítulo 10).

### Calibración

Un modelo está **calibrado** si sus probabilidades significan lo que dicen:
de todas las viviendas a las que asigna "70 % de ser estrato 4", alrededor
del 70 % deberían serlo. Un modelo puede acertar la clase y aun así dar
probabilidades poco fiables (capítulo 10).
