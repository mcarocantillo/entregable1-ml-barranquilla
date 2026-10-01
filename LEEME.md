# Entregable 1 — Jupyter Book (estrato y catastro, Barranquilla)

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329). Machine Learning, Maestría (Universidad del Norte). Profesor: Lihki Rubio.

## Qué hay en esta carpeta

```
entregable1_jbook/
├── intro.md                 portada del libro
├── _config.yml, _toc.yml    configuración de Jupyter Book
├── requirements.txt         dependencias
├── ejecutar_libro.py        corre todos los capítulos en orden y construye el HTML
├── src/                     código reutilizable (limpieza, partición, estadística, espacial, modelo)
├── notebooks/               capítulos 01–10 (.ipynb, SIN salidas), conclusiones y apéndices (.md)
├── datos/                   aquí va el CSV (o déjalo en la carpeta de arriba)
│   └── procesados/          lo llenan los notebooks: partición, decisiones del EDA, resultados
├── figuras/                 lo llenan los notebooks (PNG de cada figura)
└── tests/                   generador de datos SINTÉTICOS y script de prueba (no son parte del informe)
```

## Pasos (Windows, PowerShell)

1. **Ubica la carpeta** `entregable1_jbook` DENTRO de tu carpeta
   `catastro_arquetipos` (la que tiene `predios_residenciales_barranquilla.csv`).
   El libro encuentra el CSV solo. (Alternativa: copia el CSV a
   `entregable1_jbook\datos\`.)

2. **Instala dependencias** (una vez):

   ```powershell
   cd "C:\Users\mcaro\Downloads\catastro_arquetipos (1)\catastro_arquetipos\entregable1_jbook"
   python -m pip install -r requirements.txt
   ```

   Si `jupyter-book` no instala en Python 3.14, crea un entorno con 3.12:
   `py -3.12 -m venv .venv` → `.venv\Scripts\activate` → repite el `pip install`.

3. **Corre todo** (≈10–20 min según tu computador):

   ```powershell
   python ejecutar_libro.py
   ```

   Ejecuta los capítulos 01→12 en orden, guarda las salidas dentro de cada
   notebook y construye el libro. Al final abre `_build\html\index.html`.

   - Solo algunos capítulos: `python ejecutar_libro.py 05 06`
   - Solo reconstruir el HTML: `python ejecutar_libro.py --solo-build`
   - También puedes abrir cada notebook en Jupyter/VS Code y correrlo celda a
     celda, **en orden** (01 y 02 primero).

4. **Revisa y reescribe las interpretaciones.** Cada cuadro
   "Interpretación (revisar con tus resultados)" está redactado con base en la
   corrida previa y en lo esperable; compáralo con tus salidas reales y
   ajusta.

## Importante

- **Orden de trabajo del rubric:** el capítulo 2 reserva el test (bloques
  espaciales) ANTES del EDA; los capítulos 3–9 solo usan train; el test se usa
  una única vez en el capítulo 10.
- **Trazabilidad:** cada capítulo del EDA guarda sus decisiones en
  `datos/procesados/decisiones_eda.json`, y el modelo las lee de ahí. Si
  cambias algo en un capítulo del EDA, vuelve a correr desde ese capítulo
  hasta el 10.
- **Cambio en la limpieza:** ya NO se imputa con la mediana de todo el
  dataset; los valores imposibles pasan a NaN y la mediana se aprende dentro
  del Pipeline solo con train (ver Apéndice A).
- **tests/** contiene un generador de datos sintéticos con el mismo esquema
  del CSV. Solo sirvió para comprobar que todo corre de punta a punta; sus
  números no significan nada. Si alguna vez ves el aviso "DATASET SINTÉTICO"
  en un notebook, no uses esas cifras.

## Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `FileNotFoundError` del CSV | revisa el paso 1 o define `$env:CATASTRO_CSV="ruta\al\csv"` |
| error en un capítulo | `ejecutar_libro.py` se detiene y guarda el notebook hasta donde llegó: ábrelo y mira la celda con error |
| DBSCAN da un único cluster gigante (cap. 8) | baja `eps_km` (p. ej. 0.15) mirando la gráfica k-distancia |
| el buffer excluye demasiado train (cap. 8/10) | es esperable (30–50 %); el tope de 1 km está en el cap. 8 |
| no aparece el mapa interactivo | `python -m pip install folium` |

## Si Windows bloquea las librerías de Python

En algunos equipos una política de seguridad de Windows ("Una directiva de
Control de aplicaciones bloqueó este archivo") impide cargar librerías
compiladas (scikit-learn, pandas). El libro puede ejecutarse completo en
Google Colab: subir el libro comprimido (sin la carpeta `.venv`) y el CSV a
`datos/`, y correr `python ejecutar_libro.py 01 02 03 04 05 06 07 08 09 10 12`.
La partición y los folds se construyen con una función propia determinista
(`src/particion.py`), así que dan lo mismo en cualquier versión de
scikit-learn.

## Diagnóstico de rangos

`diagnostico_rangos.py` (raíz del libro) revisa los rangos de imposibilidad y
las incoherencias dentro de la fila. Uso:
`python diagnostico_rangos.py RUTA.csv [datos/procesados/dataset_modelado.parquet]`
(con el parquet, solo sobre train). El reporte usado para fijar las reglas
actuales está en `diagnostico_rangos_reporte.txt`.
