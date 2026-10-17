# Dashboard del Entregable 1

Dashboard en Dash Plotly del Entregable 1 (estrato socioeconómico y catastro, Barranquilla), con un selector para ver
las dos versiones del libro:

- **Versión 1** (la entregada): las viviendas sin coordenadas (16.2 %) quedan fuera del modelo.
  Libro: https://mcarocantillo.github.io/entregable1-ml-barranquilla/
- **Versión 2**: las viviendas sin coordenadas se ubican con el centroide de su manzana (96.9 % de la base entra al
  modelo). Libro: https://mcarocantillo.github.io/entregable1-ml-barranquilla/v2/

Cuatro pestañas: contexto del problema, base de datos y calidad, análisis exploratorio y modelo base (regresión
logística frente a líneas base, diagnóstico, coeficientes, residuos espaciales y simulador).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329).

## Contenido

| Ruta | Qué es |
|---|---|
| `app.py` | La aplicación (cabecera con el selector de versión y callbacks) |
| `paginas_v1.py`, `paginas_v2.py` | Las cuatro pestañas de cada versión, con sus textos e interpretaciones |
| `figuras.py` | Todas las figuras; leen `datos/<versión>/` |
| `comun.py` | Componentes comunes |
| `datos/v1`, `datos/v2` | Resultados de cada versión (viviendas sin número predial, coordenadas redondeadas a ~110 m) |
| `src/` | Módulos del Entregable 1 (necesarios para cargar el modelo guardado) |
| `assets/` | Estilos |

Los datos de `datos/` se generaron con `preparar_entregable1.py` y `preparar_datos.py` del Entregable 2, cada uno con
el código de su versión del Entregable 1. El modelo guardado necesita scikit-learn 1.8.0 (ver `requirements.txt`).

## Probarlo en tu computador

```
pip install -r requirements.txt
python app.py
```

y abrir http://localhost:8050.

## Publicarlo en Render

1. En https://render.com: **New → Web Service** y conectar el repositorio `entregable1-ml-barranquilla`.
2. **Branch:** `v2`. **Root Directory:** `dashboard`.
3. **Language:** Python 3. **Build Command:** `pip install -r requirements.txt`.
   **Start Command:** `gunicorn app:server --workers 1 --threads 2 --timeout 120`.
4. **Environment:** agregar la variable `PYTHON_VERSION` = `3.11.15`. **Instance type:** Free.

En el plan gratuito el servicio se suspende sin visitas; la primera carga tarda alrededor de un minuto. Cada versión
se construye la primera vez que se elige en el selector.
