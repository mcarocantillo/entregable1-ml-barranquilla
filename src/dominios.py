# -*- coding: utf-8 -*-
"""
Descarga (una sola vez) los metadatos de los campos del servicio ArcGIS:
alias, tipo y dominio de códigos (ej. qué significa tipo_vivienda = 4).
Sirve para el diccionario de variables. Si no hay internet o el servidor
bloquea la consulta, el libro sigue funcionando sin esta información.

Por qué importa: varias columnas del catastro llegan como códigos numéricos
cuyo significado solo está en el dominio del servicio. Sin él, un código
podría confundirse con una magnitud (p. ej. tratar tipo_vivienda = 4 como
"el doble" de 2). El resultado se guarda en caché (config.ARCHIVO_DOMINIOS)
para no depender del servidor en cada ejecución y hacer el libro reproducible.
"""

import json

from . import config as C

# Capas del MapServer consultadas: {id de capa en el servicio: nombre legible}
CAPAS = {500: "Predio", 505: "Características", 305: "Unidad de construcción"}


def _get_json(url):
    """Descarga la descripción JSON de una capa del servicio ArcGIS.

    Parámetros
    ----------
    url : str
        URL de la capa (sin parámetros); se añade f=json.

    Devuelve
    --------
    dict
        Metadatos de la capa (incluye la lista 'fields').

    Por qué
    -------
    Primero intenta con curl_cffi imitando a un navegador, porque el servidor
    de la Alcaldía puede rechazar clientes HTTP genéricos; si esa librería no
    está instalada o falla, recurre a urllib de la biblioteca estándar.
    """
    try:
        from curl_cffi import requests as r  # el mismo cliente que usa 01_descarga_predios.py
        # impersonate: replica la huella TLS de Chrome para no ser bloqueado
        return r.get(url, params={"f": "json"}, impersonate="chrome120", timeout=60).json()
    except Exception:
        # Respaldo sin dependencias externas
        import urllib.request
        with urllib.request.urlopen(url + "?f=json", timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))


def descargar_dominios(forzar=False):
    """Obtiene alias, tipo y dominio de códigos de cada campo de las CAPAS.

    Parámetros
    ----------
    forzar : bool, por defecto False
        Si True, ignora la caché local y vuelve a consultar el servidor.

    Devuelve
    --------
    dict
        {nombre_campo_en_minúsculas: {"capa", "alias", "tipo_arcgis",
        "dominio": {código (str): significado}}}.

    Por qué
    -------
    Se consulta el servicio una sola vez y se reutiliza el JSON guardado: así
    el diccionario de variables no cambia entre ejecuciones ni depende de que
    el servidor esté disponible. Los nombres se pasan a minúsculas para que
    coincidan con las columnas del CSV.
    """
    # Caché: si ya existe el archivo, no se vuelve a descargar
    if C.ARCHIVO_DOMINIOS.exists() and not forzar:
        return json.loads(C.ARCHIVO_DOMINIOS.read_text(encoding="utf-8"))
    salida = {}
    for capa, nombre in CAPAS.items():
        meta = _get_json(f"{C.URL_SERVICIO}/{capa}")
        for f in meta.get("fields", []):
            dom = f.get("domain") or {}  # campos sin dominio de códigos -> dict vacío
            # codedValues: lista de {"code": ..., "name": ...}; la clave se pasa a
            # texto porque JSON solo admite claves de tipo cadena
            codigos = {str(cv["code"]): cv["name"] for cv in dom.get("codedValues", [])}
            # Si un campo aparece en varias capas, prevalece la última consultada
            salida[f["name"].lower()] = {"capa": f"{capa} {nombre}", "alias": f.get("alias"),
                                         "tipo_arcgis": f.get("type"), "dominio": codigos}
    C.ARCHIVO_DOMINIOS.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    return salida
