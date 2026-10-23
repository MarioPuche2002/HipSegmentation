"""
Carga config.yaml (parametros del proyecto) y paths.toml (rutas locales
de la maquina) y los combina en un solo diccionario de configuracion.

paths.toml solo hace falta para preprocesar/entrenar (src.preprocess).
La API (y el contenedor Docker / Render) no lo necesita: si no existe,
se continua con config["paths"] vacio.
"""
import os
import sys

import yaml

from src.logger import get_logger

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = get_logger(__name__)


def load_config(config_path: str = "config.yaml", paths_path: str = "paths.toml",
                require_paths: bool = False) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if os.path.isfile(paths_path):
        with open(paths_path, "rb") as f:
            config["paths"] = tomllib.load(f)["data"]
    elif require_paths:
        raise FileNotFoundError(
            f"No se encontro '{paths_path}'. Copia 'paths.example.toml' como "
            f"'{paths_path}' y coloca ahi las rutas reales de tu dataset."
        )
    else:
        logger.debug("No hay '%s'; se continua sin rutas del dataset (modo API).", paths_path)
        config["paths"] = {}

    target = config["preprocessing"]["target_size"]
    image_size = config["model"]["image_size"]
    if target != image_size:
        raise ValueError(
            f"preprocessing.target_size ({target}) y model.image_size ({image_size}) "
            f"deben ser iguales en config.yaml"
        )
    return config