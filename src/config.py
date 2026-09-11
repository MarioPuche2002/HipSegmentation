"""
Carga config.yaml (parametros del proyecto) y paths.toml (rutas locales
de la maquina) y los combina en un solo diccionario de configuracion.
"""
import sys

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def load_config(config_path: str = "config.yaml", paths_path: str = "paths.toml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    try:
        with open(paths_path, "rb") as f:
            paths = tomllib.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No se encontro '{paths_path}'. Copia 'paths.example.toml' como "
            f"'{paths_path}' y coloca ahi las rutas reales de tu dataset."
        ) from exc

    config["paths"] = paths["data"]
    return config