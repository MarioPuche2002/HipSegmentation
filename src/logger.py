"""
Configuracion central de logs del proyecto.

Todos los modulos piden su logger asi:

    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("mensaje")

Los logs salen por:
  - consola (stdout): es lo que ven `docker logs` y el panel "Logs" de Render
  - archivo logs/app.log con rotacion (5 archivos de 5 MB maximo)

Variables de entorno opcionales:
  LOG_LEVEL      DEBUG | INFO | WARNING | ERROR   (por defecto INFO)
  LOG_DIR        carpeta de los archivos de log    (por defecto logs)
  LOG_TO_FILE    "false" para no escribir archivo  (por defecto true)
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging() -> None:
    """Configura el logger raiz una sola vez (llamarlo varias veces no duplica handlers)."""
    global _configured
    if _configured:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if os.getenv("LOG_TO_FILE", "true").lower() != "false":
        log_dir = os.getenv("LOG_DIR", "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = RotatingFileHandler(
                os.path.join(log_dir, "app.log"),
                maxBytes=5 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:
            root.warning("No se pudo crear el archivo de logs en '%s': %s", log_dir, exc)

    # TensorFlow y uvicorn son muy verbosos en DEBUG; se limitan a WARNING
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)