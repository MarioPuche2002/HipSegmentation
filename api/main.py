"""
API de segmentacion de cadera. Recibe una radiografia (solo eso, NUNCA
una mascara -- la mascara es lo que este servicio genera) y devuelve
la mascara de segmentacion predicha, coloreada, como PNG en base64.

Ejecutar en local:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Luego abrir http://localhost:8000/docs
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # DEBE ir antes de importar tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # oculta los avisos internos de TF

import logging
import time
import uuid
from typing import Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pandera.errors import SchemaErrors
from pydantic import BaseModel

from src.config import load_config
from src.logger import get_logger
from src.predict import load_model, predict_mask

logger = get_logger("api")

CONFIG = load_config()

app = FastAPI(
    title="API de Segmentacion de Cadera",
    description="Segmentacion de estructuras de cadera en radiografias (5 clases)",
    version="1.0.0",
)

MODEL = None


class SegmentationResponse(BaseModel):
    mascara_png_base64: str
    porcentaje_pixeles_por_clase: Dict[str, float]
    se_aplico_preprocesamiento: bool


@app.on_event("startup")
def startup_event():
    global MODEL
        # MODEL_PATH permite elegir el modelo sin tocar config.yaml.
    # Por defecto la API usa el .tflite (liviano); si no existe, el .h5.
    model_path = os.getenv("MODEL_PATH") or CONFIG["model"]["serving_path"]
    if not os.path.isfile(model_path):
        logger.warning("No existe %s, se usa el modelo Keras %s", model_path, CONFIG["model"]["output_path"])
        model_path = CONFIG["model"]["output_path"]
    logger.info("Cargando modelo desde %s ...", model_path)
    inicio = time.perf_counter()
    try:
        MODEL = load_model(model_path, expected_size=CONFIG["model"]["image_size"])
    except Exception:
        logger.exception("No se pudo cargar el modelo desde %s", model_path)
        raise
    logger.info(
        "Modelo cargado en %.2f s (entrada %s, corriendo en CPU)",
        time.perf_counter() - inicio, MODEL.input_shape,
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Registra cada peticion: id, metodo, ruta, codigo de respuesta y duracion."""
    request_id = uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    inicio = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duracion_ms = (time.perf_counter() - inicio) * 1000
        logger.exception("[%s] %s %s -> 500 (%.1f ms) error no controlado",
                         request_id, request.method, request.url.path, duracion_ms)
        return JSONResponse(status_code=500, content={"detail": "Error interno del servidor"})

    duracion_ms = (time.perf_counter() - inicio) * 1000
    nivel = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(nivel, "[%s] %s %s -> %d (%.1f ms)", request_id, request.method,
               request.url.path, response.status_code, duracion_ms)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def error_de_validacion_en_espanol(request: Request, exc: RequestValidationError):
    """
    FastAPI/Pydantic devuelven los errores de validacion (422) en ingles
    por defecto (ej. 'Field required'). Este manejador los traduce a
    los mensajes mas comunes, para que la API responda en español.
    """
    traducciones = {
        "Field required": "Este campo es obligatorio",
        "field required": "Este campo es obligatorio",
        "Input should be a valid": "El valor no es del tipo esperado",
    }

    errores_traducidos = []
    for error in exc.errors():
        mensaje = error.get("msg", "")
        for texto_en, texto_es in traducciones.items():
            if texto_en in mensaje:
                mensaje = texto_es
                break
        errores_traducidos.append({
            "campo": ".".join(str(loc) for loc in error.get("loc", [])),
            "mensaje": mensaje,
        })

    logger.warning("Peticion invalida en %s: %s", request.url.path, errores_traducidos)
    return JSONResponse(status_code=422, content={"detail": errores_traducidos})


@app.get("/health")
def health():
    size = CONFIG["model"]["image_size"]
    return {
        "status": "ok",
        "modelo_cargado": MODEL is not None,
        "resolucion_entrada": f"{size}x{size}",
    }


@app.post("/predict", response_model=SegmentationResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    request_id = getattr(request.state, "request_id", "-")

    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        logger.warning("[%s] Archivo rechazado '%s': tipo %s no permitido",
                       request_id, file.filename, file.content_type)
        raise HTTPException(status_code=400, detail="Solo se aceptan imagenes PNG o JPEG")

    image_bytes = await file.read()
    logger.info("[%s] Imagen recibida '%s' (%s, %.1f KB)", request_id, file.filename,
                file.content_type, len(image_bytes) / 1024)

    try:
        result = predict_mask(MODEL, image_bytes, CONFIG)
    except SchemaErrors as exc:
        fallas = exc.failure_cases[["column", "check"]].to_dict("records")
        logger.warning("[%s] La imagen '%s' no paso la validacion de pandera: %s",
                       request_id, file.filename, fallas)
        raise HTTPException(
            status_code=400,
            detail=f"La imagen no paso la validacion: {fallas}",
        )

    logger.info("[%s] Segmentacion lista. Pixeles por clase (%%): %s",
                request_id, result["porcentaje_pixeles_por_clase"])
    return result