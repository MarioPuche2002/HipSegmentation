"""
API de segmentacion de cadera. Recibe una radiografia (solo eso, NUNCA
una mascara -- la mascara es lo que este servicio genera).

Dos endpoints:
    POST /predict        -> JSON con la mascara en base64 + metadatos
    POST /predict/image   -> PNG crudo de la mascara (para ver directo)

Ejecutar en local:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Luego abrir http://localhost:8000/docs
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # DEBE ir antes de importar tensorflow

from typing import Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pandera.errors import SchemaErrors
from pydantic import BaseModel

from src.config import load_config
from src.predict import load_model, predict_mask, predict_mask_png_bytes

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
    MODEL = load_model(CONFIG["model"]["output_path"])
    print(f"Modelo cargado desde {CONFIG['model']['output_path']} (corriendo en CPU)")


@app.exception_handler(RequestValidationError)
async def error_de_validacion_en_espanol(request: Request, exc: RequestValidationError):
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

    return JSONResponse(status_code=422, content={"detail": errores_traducidos})


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": MODEL is not None}


@app.post("/predict", response_model=SegmentationResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imagenes PNG o JPEG")

    image_bytes = await file.read()

    try:
        result = predict_mask(MODEL, image_bytes, CONFIG)
    except SchemaErrors as exc:
        raise HTTPException(
            status_code=400,
            detail=f"La imagen no paso la validacion: {exc.failure_cases[['column', 'check']].to_dict('records')}",
        )

    return result


@app.post(
    "/predict/image",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def predict_image(file: UploadFile = File(...)):
    """
    Igual que /predict, pero devuelve la mascara como imagen PNG cruda
    (Content-Type: image/png) en vez de JSON con base64 -- util para
    abrir el resultado directamente en el navegador o en Swagger UI sin
    tener que decodificar nada. No trae 'porcentaje_pixeles_por_clase'
    ni 'se_aplico_preprocesamiento' -- para eso, usa /predict.
    """
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imagenes PNG o JPEG")

    image_bytes = await file.read()

    try:
        png_bytes = predict_mask_png_bytes(MODEL, image_bytes, CONFIG)
    except SchemaErrors as exc:
        raise HTTPException(
            status_code=400,
            detail=f"La imagen no paso la validacion: {exc.failure_cases[['column', 'check']].to_dict('records')}",
        )

    return Response(content=png_bytes, media_type="image/png")