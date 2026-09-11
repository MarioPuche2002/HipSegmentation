"""
Inferencia sobre UNA radiografia subida a la API. Logica:

    1. pandera valida que sea una imagen real, con canales y tamaño
       dentro de lo permitido (NO valida si es cadera o no -- eso lo
       decide el modelo, no pandera)
    2. si la imagen YA cumple exactamente lo que el modelo espera
       (128x128, 1 canal), se segmenta directamente
    3. si no cumple, se le aplica el pipeline de preprocesamiento
       (resize + ecualizacion por gradiente) y LUEGO se segmenta

No hay mascara de entrada en ningun punto de este archivo -- la
mascara es la SALIDA que genera la U-Net, nunca un dato que alguien
suba.
"""
import base64
import io
import os
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.image_utils import extract_image_metadata_from_bytes
from src.pipeline import build_image_pipeline
from src.schema import build_single_image_schema, validate_single_image

CLASS_COLORS = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
], dtype=np.uint8)


def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def validate_uploaded_image(image_bytes: bytes, val_cfg: dict) -> dict:
    metadata = extract_image_metadata_from_bytes(image_bytes)
    df = pd.DataFrame([metadata])

    schema = build_single_image_schema(
        allowed_image_formats=val_cfg["allowed_image_formats"],
        allowed_image_channels=val_cfg["allowed_image_channels"],
        min_size=val_cfg["min_size"],
        max_size=val_cfg["max_size"],
    )
    validate_single_image(df, schema)

    return metadata


def cumple_condiciones_del_modelo(metadata: dict, target_size: int) -> bool:
    return (
        metadata["img_width"] == target_size
        and metadata["img_height"] == target_size
        and metadata["img_channels"] == 1
    )


def preparar_imagen_para_el_modelo(image_bytes: bytes, target_size: int, ya_cumple: bool) -> np.ndarray:
    if ya_cumple:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        array = np.array(image, dtype=np.float32) / 255.0
        array = np.expand_dims(array, axis=-1)   # canal
        array = np.expand_dims(array, axis=0)    # batch
        return array

    # No cumple -> pipeline completo. El pipeline trabaja con RUTAS de
    # archivo (asi esta escrito preprocess.py), asi que se guarda un
    # archivo temporal con los bytes subidos.
    #
    # IMPORTANTE (Windows): no se puede usar delete=True aqui. En Windows,
    # un archivo abierto por NamedTemporaryFile queda BLOQUEADO para
    # cualquier otro lector mientras el archivo siga abierto -- si cv2
    # intenta abrirlo dentro del mismo "with", Windows se lo niega y
    # cv2.imread devuelve None (causando luego el error en cv2.resize).
    # Por eso aqui se cierra el archivo explicitamente ANTES de que el
    # pipeline lo lea, y se borra manualmente despues.
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        tmp.write(image_bytes)
        tmp.close()  # libera el archivo para que cv2 pueda abrirlo

        image_pipeline = build_image_pipeline(target_size=target_size, normalize=True)
        batch = image_pipeline.fit_transform([tmp.name])  # shape: (1, H, W, 1)
    finally:
        os.unlink(tmp.name)

    return batch


def predict_mask(model: tf.keras.Model, image_bytes: bytes, config: dict) -> dict:
    val_cfg = config["image_validation"]
    target_size = config["model"]["image_size"]

    metadata = validate_uploaded_image(image_bytes, val_cfg)

    ya_cumple = cumple_condiciones_del_modelo(metadata, target_size)
    if ya_cumple:
        print("La imagen ya cumple las condiciones del modelo -- se segmenta directamente")
    else:
        print("La imagen no cumple las condiciones del modelo -- se aplica preprocesamiento")

    batch = preparar_imagen_para_el_modelo(image_bytes, target_size, ya_cumple)

    pred = model.predict(batch, verbose=0)[0]
    mask = np.argmax(pred, axis=-1).astype(np.uint8)

    total_pixels = mask.size
    distribucion_por_clase = {
        f"clase_{i}": round(float((mask == i).sum()) / total_pixels * 100, 2)
        for i in range(pred.shape[-1])
    }

    colored_mask = CLASS_COLORS[mask]
    mask_image = Image.fromarray(colored_mask)
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "mascara_png_base64": mask_base64,
        "porcentaje_pixeles_por_clase": distribucion_por_clase,
        "se_aplico_preprocesamiento": not ya_cumple,
    }