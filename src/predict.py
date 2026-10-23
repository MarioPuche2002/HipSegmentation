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
import time

import numpy as np
import pandas as pd
from PIL import Image

from src.image_utils import extract_image_metadata_from_bytes
from src.logger import get_logger
from src.pipeline import build_image_pipeline
from src.schema import build_single_image_schema, validate_single_image

logger = get_logger(__name__)

CLASS_COLORS = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
], dtype=np.uint8)


class TFLiteSegmenter:
    """
    Envuelve un interprete de TensorFlow Lite con la misma interfaz que
    usa la API de un modelo Keras: .input_shape y .predict(batch).
    Usa tflite-runtime (liviano, para Docker/Render) y, si no esta
    instalado, el interprete incluido en TensorFlow completo.
    """

    def __init__(self, model_path: str):
        try:
            from tflite_runtime.interpreter import Interpreter
            backend = "tflite-runtime"
        except ImportError:  # p. ej. en Windows, con el TensorFlow de requirements.txt
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            backend = "tensorflow.lite"

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]
        self.input_shape = tuple(None if i == 0 else int(d) for i, d in enumerate(self._input["shape"]))
        self.backend = backend

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        salidas = []
        for imagen in batch:  # el modelo TFLite se exporto con batch = 1
            self.interpreter.set_tensor(self._input["index"], imagen[np.newaxis].astype(np.float32))
            self.interpreter.invoke()
            salidas.append(self.interpreter.get_tensor(self._output["index"])[0])
        return np.stack(salidas)


def load_model(model_path: str, expected_size: int = None):
    """Carga un modelo .tflite (produccion) o .h5 de Keras (desarrollo)."""
    if model_path.endswith(".tflite"):
        model = TFLiteSegmenter(model_path)
        logger.info("Modelo TFLite cargado con %s", model.backend)
    else:
        import tensorflow as tf  # solo se importa si de verdad se usa el .h5
        model = tf.keras.models.load_model(model_path)

    # Verifica que el modelo guardado espere la misma resolucion que
    # config.yaml (128x128x1). Si alguien cambia el tamaño en config.yaml
    # sin reentrenar, la API falla al arrancar con un mensaje claro en
    # vez de fallar en cada peticion.
    _, height, width, channels = model.input_shape
    if expected_size is not None and (height, width) != (expected_size, expected_size):
        raise ValueError(
            f"El modelo en '{model_path}' espera entradas de {height}x{width}, "
            f"pero config.yaml define image_size={expected_size}. "
            f"Reentrena el modelo o corrige config.yaml."
        )
    return model


def validate_uploaded_image(image_bytes: bytes, val_cfg: dict) -> dict:
    metadata = extract_image_metadata_from_bytes(image_bytes)
    df = pd.DataFrame([metadata])

    schema = build_single_image_schema(
        allowed_image_formats=val_cfg["allowed_image_formats"],
        allowed_image_channels=val_cfg["allowed_image_channels"],
        min_size=val_cfg["min_size"],
        max_size=val_cfg["max_size"],
    )
    logger.debug("Metadatos de la imagen subida: %s", metadata)
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


def predict_mask(model, image_bytes: bytes, config: dict) -> dict:
    val_cfg = config["image_validation"]
    target_size = config["model"]["image_size"]

    metadata = validate_uploaded_image(image_bytes, val_cfg)

    ya_cumple = cumple_condiciones_del_modelo(metadata, target_size)
    if ya_cumple:
        logger.info("Imagen %sx%s, %s canal(es): ya cumple las condiciones del modelo -- se segmenta directamente",
                    metadata["img_width"], metadata["img_height"], metadata["img_channels"])
    else:
        logger.info("Imagen %sx%s, %s canal(es): se aplica preprocesamiento (resize a %dx%d + ecualizacion)",
                    metadata["img_width"], metadata["img_height"], metadata["img_channels"],
                    target_size, target_size)

    batch = preparar_imagen_para_el_modelo(image_bytes, target_size, ya_cumple)

    inicio = time.perf_counter()
    pred = model.predict(batch, verbose=0)[0]
    logger.info("Inferencia del U-Net en %.0f ms", (time.perf_counter() - inicio) * 1000)
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