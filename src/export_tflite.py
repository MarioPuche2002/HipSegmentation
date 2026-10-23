"""
Convierte el modelo entrenado (.h5 de Keras) a TensorFlow Lite (.tflite)
para servirlo en produccion.

Por que: importar TensorFlow completo ocupa ~400 MB de RAM y el plan
gratuito de Render solo da 512 MB. El runtime de TFLite (tflite-runtime)
ocupa una fraccion de eso y produce la MISMA mascara (el modelo se
exporta en float32, sin cuantizar).

Uso (con el entorno de entrenamiento, que si tiene TensorFlow):
    python -m src.export_tflite
"""
import numpy as np
import tensorflow as tf

from src.config import load_config
from src.logger import get_logger

logger = get_logger(__name__)


def export_tflite(h5_path: str, tflite_path: str, image_size: int) -> None:
    model = tf.keras.models.load_model(h5_path)
    logger.info("Modelo Keras cargado desde %s (entrada %s)", h5_path, model.input_shape)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    logger.info("Modelo TFLite guardado en %s (%.1f MB)", tflite_path, len(tflite_model) / 1e6)

    # Verificacion: ambos modelos deben dar la misma mascara
    x = np.random.rand(1, image_size, image_size, 1).astype(np.float32)
    keras_mask = np.argmax(model(x, training=False).numpy(), axis=-1)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
    interpreter.invoke()
    tflite_mask = np.argmax(interpreter.get_tensor(interpreter.get_output_details()[0]["index"]), axis=-1)

    coincidencia = (keras_mask == tflite_mask).mean() * 100
    logger.info("Coincidencia de pixeles Keras vs TFLite: %.2f %%", coincidencia)


def main():
    config = load_config()
    model_cfg = config["model"]
    export_tflite(model_cfg["output_path"], model_cfg["serving_path"], model_cfg["image_size"])


if __name__ == "__main__":
    main()