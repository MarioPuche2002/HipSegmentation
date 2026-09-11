"""
Entrena el U-Net sobre los splits generados por src/preprocess.py +
src/features.py.

Uso:
    python -m src.train
"""
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.config import load_config
from src.model import create_unet_model


def gpu_setup():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("Error configuring GPU memory growth:", {e})
    else:
        print("No GPUs found.")


def load_split(df: pd.DataFrame, normalize_images: bool):
    images, masks = [], []
    for _, row in df.iterrows():
        img = cv2.imread(row["img_filepath"], 0)
        msk = cv2.imread(row["msk_filepath"], 0)
        images.append(img)
        masks.append(msk)

    images = np.expand_dims(np.array(images), axis=-1)
    masks = np.expand_dims(np.array(masks), axis=-1)

    if normalize_images:
        images = images.astype(np.float32) / 255.0

    return images, masks


def train_model(unet, x_train, y_train, x_val, y_val, batch_size, epochs, patience):
    early_stop = EarlyStopping(monitor="val_loss", patience=patience)

    with tf.device('/CPU:0'):
        dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        dataset_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    unet.fit(dataset_train, epochs=epochs, validation_data=dataset_val, callbacks=[early_stop])

    return unet


def main():
    config = load_config()
    data_cfg, model_cfg, prep_cfg = config["data"], config["model"], config["preprocessing"]

    gpu_setup()

    train_df = pd.read_csv(data_cfg["train_path"])
    val_df = pd.read_csv(data_cfg["val_path"])

    print(f"Cargando {len(train_df)} imagenes de entrenamiento y {len(val_df)} de validacion...")
    x_train, y_train = load_split(train_df, prep_cfg["normalize_images"])
    x_val, y_val = load_split(val_df, prep_cfg["normalize_images"])

    unet = create_unet_model(model_cfg["image_size"], model_cfg["num_classes"], model_cfg["base_filters"])
    unet.summary()

    unet = train_model(
        unet, x_train, y_train, x_val, y_val,
        batch_size=model_cfg["batch_size"],
        epochs=model_cfg["epochs"],
        patience=model_cfg["early_stopping_patience"],
    )

    unet.save(model_cfg["output_path"])
    print(f"Modelo guardado en {model_cfg['output_path']}")

    history_df = pd.DataFrame(unet.history.history)
    history_df.to_csv("models/training_history.csv", index=False)
    print("Historial de entrenamiento guardado en models/training_history.csv")


if __name__ == "__main__":
    main()