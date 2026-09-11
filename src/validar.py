"""
Evalua el modelo entrenado sobre el conjunto de TEST: Dice score por
clase + matriz de confusion + curvas de entrenamiento.

Uso:
    python -m src.validar
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from src.config import load_config
from src.train import load_split


def dice_sklearn(y_true, y_pred, num_classes):
    dice_scores = []
    for i in range(num_classes):
        y_true_class = (y_true == i).astype(int)
        y_pred_class = (y_pred == i).astype(int)
        dice = f1_score(y_true_class.ravel(), y_pred_class.ravel())
        dice_scores.append(dice)
    return dice_scores


def plot_training_metrics(losses, output_path="models/reports/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(losses["accuracy"], label="Train Accuracy")
    axes[0].plot(losses["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Accuracy durante el entrenamiento")
    axes[0].set_xlabel("Epocas")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(losses["loss"], label="Train Loss")
    axes[1].plot(losses["val_loss"], label="Validation Loss")
    axes[1].set_title("Evolucion de la perdida durante el entrenamiento")
    axes[1].set_xlabel("Epocas")
    axes[1].set_ylabel("Perdida")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Curvas de entrenamiento guardadas en {output_path}")


def evaluate_model(unet, x_val, y_val, num_classes):
    with tf.device('/CPU:0'):
        preds = unet.predict(x_val)

    y_preds = np.argmax(preds, axis=-1)
    dice_scores = dice_sklearn(y_val, y_preds, num_classes=num_classes)
    dice_global = np.mean(dice_scores)

    dice_scores = [round(score, 4) for score in dice_scores]
    dice_global = round(dice_global, 4)

    y_val_flat = y_val.flatten()
    y_preds_flat = y_preds.flatten()

    cm = confusion_matrix(y_val_flat, y_preds_flat, labels=np.arange(num_classes))

    ConfusionMatrixDisplay(cm, display_labels=[f"Clase {i}" for i in range(num_classes)]).plot(
        cmap="Blues", colorbar=True, values_format='d'
    )
    plt.title("Matriz de confusion por pixel - conjunto de TEST")
    os.makedirs("models/reports", exist_ok=True)
    plt.savefig("models/reports/confusion_matrix.png", bbox_inches="tight")
    print("Matriz de confusion guardada en models/reports/confusion_matrix.png")

    return dice_scores, dice_global, cm


def main():
    config = load_config()
    data_cfg, model_cfg, prep_cfg = config["data"], config["model"], config["preprocessing"]

    test_df = pd.read_csv(data_cfg["test_path"])
    x_test, y_test = load_split(test_df, prep_cfg["normalize_images"])

    unet = tf.keras.models.load_model(model_cfg["output_path"])
    print(unet.metrics_names)

    history_path = "models/training_history.csv"
    if os.path.exists(history_path):
        losses = pd.read_csv(history_path)
        plot_training_metrics(losses)
    else:
        print(f"[AVISO] No se encontro {history_path}, se omite el grafico de curvas.")

    dice_scores, dice_global, cm = evaluate_model(unet, x_test, y_test, num_classes=model_cfg["num_classes"])

    print("Dice por clases:", dice_scores)
    print("Dice global:", dice_global)
    print("Cantidad de pixeles clasificados correctamente:", np.trace(cm))


if __name__ == "__main__":
    main()