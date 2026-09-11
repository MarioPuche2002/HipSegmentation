"""
Preprocesamiento de imagenes y mascaras: lee dataset.csv, corre los
pipelines de sklearn (resize + ecualizacion por gradiente en imagenes;
resize nearest en mascaras) y guarda el resultado en data/processed/.

Uso:
    python -m src.preprocess
"""
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import load_config
from src.pipeline import build_image_pipeline, build_mask_pipeline


def build_filepaths(dataset_csv: str, images_dir: str, masks_dir: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    if not {"imgs", "msks"}.issubset(df.columns):
        raise ValueError(f"{dataset_csv} debe tener columnas 'imgs' y 'msks'. Columnas encontradas: {list(df.columns)}")

    df["img_filepath"] = df["imgs"].apply(lambda name: os.path.join(images_dir, name))
    df["msk_filepath"] = df["msks"].apply(lambda name: os.path.join(masks_dir, name))
    return df[["img_filepath", "msk_filepath"]]


def save_processed_pairs(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    prep_cfg = config["preprocessing"]
    data_cfg = config["data"]
    num_classes = config["image_validation"]["num_classes"]

    os.makedirs(data_cfg["processed_images_dir"], exist_ok=True)
    os.makedirs(data_cfg["processed_masks_dir"], exist_ok=True)

    image_pipeline = build_image_pipeline(prep_cfg["target_size"], normalize=False)
    mask_pipeline = build_mask_pipeline(prep_cfg["target_size"], num_classes)

    img_paths = df["img_filepath"].tolist()
    msk_paths = df["msk_filepath"].tolist()

    print("Corriendo pipeline de sklearn sobre imagenes (resize + ecualizacion por gradiente)...")
    processed_imgs = image_pipeline.fit_transform(img_paths)
    print("Corriendo pipeline de sklearn sobre mascaras (resize nearest)...")
    processed_msks = mask_pipeline.fit_transform(msk_paths)

    final_img_paths, final_msk_paths = [], []
    for i, (img_arr, msk_arr) in enumerate(tqdm(zip(processed_imgs, processed_msks), total=len(processed_imgs), desc="guardando")):
        img_name = f"img_{i}.png"
        msk_name = f"msk_{i}.png"
        Image.fromarray(img_arr.squeeze()).save(os.path.join(data_cfg["processed_images_dir"], img_name))
        Image.fromarray(msk_arr.squeeze()).save(os.path.join(data_cfg["processed_masks_dir"], msk_name))
        final_img_paths.append(os.path.join(data_cfg["processed_images_dir"], img_name))
        final_msk_paths.append(os.path.join(data_cfg["processed_masks_dir"], msk_name))

    return pd.DataFrame({"img_filepath": final_img_paths, "msk_filepath": final_msk_paths})


def main():
    config = load_config()
    data_cfg = config["data"]

    os.makedirs(data_cfg["processed_dir"], exist_ok=True)

    print("=== 1/2 Armando rutas desde dataset.csv ===")
    df = build_filepaths(config["paths"]["dataset_csv"], config["paths"]["images_dir"], config["paths"]["masks_dir"])
    print(f"Total de pares: {len(df)}")

    print("\n=== 2/2 Corriendo pipelines (sklearn) de resize/ecualizacion ===")
    processed_df = save_processed_pairs(df, config)
    processed_df.to_csv(data_cfg["manifest_path"], index=False)

    print(f"\nListo. {len(processed_df)} pares procesados. Manifest en: {data_cfg['manifest_path']}")
    print("Siguiente paso: python -m src.features")


if __name__ == "__main__":
    main()