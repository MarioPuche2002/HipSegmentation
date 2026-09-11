"""
Genera los splits train/val/test a partir del manifest producido por
src/preprocess.py.

Uso:
    python -m src.features
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config


def split_manifest(df: pd.DataFrame, val_size: float, test_size: float, random_state: int):
    train_df, temp_df = train_test_split(df, test_size=(val_size + test_size), random_state=random_state)
    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(temp_df, test_size=rel_test, random_state=random_state)
    return train_df, val_df, test_df


def main():
    config = load_config()
    data_cfg = config["data"]

    manifest = pd.read_csv(data_cfg["manifest_path"])
    print(f"Manifest cargado: {len(manifest)} pares procesados")

    train_df, val_df, test_df = split_manifest(
        manifest, data_cfg["val_size"], data_cfg["test_size"], data_cfg["random_state"]
    )

    train_df.to_csv(data_cfg["train_path"], index=False)
    val_df.to_csv(data_cfg["val_path"], index=False)
    test_df.to_csv(data_cfg["test_path"], index=False)

    print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")


if __name__ == "__main__":
    main()