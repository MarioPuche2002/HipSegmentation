"""
Prueba que pandera detecta pares corruptos/desalineados (entrenamiento)
Y que detecta imagenes corruptas subidas a la API (imagen unica).

Ejecutar:
    pytest tests/test_schema.py -v
"""
import os

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors
from PIL import Image

from src.image_utils import extract_image_metadata_from_bytes, extract_pair_metadata
from src.schema import build_pair_schema, build_single_image_schema

NUM_CLASSES = 5


def _schema():
    return build_pair_schema(
        allowed_image_formats=["JPEG", "JPG", "PNG"],
        allowed_mask_formats=["PNG"],
        allowed_image_channels=[1, 3],
        allowed_mask_channels=[1],
        min_size=32,
        max_size=4096,
        num_classes=NUM_CLASSES,
    )


@pytest.fixture
def tmp_pairs(tmp_path):
    paths = {}

    img_ok = tmp_path / "img_ok.jpg"
    Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8), mode="RGB").save(img_ok)
    msk_ok = tmp_path / "msk_ok.png"
    Image.fromarray(np.random.randint(0, NUM_CLASSES, (100, 100), dtype=np.uint8), mode="L").save(msk_ok)
    paths["valid"] = (str(img_ok), str(msk_ok))

    img_2 = tmp_path / "img_2.jpg"
    Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8), mode="RGB").save(img_2)
    msk_2 = tmp_path / "msk_2.png"
    msk_2.write_bytes(os.urandom(80))
    paths["corrupt_mask"] = (str(img_2), str(msk_2))

    img_3 = tmp_path / "img_3.jpg"
    Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8), mode="RGB").save(img_3)
    msk_3 = tmp_path / "msk_3.png"
    Image.fromarray(np.random.randint(0, NUM_CLASSES, (50, 50), dtype=np.uint8), mode="L").save(msk_3)
    paths["mismatched_size"] = (str(img_3), str(msk_3))

    img_4 = tmp_path / "img_4.jpg"
    Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8), mode="RGB").save(img_4)
    msk_4 = tmp_path / "msk_4.png"
    bad_mask = np.random.randint(0, NUM_CLASSES, (100, 100), dtype=np.uint8)
    bad_mask[0, 0] = 99
    Image.fromarray(bad_mask, mode="L").save(msk_4)
    paths["invalid_label"] = (str(img_4), str(msk_4))

    return paths


def test_valid_pair_passes_metadata_extraction(tmp_pairs):
    img_fp, msk_fp = tmp_pairs["valid"]
    meta = extract_pair_metadata(img_fp, msk_fp)
    assert meta.img_is_valid and meta.msk_is_valid
    assert meta.img_width == meta.msk_width == 100
    assert meta.msk_max_label <= NUM_CLASSES - 1


def test_corrupt_mask_is_flagged(tmp_pairs):
    img_fp, msk_fp = tmp_pairs["corrupt_mask"]
    meta = extract_pair_metadata(img_fp, msk_fp)
    assert meta.img_is_valid is True
    assert meta.msk_is_valid is False


def test_pandera_rejects_all_bad_cases_and_keeps_valid(tmp_pairs):
    rows = []
    order = ["valid", "corrupt_mask", "mismatched_size", "invalid_label"]
    for key in order:
        img_fp, msk_fp = tmp_pairs[key]
        rows.append(vars(extract_pair_metadata(img_fp, msk_fp)))

    df = pd.DataFrame(rows)
    schema = _schema()

    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(df, lazy=True)

    bad_indices = set(exc_info.value.failure_cases["index"].dropna().astype(int))

    assert 0 not in bad_indices
    assert 1 in bad_indices
    assert 2 in bad_indices
    assert 3 in bad_indices


def _single_image_schema():
    return build_single_image_schema(
        allowed_image_formats=["JPEG", "JPG", "PNG"],
        allowed_image_channels=[1, 3],
        min_size=32,
        max_size=4096,
    )


def test_valid_uploaded_image_passes():
    import io
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(128, 128) * 255).astype(np.uint8), mode="L").save(buf, format="PNG")

    metadata = extract_image_metadata_from_bytes(buf.getvalue())
    df = pd.DataFrame([metadata])

    validated = _single_image_schema().validate(df, lazy=True)
    assert validated.loc[0, "img_is_valid"] == True


def test_corrupt_uploaded_image_is_rejected():
    metadata = extract_image_metadata_from_bytes(os.urandom(100))
    df = pd.DataFrame([metadata])

    with pytest.raises(SchemaErrors):
        _single_image_schema().validate(df, lazy=True)