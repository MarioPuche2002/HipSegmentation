"""
Extraccion de metadatos de un PAR imagen+mascara (entrenamiento) y de
una imagen UNICA en memoria (API, sin mascara).
"""
import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

PIL_MODE_TO_CHANNELS = {
    "1": 1, "L": 1, "I": 1, "F": 1,
    "RGB": 3, "YCbCr": 3, "LAB": 3, "HSV": 3,
    "RGBA": 4, "CMYK": 4,
}


@dataclass
class PairMetadata:
    img_filepath: str
    msk_filepath: str
    img_is_valid: bool = False
    msk_is_valid: bool = False
    img_format: Optional[str] = None
    msk_format: Optional[str] = None
    img_channels: Optional[int] = None
    msk_channels: Optional[int] = None
    img_width: Optional[int] = None
    img_height: Optional[int] = None
    msk_width: Optional[int] = None
    msk_height: Optional[int] = None
    msk_max_label: Optional[int] = None
    msk_min_label: Optional[int] = None
    error: Optional[str] = None


def _open_and_describe(filepath: str):
    if not os.path.isfile(filepath):
        return False, None, None, None, None, "El archivo no existe en disco"
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            width, height = img.size
            fmt = img.format
            mode = img.mode
            channels = PIL_MODE_TO_CHANNELS.get(mode) or len(img.convert("RGB").getbands())
        return True, fmt, channels, width, height, None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return False, None, None, None, None, f"{type(exc).__name__}: {exc}"


def extract_pair_metadata(img_filepath: str, msk_filepath: str) -> PairMetadata:
    meta = PairMetadata(img_filepath=img_filepath, msk_filepath=msk_filepath)

    img_valid, img_fmt, img_ch, img_w, img_h, img_err = _open_and_describe(img_filepath)
    meta.img_is_valid, meta.img_format, meta.img_channels = img_valid, img_fmt, img_ch
    meta.img_width, meta.img_height = img_w, img_h

    msk_valid, msk_fmt, msk_ch, msk_w, msk_h, msk_err = _open_and_describe(msk_filepath)
    meta.msk_is_valid, meta.msk_format, meta.msk_channels = msk_valid, msk_fmt, msk_ch
    meta.msk_width, meta.msk_height = msk_w, msk_h

    errors = [e for e in (img_err, msk_err) if e]
    if errors:
        meta.error = " | ".join(errors)

    if msk_valid:
        try:
            arr = np.array(Image.open(msk_filepath))
            meta.msk_max_label = int(arr.max())
            meta.msk_min_label = int(arr.min())
        except Exception as exc:
            meta.msk_is_valid = False
            meta.error = f"{meta.error + ' | ' if meta.error else ''}No se pudo leer como array: {exc}"

    return meta


def extract_image_metadata_from_bytes(image_bytes: bytes) -> dict:
    """Version para la API: la imagen llega como bytes en memoria, sin mascara."""
    result = {
        "img_filepath": "<uploaded_in_memory>",
        "img_is_valid": False,
        "img_format": None,
        "img_channels": None,
        "img_width": None,
        "img_height": None,
        "error": None,
    }
    try:
        buffer = io.BytesIO(image_bytes)
        with Image.open(buffer) as img:
            img.verify()
        buffer.seek(0)
        with Image.open(buffer) as img:
            width, height = img.size
            fmt = img.format
            mode = img.mode
            channels = PIL_MODE_TO_CHANNELS.get(mode) or len(img.convert("RGB").getbands())
        result.update(img_is_valid=True, img_format=fmt, img_channels=channels,
                      img_width=width, img_height=height)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result