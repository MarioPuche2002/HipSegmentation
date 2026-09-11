"""
Pipelines de scikit-learn (sklearn.pipeline.Pipeline) para el
preprocesamiento de imagenes y mascaras. El mismo pipeline se usa en
preprocess.py (entrenamiento) y en predict.py (API), asi la imagen se
procesa exactamente igual en ambos lados.
"""
import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class ImageLoader(BaseEstimator, TransformerMixin):
    """Carga cada ruta de archivo como array en escala de grises (1 canal)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) for filepath in X]


class Resizer(BaseEstimator, TransformerMixin):
    """Redimensiona cada array al tamaño objetivo. interpolation='nearest' para mascaras."""

    def __init__(self, target_size: int, interpolation: str = "linear"):
        self.target_size = target_size
        self.interpolation = interpolation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cv2_interp = cv2.INTER_NEAREST if self.interpolation == "nearest" else cv2.INTER_LINEAR
        return [cv2.resize(arr, (self.target_size, self.target_size), interpolation=cv2_interp) for arr in X]


class GradientWeightedEqualizer(BaseEstimator, TransformerMixin):
    """
    Ecualizacion de contraste ponderada por gradiente (tecnica propia).
    Pondera cada nivel de gris por la magnitud del gradiente (Sobel) de
    los pixeles en ese nivel, en vez de por su frecuencia -- el fondo
    (gradiente ~0) casi no influye, el contraste se redistribuye hacia
    donde hay bordes reales (contornos oseos).
    """

    def __init__(self, bins: int = 256):
        self.bins = bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._equalize(arr) for arr in X]

    def _equalize(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.uint8)
        gx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        weighted_hist = np.zeros(self.bins, dtype=np.float64)
        flat_intensities = arr.ravel().astype(np.int64)
        flat_mag = magnitude.ravel()
        np.add.at(weighted_hist, flat_intensities, flat_mag)

        weighted_hist += 1e-6
        cdf = np.cumsum(weighted_hist)
        cdf_norm = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * (self.bins - 1)
        lut = np.clip(cdf_norm, 0, self.bins - 1).astype(np.uint8)

        return lut[arr]


class ChannelExpander(BaseEstimator, TransformerMixin):
    """Agrega el eje de canal: (H, W) -> (H, W, 1)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [np.expand_dims(arr, axis=-1) for arr in X]


class Normalizer(BaseEstimator, TransformerMixin):
    """Escala a [0, 1]. NUNCA se usa sobre mascaras."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [arr.astype(np.float32) / 255.0 for arr in X]


class MaskLabelClipper(BaseEstimator, TransformerMixin):
    """Cualquier etiqueta >= num_classes se colapsa a la ultima clase valida."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [np.where(arr >= self.num_classes, self.num_classes - 1, arr) for arr in X]


class ToNumpyArray(BaseEstimator, TransformerMixin):
    """Ultimo paso: convierte la lista de arrays en un solo tensor numpy."""

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(X, dtype=self.dtype)


def build_image_pipeline(target_size: int, normalize: bool = True, gradient_equalize: bool = True) -> Pipeline:
    steps = [
        ("load", ImageLoader()),
        ("resize", Resizer(target_size=target_size, interpolation="linear")),
    ]
    if gradient_equalize:
        steps.append(("gradient_equalize", GradientWeightedEqualizer()))
    steps.append(("expand_channel", ChannelExpander()))
    if normalize:
        steps.append(("normalize", Normalizer()))
    steps.append(("to_array", ToNumpyArray(dtype=np.float32 if normalize else np.uint8)))
    return Pipeline(steps)


def build_mask_pipeline(target_size: int, num_classes: int) -> Pipeline:
    return Pipeline([
        ("load", ImageLoader()),
        ("resize", Resizer(target_size=target_size, interpolation="nearest")),
        ("clip_labels", MaskLabelClipper(num_classes=num_classes)),
        ("expand_channel", ChannelExpander()),
        ("to_array", ToNumpyArray(dtype=np.uint8)),
    ])