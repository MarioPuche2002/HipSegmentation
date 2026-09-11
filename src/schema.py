"""
Schema de pandera para pares imagen+mascara (entrenamiento) y para
imagen unica (API, sin mascara).
"""
from pandera import Check, Column, DataFrameSchema


def build_pair_schema(
    allowed_image_formats,
    allowed_mask_formats,
    allowed_image_channels,
    allowed_mask_channels,
    min_size: int,
    max_size: int,
    num_classes: int,
) -> DataFrameSchema:
    return DataFrameSchema(
        {
            "img_filepath": Column(str, nullable=False),
            "msk_filepath": Column(str, nullable=False),
            "img_is_valid": Column(bool, Check.eq(True, error="La imagen no pudo abrirse (corrupta o no es una imagen)"), nullable=False),
            "msk_is_valid": Column(bool, Check.eq(True, error="La mascara no pudo abrirse (corrupta o no es una imagen)"), nullable=False),
            "img_format": Column(str, Check.isin(allowed_image_formats, error=f"Formato de imagen no permitido, se esperaba {allowed_image_formats}"), nullable=False),
            "msk_format": Column(str, Check.isin(allowed_mask_formats, error=f"Formato de mascara no permitido, se esperaba {allowed_mask_formats}"), nullable=False),
            "img_channels": Column(int, Check.isin(allowed_image_channels, error=f"Canales de imagen no soportados, se esperaba {allowed_image_channels}"), nullable=False),
            "msk_channels": Column(int, Check.isin(allowed_mask_channels, error=f"Canales de mascara no soportados, se esperaba {allowed_mask_channels}"), nullable=False),
            "img_width": Column(int, Check.in_range(min_size, max_size), nullable=False),
            "img_height": Column(int, Check.in_range(min_size, max_size), nullable=False),
            "msk_width": Column(int, Check.in_range(min_size, max_size), nullable=False),
            "msk_height": Column(int, Check.in_range(min_size, max_size), nullable=False),
            "msk_max_label": Column(int, Check.in_range(0, num_classes - 1, error=f"La mascara tiene una etiqueta fuera de rango (max permitido: {num_classes - 1})"), nullable=False),
            "msk_min_label": Column(int, Check.in_range(0, num_classes - 1), nullable=False),
        },
        checks=[
            Check(lambda df: df["img_width"] == df["msk_width"], error="El ancho de la imagen no coincide con el ancho de su mascara"),
            Check(lambda df: df["img_height"] == df["msk_height"], error="El alto de la imagen no coincide con el alto de su mascara"),
        ],
        strict=False,
        coerce=False,
    )


def validate_pairs(df, schema: DataFrameSchema):
    return schema.validate(df, lazy=True)


def build_single_image_schema(
    allowed_image_formats,
    allowed_image_channels,
    min_size: int,
    max_size: int,
) -> DataFrameSchema:
    """Este es el que usa la API -- solo imagen, sin ninguna columna de mascara."""
    return DataFrameSchema(
        {
            "img_filepath": Column(str, nullable=False),
            "img_is_valid": Column(bool, Check.eq(True, error="La imagen no pudo abrirse (corrupta o no es una imagen)"), nullable=False),
            "img_format": Column(str, Check.isin(allowed_image_formats, error=f"Formato de imagen no permitido, se esperaba {allowed_image_formats}"), nullable=False),
            "img_channels": Column(int, Check.isin(allowed_image_channels, error=f"Canales de imagen no soportados, se esperaba {allowed_image_channels}"), nullable=False),
            "img_width": Column(int, Check.in_range(min_size, max_size), nullable=False),
            "img_height": Column(int, Check.in_range(min_size, max_size), nullable=False),
        },
        strict=False,
        coerce=False,
    )


def validate_single_image(df, schema: DataFrameSchema):
    return schema.validate(df, lazy=True)