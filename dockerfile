# ---------------------------------------------------------------
# Imagen Docker de la API de segmentacion de cadera (U-Net 128x128)
# Usa requirements-api.txt (tflite-runtime): ~230 MB de RAM en uso
#
# Construir:  docker build -t hip-segmentation .
# Ejecutar:   docker run -p 8000:8000 hip-segmentation
# Abrir:      http://localhost:8000/docs
# ---------------------------------------------------------------
FROM python:3.9-slim

# PYTHONUNBUFFERED=1 hace que los logs salgan de inmediato por consola
# (necesario para verlos en `docker logs` y en el panel de Render)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOG_LEVEL=INFO \
    LOG_DIR=/app/logs \
    PORT=8000

WORKDIR /app

# 1) Dependencias primero: Docker cachea esta capa y no reinstala
#    las librerias cada vez que cambias el codigo
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 2) Codigo, configuracion y modelo entrenado (version TFLite, 1.5 MB)
COPY config.yaml .
COPY src/ src/
COPY api/ api/
COPY models/hip_segmentation_unet.tflite models/hip_segmentation_unet.tflite

# 3) Usuario sin privilegios de root + carpeta de logs
RUN useradd --create-home appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health')"

# Render asigna el puerto en la variable PORT; en local se usa 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]