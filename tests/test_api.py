"""
Tests de la API. Usan el modelo versionado en models/ (.tflite).

Ejecutar:
    pytest tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app

SAMPLE = "tests/manual_samples/Hip_Image1.jpg"


@pytest.fixture(scope="module")
def client():
    # El "with" es necesario: asi se ejecuta el evento startup que carga el modelo
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["modelo_cargado"] is True
    assert response.json()["resolucion_entrada"] == "128x128"


def test_predict_devuelve_mascara(client):
    with open(SAMPLE, "rb") as f:
        response = client.post("/predict", files={"file": ("Hip_Image1.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["se_aplico_preprocesamiento"] is True
    assert set(body["porcentaje_pixeles_por_clase"]) == {f"clase_{i}" for i in range(5)}
    assert "X-Request-ID" in response.headers


def test_predict_rechaza_archivos_que_no_son_imagen(client):
    response = client.post("/predict", files={"file": ("notas.txt", b"hola", "text/plain")})
    assert response.status_code == 400