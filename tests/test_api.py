"""
Test de humo de la API. Requiere un modelo entrenado en models/.

Ejecutar:
    pytest tests/test_api.py -v
"""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["modelo_cargado"] is True