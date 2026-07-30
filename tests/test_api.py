from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_invalid_prediction() -> None:
    response = client.post("/predict", json={"pm2_5": 10})
    assert response.status_code == 422