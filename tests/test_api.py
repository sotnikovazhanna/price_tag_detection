from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_serves_upload_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "Обнаружение ценников" in response.text


def test_health_reports_bundled_model():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_available"] is True


def test_metrics_endpoint_returns_documented_values():
    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert response.json()["detection"]["f1_score"] == 0.995


def test_process_rejects_an_unsupported_file_type():
    response = client.post(
        "/api/process",
        files={"video": ("not-a-video.txt", b"not a video", "text/plain")},
    )

    assert response.status_code == 400
