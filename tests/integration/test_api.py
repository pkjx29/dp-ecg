from fastapi.testclient import TestClient

from dp_ecg.app.api import create_app

client = TestClient(create_app(with_ui=False))


def test_health_and_metadata():
    assert client.get("/health").json()["status"] == "ok"
    meta = client.get("/metadata").json()
    assert "record-level" in meta["privacy_unit"] and "Not a medical device" in meta["disclaimer"]


def test_experiments_and_privacy_results_respond():
    assert client.get("/experiments").status_code == 200
    assert client.get("/privacy-results").status_code == 200


def test_predict_validates_input():
    bad_shape = {"run_id": "baseline_seed0", "waveform": [[0.0] * 10] * 12}
    assert client.post("/predict", json=bad_shape).status_code == 422
    bad_id = {"run_id": "../../etc/passwd", "waveform": [[0.0] * 1000] * 12}
    assert client.post("/predict", json=bad_id).status_code == 422
    unknown = {"run_id": "no_such_run", "waveform": [[0.0] * 1000] * 12}
    r = client.post("/predict", json=unknown)
    assert r.status_code == 404 and "/" not in r.json()["detail"]


def test_dashboard_builds_and_mounts():
    """Catches Gradio API changes: the full app (API + dashboard) must construct."""
    app = create_app(with_ui=True)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/").status_code == 200
