"""Backend checks use training examples only; no training or test evaluation."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import httpx
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.api import main


@pytest.fixture(scope="session", autouse=True)
def preserve_existing_files():
    before = json.loads((main.ROOT / "artifacts/backend/before.json").read_text())
    yield
    for path, expected in before["sha256"].items():
        assert hashlib.sha256((main.ROOT / path).read_bytes()).hexdigest() == expected, path


@pytest.fixture(scope="module")
def client():
    # Any accidental fit call in the API or its loaded pipeline fails the tests.
    with pytest.MonkeyPatch.context() as patch:
        def reject_fit(*args, **kwargs):
            raise AssertionError("Model fitting is forbidden in inference tests")
        patch.setattr(Pipeline, "fit", reject_fit)
        with TestClient(main.app) as test_client:
            yield test_client


@pytest.fixture
def diamond():
    return {"carat": 1.0, "cut": "Ideal", "color": "G", "clarity": "VS2",
            "depth": 61.5, "table": 57.0, "x": 6.4, "y": 6.4, "z": 3.94}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] and response.json()["artifact_verified"]


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["input_features"] == main.FEATURES
    assert info["transformed_feature_count"] == 26
    assert info["test_performance"]["sample_count"] == 8089
    assert info["model_type"] == "HistGradientBoostingRegressor"


def test_valid_prediction_and_repeat(client, diamond):
    first = client.post("/predict", json=diamond)
    assert first.status_code == 200
    body = first.json()
    assert np.isfinite(body["predicted_price_usd"]) and body["predicted_price_usd"] > 0
    assert body["currency"] == "USD" and body["disclaimer"]
    assert body["model_sha256"] == client.get("/model/info").json()["model_sha256"]
    for _ in range(3):
        assert client.post("/predict", json=diamond).json() == body


@pytest.mark.parametrize("field", main.FEATURES)
def test_missing_fields(client, diamond, field):
    del diamond[field]
    assert client.post("/predict", json=diamond).status_code == 422


@pytest.mark.parametrize("field,value", [("cut", "ideal"), ("color", "K"), ("clarity", "UNKNOWN")])
def test_invalid_categories(client, diamond, field, value):
    diamond[field] = value
    assert client.post("/predict", json=diamond).status_code == 422


@pytest.mark.parametrize("field", ["carat", "x", "y", "z"])
@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_measurements(client, diamond, field, value):
    diamond[field] = value
    assert client.post("/predict", json=diamond).status_code == 422


@pytest.mark.parametrize("field", ["depth", "table"])
@pytest.mark.parametrize("value", [0, -1, 101])
def test_percentages(client, diamond, field, value):
    diamond[field] = value
    assert client.post("/predict", json=diamond).status_code == 422


@pytest.mark.parametrize("field", ["y", "z"])
def test_existing_dimension_limit(client, diamond, field):
    diamond[field] = 20.01
    assert client.post("/predict", json=diamond).status_code == 422


@pytest.mark.parametrize("field", ["carat", "depth", "table", "x", "y", "z"])
@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json(client, diamond, field, token):
    # Raw bytes exercise JSON parsing; HTTPX's json encoder rejects NaN itself.
    diamond[field] = "TOKEN"
    body = json.dumps(diamond).replace('"TOKEN"', token)
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize("value", ["1.0", True, None, [], {}])
def test_wrong_numeric_type(client, diamond, value):
    diamond["carat"] = value
    assert client.post("/predict", json=diamond).status_code == 422


@pytest.mark.parametrize("field", ["price", "source_row", "Unnamed: 0", "unexpected"])
def test_extra_fields(client, diamond, field):
    diamond[field] = 123
    assert client.post("/predict", json=diamond).status_code == 422


def test_out_of_range_warning_without_input_change(client, diamond):
    diamond["carat"] = 5.0
    before = diamond.copy()
    response = client.post("/predict", json=diamond)
    assert response.status_code == 200
    assert any("carat" in warning for warning in response.json()["warnings"])
    assert diamond == before


def test_depth_mismatch_is_retained(client, diamond):
    diamond["depth"] = 50.0
    response = client.post("/predict", json=diamond)
    assert response.status_code == 200
    assert any("Depth differs" in warning for warning in response.json()["warnings"])


def test_direct_inference_consistency(client):
    info = client.get("/model/info").json()
    path = main.ROOT / "artifacts/tuning/HistGradientBoostingRegressor_pipeline.joblib"
    direct = joblib.load(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == info["model_sha256"]
    rows = pd.read_csv(main.ROOT / "data/processed/train.csv", usecols=main.FEATURES).iloc[[0, 100, 1000, 10000]]
    rows = rows[main.FEATURES]
    expected = direct.predict(rows)
    for payload, prediction in zip(rows.to_dict("records"), expected):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        np.testing.assert_allclose(response.json()["predicted_price_usd"], prediction, rtol=0, atol=1e-9)


def test_nonfinite_prediction_returns_503(client, diamond, monkeypatch):
    monkeypatch.setattr(main.app.state.pipeline, "predict", lambda frame: np.array([np.nan]))
    assert client.post("/predict", json=diamond).status_code == 503


def test_corrupt_model_refuses_startup(monkeypatch):
    original = Path.read_bytes
    def corrupted(path):
        data = original(path)
        return data + b"corrupt" if path.name == "HistGradientBoostingRegressor_pipeline.joblib" else data
    monkeypatch.setattr(Path, "read_bytes", corrupted)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        with TestClient(main.app):
            pass


def test_live_uvicorn(diamond):
    # Start an actual local HTTP server, make requests, and always stop it.
    with socket.socket() as temporary:
        temporary.bind(("127.0.0.1", 0))
        port = temporary.getsockname()[1]
    folder = main.ROOT / "artifacts/integration"
    with (folder / "uvicorn.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=main.ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=10) as http:
                deadline = time.monotonic() + 120
                while True:
                    assert process.poll() is None, "Server exited; inspect artifacts/backend/uvicorn.log"
                    try:
                        if http.get("/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    assert time.monotonic() < deadline, "Server startup timed out"
                    time.sleep(.25)
                response = http.post("/predict", json=diamond)
                assert response.status_code == 200
                (folder / "sample_response.json").write_text(json.dumps({"request": diamond, "response": response.json()}, indent=2), encoding="utf-8")
                assert http.get("/model/info").json()["artifact_verified"]
        finally:
            process.terminate()
            process.wait(timeout=20)
