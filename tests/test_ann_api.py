"""ANN API checks on synthetic inputs and training rows only."""
import json
import hashlib
from pathlib import Path

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.api import main


@pytest.fixture(scope="module")
def client():
    with pytest.MonkeyPatch.context() as patch:
        def reject_fit(*args, **kwargs):
            raise AssertionError("Training is forbidden")
        patch.setattr(Pipeline, "fit", reject_fit)
        with TestClient(main.app) as test_client:
            # Keras fit must also never be called by an inference request.
            patch.setattr(type(main.app.state.ann), "fit", reject_fit)
            yield test_client


@pytest.fixture
def diamond():
    return {"carat": 1.0, "cut": "Ideal", "color": "G", "clarity": "VS2",
            "depth": 61.5, "table": 57.0, "x": 6.4, "y": 6.4, "z": 3.94}


def test_ann_identity_and_repeatability(client, diamond):
    info = client.get("/model/info/ann").json()
    assert info["seed"] == 42 and info["transformed_feature_count"] == 26
    manifest = json.loads((main.ROOT / "artifacts/final_evaluation/frozen_selection.json").read_text())
    path = manifest["model_paths"]["phase4a_deep_raw"]
    assert hashlib.sha256((main.ROOT / path).read_bytes()).hexdigest() == info["model_sha256"]
    response = client.post("/predict/ann", json=diamond)
    assert response.status_code == 200
    assert np.isfinite(response.json()["predicted_price_usd"])
    assert client.post("/predict/ann", json=diamond).json() == response.json()
    assert client.get("/model/info").json()["model_type"] == "HistGradientBoostingRegressor"


def test_ann_direct_saved_model_parity(client):
    from tensorflow import keras
    model = keras.models.load_model(main.ROOT / "artifacts/deep_learning/final/selected_ann.keras", compile=False)
    preprocessing = joblib.load(main.ROOT / "artifacts/preparation/preprocessor.joblib")
    rows = pd.read_csv(main.ROOT / "data/processed/train.csv", usecols=main.FEATURES).iloc[[0, 100, 1000, 10000]][main.FEATURES]
    transformed = preprocessing.transform(rows).astype("float32")
    assert transformed.shape == (4, 26)
    expected = model(transformed, training=False).numpy().ravel()
    for row, prediction in zip(rows.to_dict("records"), expected):
        actual = client.post("/predict/ann", json=row)
        assert actual.status_code == 200
        np.testing.assert_allclose(actual.json()["predicted_price_usd"], prediction, rtol=1e-6, atol=.01)


@pytest.mark.parametrize("field,value", [("x", 0), ("y", -1), ("cut", "bad"), ("price", 1000), ("carat", "1")])
def test_ann_validation(client, diamond, field, value):
    diamond[field] = value
    assert client.post("/predict/ann", json=diamond).status_code == 422


def test_ann_missing_field(client, diamond):
    del diamond["clarity"]
    assert client.post("/predict/ann", json=diamond).status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_ann_nonfinite(client, diamond, token):
    diamond["x"] = "TOKEN"
    body = json.dumps(diamond).replace('"TOKEN"', token)
    assert client.post("/predict/ann", content=body, headers={"content-type": "application/json"}).status_code == 422


def test_ann_domain_warning(client, diamond):
    diamond["carat"] = 5.0
    result = client.post("/predict/ann", json=diamond)
    assert result.status_code == 200 and result.json()["warnings"]


def test_verified_insight_reports(client):
    info = client.get("/model/info").json()
    assert len(info["comparison_models"]) == 3
    insight = client.get("/model/insights")
    assert insight.status_code == 200
    assert len(insight.json()["importance"]) == 9
    assert insight.json()["importance"][0]["feature"] == "y"


def test_corrupt_ann_refuses_startup(monkeypatch):
    original = Path.read_bytes
    def corrupted(path):
        data = original(path)
        return data + b"corrupt" if path.name == "selected_ann.keras" else data
    monkeypatch.setattr(Path, "read_bytes", corrupted)
    with pytest.raises(RuntimeError, match="Frozen ANN SHA256 mismatch"):
        with TestClient(main.app):
            pass
