"""Freeze candidates and predict the held-out test set once; never fit models."""
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from threadpoolctl import threadpool_limits

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
folder = Path("artifacts/final_evaluation")
folder.mkdir(exist_ok=True)
manifest_path = folder / "frozen_selection.json"
if manifest_path.exists():
    raise SystemExit("Selection already frozen. Use explain_final.py to reproduce reports from saved predictions; do not repeat test evaluation.")

preparation = json.loads(Path("artifacts/preparation/preparation_report.json").read_text())
selection = json.loads(Path("artifacts/deep_learning/final/candidate_selection.json").read_text())
assert selection["provisional_production_model"] == "HistGradientBoostingRegressor"
assert selection["ann_representative_seed"] == 42 and not selection["refinement_accepted"]
paths = {
    "HistGradientBoostingRegressor": "artifacts/tuning/HistGradientBoostingRegressor_pipeline.joblib",
    "RandomForestRegressor": "artifacts/tuning/RandomForestRegressor_pipeline.joblib",
    "phase4a_deep_raw": "artifacts/deep_learning/final/selected_ann.keras",
}
preprocessor = joblib.load("artifacts/preparation/preprocessor.joblib")
columns = preprocessor.feature_names_in_.tolist()
assert not {"price", "source_row", "Unnamed: 0"}.intersection(columns)
assert preprocessor.get_feature_names_out().tolist() == preparation["transformed_features"]
models = {name: joblib.load(path) for name, path in paths.items() if name != "phase4a_deep_raw"}
ann = keras.models.load_model(paths["phase4a_deep_raw"], compile=False)
assert ann.input_shape == (None, 26) and ann.output_shape == (None, 1)
for name, pipeline in models.items():
    assert pipeline.feature_names_in_.tolist() == columns
    assert pipeline.named_steps["model"].n_features_in_ == 26
    assert pipeline.named_steps["preprocessing"].get_feature_names_out().tolist() == preparation["transformed_features"]

# Snapshot existing project inputs before reading any held-out target outcomes.
protected = [p for root in ["data", "artifacts", "src"] for p in Path(root).rglob("*")
             if p.is_file() and folder not in p.parents and "__pycache__" not in p.parts]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
assert hashes[str(Path("data/raw/10-diamonds (1).csv"))] == preparation["raw_sha256"]
baseline = json.loads(Path("artifacts/baselines/experiment_config.json").read_text())
for path, expected in baseline["input_sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
manifest = {
    "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    "primary_model": "HistGradientBoostingRegressor", "ann_seed": 42,
    "selection_basis": "Prior validation MAE only; test outcomes cannot change selection.",
    "model_paths": paths, "input_columns": columns,
    "transformed_features": preparation["transformed_features"],
    "model_parameters": {n: m.named_steps["model"].get_params() for n, m in models.items()},
    "ann_configuration": ann.get_config(),
    "validation_results": pd.read_csv("artifacts/deep_learning/final/model_comparison.csv").to_dict("records"),
    "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    "protected_sha256": hashes,
    "evaluation_policy": "One prediction pass per frozen candidate. Save predictions before diagnostics; reports reuse those predictions. No fitting or tuning.",
}
with manifest_path.open("x", encoding="utf-8") as file:
    json.dump(manifest, file, indent=2)
print("Selection manifest frozen before loading test outcomes.", flush=True)

frames = {}
for split in ["train", "validation", "test"]:
    frame = pd.read_csv(f"data/processed/{split}.csv")
    assert len(frame) == preparation["splits"][split]["rows"]
    assert set(frame.columns) == {*columns, "price", "source_row"}
    assert not frame.isna().any().any()
    assert np.isfinite(frame.select_dtypes(include="number")).all().all()
    with np.load(f"data/processed/{split}_transformed.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_row"], frame["source_row"])
        np.testing.assert_array_equal(saved["y"], frame["price"])
        assert saved["feature_names"].tolist() == preparation["transformed_features"]
        assert saved["X"].shape == (len(frame), 26)
        np.testing.assert_allclose(saved["X"], preprocessor.transform(frame[columns]), rtol=1e-12, atol=1e-12)
    frames[split] = frame
for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert set(frames[left]["source_row"]).isdisjoint(frames[right]["source_row"])
    assert set(frames[left][columns].itertuples(index=False, name=None)).isdisjoint(
        frames[right][columns].itertuples(index=False, name=None))
assert len(frames["test"]) == 8089
scaler = preprocessor.named_steps["columns"].named_transformers_["numeric"]
assert scaler.n_samples_seen_ == len(frames["train"])
np.testing.assert_allclose(scaler.mean_, frames["train"][preparation["numeric_columns"]].mean())
for pipeline in models.values():
    for split in frames:
        np.testing.assert_allclose(pipeline.named_steps["preprocessing"].transform(frames[split][columns]),
                                   preprocessor.transform(frames[split][columns]), rtol=1e-12, atol=1e-12)

test = frames["test"]
for name in paths:
    # Persist each candidate immediately, so a later reporting failure never needs new predictions.
    with threadpool_limits(limits=1):
        if name == "phase4a_deep_raw":
            X = preprocessor.transform(test[columns]).astype("float32")
            prediction = ann(X, training=False).numpy().ravel().astype("float64")
        else:
            prediction = models[name].predict(test[columns])
    assert prediction.shape == (8089,) and np.isfinite(prediction).all()
    pd.DataFrame({"source_row": test["source_row"], "actual_price": test["price"],
                  "predicted_price": prediction}).to_csv(folder / f"{name}_test_predictions.csv", index=False)
    print(f"Saved one-time predictions: {name}, {len(prediction):,} rows", flush=True)
for path, expected in hashes.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
(folder / "evaluation_verification.json").write_text(json.dumps({
    "completed_at_utc": datetime.now(timezone.utc).isoformat(), "test_rows": 8089,
    "checks_passed": ["row alignment", "26 transformed inputs", "finite inputs and predictions",
        "no target or identifier input", "all split groups disjoint", "training-only scaler",
        "persisted preprocessing compatibility", "all previous files unchanged"],
    "test_prediction_passes_per_model": 1, "model_fits": 0, "failed_checks": [],
}, indent=2), encoding="utf-8")
print("Frozen evaluation complete. Run explain_final.py for metrics, plots, and model card.")
