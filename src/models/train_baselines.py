import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from threadpoolctl import threadpool_limits


processed_dir = Path("data/processed")
preparation_dir = Path("artifacts/preparation")
output_dir = Path("artifacts/baselines")
output_dir.mkdir(parents=True, exist_ok=True)
report = json.loads((preparation_dir / "preparation_report.json").read_text())
preprocessor = joblib.load(preparation_dir / "preprocessor.joblib")
input_columns = preprocessor.feature_names_in_.tolist()
feature_names = preprocessor.get_feature_names_out().tolist()
forbidden = {"price", "source_row", "Unnamed: 0"}
assert not forbidden.intersection(input_columns)
assert set(input_columns) == set(report["numeric_columns"] + report["categorical_columns"])
assert feature_names == report["transformed_features"]
assert all(name.split("__", 1)[-1] not in forbidden for name in feature_names)

# Hash preparation inputs so this experiment is tied to exact artifacts.
input_paths = [Path("data/raw/10-diamonds (1).csv"),
               preparation_dir / "preparation_report.json",
               preparation_dir / "preprocessor.joblib"]
input_paths += list(processed_dir.glob("*.csv")) + list(processed_dir.glob("*.npz"))
input_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
assert input_hashes[str(input_paths[0])] == report["raw_sha256"]

splits = {}
row_sets = []
feature_sets = []
for name in ["train", "validation", "test"]:
    frame = pd.read_csv(processed_dir / f"{name}.csv")
    assert set(frame.columns) == {"source_row", "price", *input_columns}
    assert not frame.isna().any().any()
    with np.load(processed_dir / f"{name}_transformed.npz", allow_pickle=False) as saved:
        features = saved["X"].copy()
        target = saved["y"].copy()
        assert saved["feature_names"].tolist() == feature_names
        np.testing.assert_array_equal(saved["source_row"], frame["source_row"])
    assert features.shape == tuple(report["splits"][name]["transformed_shape"])
    assert target.shape == (len(frame),)
    assert np.isfinite(features).all() and np.isfinite(target).all()
    np.testing.assert_array_equal(target, frame["price"].to_numpy())
    np.testing.assert_allclose(features, preprocessor.transform(frame[input_columns]), rtol=1e-12, atol=1e-12)
    row_sets.append(set(frame["source_row"]))
    feature_sets.append(set(frame[input_columns].itertuples(index=False, name=None)))
    print(f"{name} artifacts verified: X={features.shape}, y={target.shape}", flush=True)
    # Test arrays are checked for integrity only and never retained for evaluation.
    if name != "test":
        splits[name] = {"X": features, "y": target, "raw": frame[input_columns]}

for first, second in [(0, 1), (0, 2), (1, 2)]:
    assert row_sets[first].isdisjoint(row_sets[second])
    assert feature_sets[first].isdisjoint(feature_sets[second])
assert sum(map(len, row_sets)) == report["clean_rows"]
X_train, y_train = splits["train"]["X"], splits["train"]["y"]
X_validation, y_validation = splits["validation"]["X"], splits["validation"]["y"]
scaler = preprocessor.named_steps["columns"].named_transformers_["numeric"]
assert scaler.n_samples_seen_ == len(y_train)
np.testing.assert_allclose(scaler.mean_, splits["train"]["raw"][report["numeric_columns"]].mean())

# Fixed first-experiment settings; no search or validation-driven tuning.
models = {
    "DummyRegressor": DummyRegressor(strategy="mean"),
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, solver="svd"),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=1),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.1, max_leaf_nodes=31,
        early_stopping=False, random_state=42),
}
configurations = {name: model.get_params() for name, model in models.items()}
results = []
validation_predictions = {}
warning_messages = []
for name, model in models.items():
    started = perf_counter()
    # One computational thread makes repeat checks predictable across this run.
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
        assert model.n_features_in_ == len(feature_names)
        train_prediction = model.predict(X_train)
        validation_prediction = model.predict(X_validation)
        assert np.isfinite(train_prediction).all() and np.isfinite(validation_prediction).all()
        # Repeat the same fit, without changing any settings or using validation in fit.
        repeated = clone(model)
        repeated.fit(X_train, y_train)
        np.testing.assert_allclose(train_prediction, repeated.predict(X_train), rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(validation_prediction, repeated.predict(X_validation), rtol=1e-12, atol=1e-9)
        del repeated
    for warning in caught:
        message = f"{name}: {warning.category.__name__}: {warning.message}"
        warning_messages.append(message)
        print(message, flush=True)
    row = {"model": name}
    for partition, actual, predicted in [
        ("train", y_train, train_prediction),
        ("validation", y_validation, validation_prediction),
    ]:
        row[f"{partition}_mae_usd"] = mean_absolute_error(actual, predicted)
        row[f"{partition}_rmse_usd"] = root_mean_squared_error(actual, predicted)
        row[f"{partition}_r2"] = r2_score(actual, predicted)
    row["mae_gap_usd"] = row["validation_mae_usd"] - row["train_mae_usd"]
    row["rmse_gap_usd"] = row["validation_rmse_usd"] - row["train_rmse_usd"]
    row["r2_drop"] = row["train_r2"] - row["validation_r2"]
    row["seconds_including_repeat_check"] = perf_counter() - started
    results.append(row)
    validation_predictions[name] = validation_prediction
    joblib.dump(model, output_dir / f"{name}.joblib", compress=3)
    print(f"{name}: validation MAE=${row['validation_mae_usd']:.2f}; repeat-fit check passed", flush=True)

comparison = pd.DataFrame(results).sort_values("validation_mae_usd").reset_index(drop=True)
comparison.to_csv(output_dir / "model_comparison.csv", index=False)
print("\n" + comparison.round(4).to_string(index=False), flush=True)
best_name = comparison.loc[0, "model"]
best_prediction = validation_predictions[best_name]
restored = joblib.load(output_dir / f"{best_name}.joblib")
with threadpool_limits(limits=1):
    # Verify raw characteristics -> persisted preprocessing -> persisted model.
    reloaded_prediction = restored.predict(preprocessor.transform(splits["validation"]["raw"]))
np.testing.assert_allclose(best_prediction, reloaded_prediction, rtol=1e-12, atol=1e-9)

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_validation, best_prediction, s=12, alpha=0.25)
limits = [min(y_validation.min(), best_prediction.min()), max(y_validation.max(), best_prediction.max())]
ax.plot(limits, limits, color="darkred", linestyle="--", label="Perfect prediction")
ax.set(title=f"Validation: Actual vs Predicted — {best_name}",
       xlabel="Actual Price (USD)", ylabel="Predicted Price (USD)")
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "validation_actual_vs_predicted.png", dpi=200)
plt.close(fig)

residuals = y_validation - best_prediction
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(residuals, bins=70, color="steelblue", edgecolor="white")
ax.axvline(0, color="darkred", linestyle="--", label="Zero error")
ax.set(title=f"Validation Residuals — {best_name}",
       xlabel="Actual − Predicted Price (USD)", ylabel="Number of Diamonds")
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "validation_residual_distribution.png", dpi=200)
plt.close(fig)

for path in input_paths:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == input_hashes[str(path)]
experiment = {
    "phase": "3A", "selection_metric": "Lowest validation MAE in USD", "best_model": best_name,
    "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                 "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
                 "matplotlib": matplotlib.__version__},
    "configurations": configurations, "input_sha256": input_hashes,
    "input_columns": input_columns, "transformed_features": feature_names,
    "sizes": {name: len(split["y"]) for name, split in splits.items()},
    "configuration_reasons": {
        "DummyRegressor": "Training mean reference, required to judge learned models.",
        "LinearRegression": "Unregularized additive linear baseline.",
        "Ridge": "Mild L2 regularization (alpha=1); SVD handles redundant one-hot columns.",
        "RandomForestRegressor": "100 trees and minimum leaf size 2 balance capacity and runtime.",
        "HistGradientBoostingRegressor": "100 iterations, learning rate 0.1, 31 leaves; early stopping off so all training rows are used.",
    },
    "reproducibility": "All five cloned models refitted on identical training arrays; train/validation predictions matched at rtol=1e-12, atol=1e-9. One computational thread.",
    "verification": "No refitting of preprocessing; no forbidden inputs; CSV/NPZ/pipeline consistency; finite data; no partition overlap; best-model reload through persisted preprocessing matches; all input hashes unchanged.",
    "test_policy": "Artifact integrity checked only. No test predictions, metrics, selection or tuning.",
    "warnings": warning_messages,
}
(output_dir / "experiment_config.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
print(f"\nBest validation MAE: {best_name}. Warnings: {len(warning_messages)}. Reload and hash checks passed.", flush=True)
print("Phase 3A complete. No test evaluation performed.", flush=True)
