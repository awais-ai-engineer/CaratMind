import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import warnings

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import tensorflow as tf
from tensorflow import keras
from threadpoolctl import threadpool_limits


tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
output_dir = Path("artifacts/deep_learning/final")
output_dir.mkdir(parents=True, exist_ok=True)
phase4a_dir = Path("artifacts/deep_learning")
phase4a_config = json.loads((phase4a_dir / "experiment_config.json").read_text())
preparation = json.loads(Path("artifacts/preparation/preparation_report.json").read_text())
ml_config = json.loads(Path("artifacts/tuning/experiment_metadata.json").read_text())
preprocessor = joblib.load("artifacts/preparation/preprocessor.joblib")
input_columns = preprocessor.feature_names_in_.tolist()
feature_names = preprocessor.get_feature_names_out().tolist()
assert input_columns == ml_config["input_columns"]
assert feature_names == preparation["transformed_features"] and len(feature_names) == 26
assert not {"price", "source_row", "Unnamed: 0"}.intersection(input_columns)

# Protect existing project outputs, excluding this phase's output directory.
protected_paths = [Path("data/raw/10-diamonds (1).csv")]
protected_paths += sorted(Path("src").rglob("*.py"))
for folder in ["data/processed", "artifacts/preparation", "artifacts/baselines", "artifacts/tuning", "artifacts/figures"]:
    protected_paths += sorted(path for path in Path(folder).rglob("*") if path.is_file())
protected_paths += sorted(path for path in phase4a_dir.iterdir() if path.is_file())
protected_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected_paths}
for previous_config in [phase4a_config, ml_config]:
    for path, expected in previous_config["protected_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
assert hashlib.sha256(Path("src/models/train_ann.py").read_bytes()).hexdigest() == phase4a_config["current_training_script_sha256"]

convergence = []
for path in sorted(phase4a_dir.glob("*_history.csv")):
    history = pd.read_csv(path)
    assert np.isfinite(history.to_numpy()).all()
    row = {"experiment": path.stem.removesuffix("_history"), "epochs": len(history),
           "best_epoch": int(history["val_mae_usd"].idxmin()) + 1,
           "best_validation_mae_usd": history["val_mae_usd"].min(),
           "last_validation_mae_usd": history["val_mae_usd"].iloc[-1]}
    for start, end in [(100, 125), (125, 150)]:
        window = history.iloc[start:end]
        row[f"mean_val_mae_epochs_{start+1}_{end}"] = window["val_mae_usd"].mean()
        row[f"mean_train_loss_epochs_{start+1}_{end}"] = window["loss"].mean()
    convergence.append(row)
pd.DataFrame(convergence).to_csv(output_dir / "phase4a_convergence.csv", index=False)
print(pd.DataFrame(convergence).round(4).to_string(index=False), flush=True)

frames = {}
arrays = {}
for name in ["train", "validation"]:
    frame = pd.read_csv(f"data/processed/{name}.csv")
    assert set(frame.columns) == {"source_row", "price", *input_columns}
    assert not frame.isna().any().any()
    with np.load(f"data/processed/{name}_transformed.npz", allow_pickle=False) as saved:
        X, y = saved["X"].copy(), saved["y"].astype("float64")
        assert saved["feature_names"].tolist() == feature_names
        np.testing.assert_array_equal(saved["source_row"], frame["source_row"])
        np.testing.assert_array_equal(y, frame["price"])
    assert X.shape == tuple(preparation["splits"][name]["transformed_shape"])
    assert y.shape == (len(frame),) and np.isfinite(X).all() and np.isfinite(y).all()
    np.testing.assert_allclose(X, preprocessor.transform(frame[input_columns]), rtol=1e-12, atol=1e-12)
    frames[name], arrays[name] = frame, {"X": X.astype("float32"), "y": y}
assert set(frames["train"]["source_row"]).isdisjoint(frames["validation"]["source_row"])
assert set(frames["train"][input_columns].itertuples(index=False, name=None)).isdisjoint(
    set(frames["validation"][input_columns].itertuples(index=False, name=None)))
X_train, y_train = arrays["train"]["X"], arrays["train"]["y"]
X_validation, y_validation = arrays["validation"]["X"], arrays["validation"]["y"]
scaler = preprocessor.named_steps["columns"].named_transformers_["numeric"]
assert scaler.n_samples_seen_ == len(X_train)
np.testing.assert_allclose(scaler.mean_, frames["train"][preparation["numeric_columns"]].mean())

# One fixed configuration; validation selects checkpoints, not seeds.
seeds = [42, 7, 2026]
configuration = {
    "phase": "4B", "seeds": seeds, "representative_seed": 42, "target": "original price (USD)",
    "architecture": [26, 128, 64, 32, 1], "hidden_activation": "relu", "output_activation": "linear",
    "optimizer": "Adam", "initial_learning_rate": 0.001, "loss": "MSE", "batch_size": 64,
    "maximum_epochs": 225,
    "reduce_lr": {"monitor": "val_mae_usd", "mode": "min", "factor": 0.5, "patience": 8,
                  "min_delta": 0.5, "min_lr": 0.00001},
    "early_stopping": {"monitor": "val_mae_usd", "mode": "min", "patience": 25,
                       "min_delta": 0, "restore_best_weights": True},
    "ann_rule": "Use the refined configuration only if both its three-seed mean validation MAE and its preassigned seed-42 MAE are below the Phase 4A deep_raw reference. Otherwise retain Phase 4A deep_raw. Never choose the lowest-scoring seed.",
    "production_rule": "Compare selected ANN representative with the two tuned ML models by validation MAE. A refined ANN must also have three-seed mean MAE below the best ML MAE to be preferred for production. Operational measurements are descriptive, not a tuning objective.",
    "stability_definition": "Mean and sample standard deviation (ddof=1) over seeds on one unchanged validation set; not sampling uncertainty, confidence intervals or statistical significance.",
    "limitations": "Only one refined protocol, three seeds and a repeatedly reused validation split. Schedule and increased epoch budget change together, so their individual effects are not isolated. Earlier ANN and ML seed variability was not measured comparably.",
    "inference_protocol": "CPU, one thread, warmed process; end-to-end preprocessing plus prediction for the first 1 and 256 validation rows, 3 warmups and 15 measured calls. Excludes package import/model load time. File formats/compression differ.",
    "reload_tolerance": {"rtol": 1e-6, "atol_usd": 0.01},
    "feature_names": feature_names, "input_columns": input_columns,
    "python": platform.python_version(),
    "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    "protected_sha256": protected_hashes,
    "test_policy": "Test files hashed only. No test arrays loaded or test evaluation.",
    "reproducibility": "Fixed seeds, deterministic CPU operations and single-thread TensorFlow; hardware/version changes can change numerical results.",
}
(output_dir / "experiment_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
results = []
prediction_frames = []
warning_messages = []
for seed in seeds:
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.Input(shape=(26,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="linear"),
    ], name=f"refined_ann_seed_{seed}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse",
                  metrics=[keras.metrics.MeanAbsoluteError(name="mae_usd")])
    assert model.count_params() == 13825
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train.astype("float32").reshape(-1, 1)))
    train_data = train_data.shuffle(len(X_train), seed=seed, reshuffle_each_iteration=True)
    train_data = train_data.batch(64).with_options(options).prefetch(1)
    validation_data = tf.data.Dataset.from_tensor_slices((X_validation, y_validation.astype("float32").reshape(-1, 1)))
    validation_data = validation_data.batch(64).with_options(options).prefetch(1)
    print(f"\nTraining fixed refinement, seed {seed}, at most 225 epochs", flush=True)
    started = perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            history = model.fit(train_data, validation_data=validation_data, epochs=225, shuffle=False, verbose=0,
                callbacks=[
                    keras.callbacks.ReduceLROnPlateau(monitor="val_mae_usd", mode="min", factor=0.5,
                        patience=8, min_delta=0.5, min_lr=0.00001),
                    keras.callbacks.EarlyStopping(monitor="val_mae_usd", mode="min", patience=25,
                        restore_best_weights=True),
                    keras.callbacks.CSVLogger(str(output_dir / f"seed_{seed}_history.csv")),
                    keras.callbacks.TerminateOnNaN(),
                ])
        duration = perf_counter() - started
        warning_messages.extend(f"seed {seed}: {item.category.__name__}: {item.message}" for item in caught)
        history_frame = pd.read_csv(output_dir / f"seed_{seed}_history.csv")
        assert np.isfinite(history_frame.to_numpy()).all()
        assert len(history_frame) == len(history.history["loss"])
        best_index = int(history_frame["val_mae_usd"].idxmin())
        train_prediction = model(X_train, training=False).numpy().ravel().astype("float64")
        prediction = model(X_validation, training=False).numpy().ravel().astype("float64")
        assert np.isfinite(train_prediction).all() and np.isfinite(prediction).all()
        path = output_dir / f"seed_{seed}.keras"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.save(path)
            restored = keras.models.load_model(path, compile=False)
        warning_messages.extend(f"seed {seed}: {item.category.__name__}: {item.message}" for item in caught)
        # Also check prediction through the persisted preprocessing pipeline.
        raw_transformed = preprocessor.transform(frames["validation"][input_columns]).astype("float32")
        np.testing.assert_allclose(prediction, restored(raw_transformed, training=False).numpy().ravel(), rtol=1e-6, atol=0.01)
        row = {"seed": seed, "epochs": len(history_frame), "restored_epoch": best_index + 1,
               "training_seconds": duration, "trainable_parameters": model.count_params(),
               "initial_lr": history_frame["learning_rate"].iloc[0],
               "final_logged_lr": history_frame["learning_rate"].iloc[-1],
               "lr_reductions": int((history_frame["learning_rate"].diff() < 0).sum()),
               "best_val_epoch_loss": history_frame["val_loss"].iloc[best_index],
               "best_val_epoch_mae": history_frame["val_mae_usd"].iloc[best_index]}
        for partition, actual, predicted in [("train", y_train, train_prediction),
                                            ("validation", y_validation, prediction)]:
            row[f"{partition}_mae_usd"] = mean_absolute_error(actual, predicted)
            row[f"{partition}_rmse_usd"] = root_mean_squared_error(actual, predicted)
            row[f"{partition}_r2"] = r2_score(actual, predicted)
        np.testing.assert_allclose(row["validation_mae_usd"], row["best_val_epoch_mae"], rtol=1e-5, atol=0.02)
        row["mae_gap_usd"] = row["validation_mae_usd"] - row["train_mae_usd"]
        row["negative_predictions"] = int((prediction < 0).sum())
        results.append(row)
        pd.DataFrame(results).to_csv(output_dir / "seed_metrics.csv", index=False)
        predictions = pd.DataFrame({"model": f"refined_seed_{seed}", "seed": seed,
            "source_row": frames["validation"]["source_row"], "actual_price": y_validation,
            "predicted_price": prediction})
        prediction_frames.append(predictions)
        pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "validation_predictions.csv", index=False)
        print(f"Completed seed {seed}: epochs={len(history_frame)}, best={best_index+1}, "
              f"validation MAE=${row['validation_mae_usd']:.2f}, LR reductions={row['lr_reductions']}", flush=True)
    except Exception as error:
        (output_dir / "failure.json").write_text(json.dumps({"seed": seed, "error": repr(error),
            "completed_seeds": [row["seed"] for row in results]}, indent=2))
        raise

seed_metrics = pd.DataFrame(results)
metric_columns = ["train_mae_usd", "train_rmse_usd", "train_r2", "validation_mae_usd", "validation_rmse_usd", "validation_r2", "mae_gap_usd"]
stability = seed_metrics[metric_columns].agg(["mean", "std"]).T
stability.index.name = "metric"
stability.to_csv(output_dir / "seed_stability.csv")
phase4a = pd.read_csv(phase4a_dir / "experiment_comparison.csv")
reference = phase4a.loc[phase4a["model"] == "deep_raw"].iloc[0]
representative = seed_metrics.loc[seed_metrics["seed"] == 42].iloc[0]
use_refinement = (stability.loc["validation_mae_usd", "mean"] < reference["validation_mae_usd"]
                  and representative["validation_mae_usd"] < reference["validation_mae_usd"])
ann_path = output_dir / "seed_42.keras" if use_refinement else phase4a_dir / "deep_raw.keras"
ann_name = "refined_ann_seed_42" if use_refinement else "phase4a_deep_raw"
ann_row = representative if use_refinement else reference
ann = keras.models.load_model(ann_path, compile=False)
# Save the preassigned representative, never the best-scoring individual seed.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ann.save(output_dir / "selected_ann.keras")
warning_messages.extend(f"selected ANN: {item.category.__name__}: {item.message}" for item in caught)
summary_lines = []
ann.summary(print_fn=summary_lines.append)
(output_dir / "selected_ann_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
selected_prediction = ann(X_validation, training=False).numpy().ravel().astype("float64")
reloaded = keras.models.load_model(output_dir / "selected_ann.keras", compile=False)
np.testing.assert_allclose(selected_prediction, reloaded(X_validation, training=False).numpy().ravel(), rtol=1e-6, atol=0.01)

ml_results = pd.read_csv("artifacts/tuning/validation_comparison.csv")
ml_results = ml_results.loc[ml_results["experiment"] == "tuned"]
ml_predictions = pd.read_csv("artifacts/tuning/validation_errors.csv")
comparison_rows = []
price_frames = prediction_frames.copy()
benchmark_rows = []
for name in ["HistGradientBoostingRegressor", "RandomForestRegressor", ann_name]:
    if name == ann_name:
        expected = ann_row
        train_values = ann(X_train, training=False).numpy().ravel().astype("float64")
        values = selected_prediction
        model_bytes = (output_dir / "selected_ann.keras").stat().st_size
    else:
        expected = ml_results.loc[ml_results["model"] == name].iloc[0]
        model_path = Path(f"artifacts/tuning/{name}_pipeline.joblib")
        pipeline = joblib.load(model_path)
        with threadpool_limits(limits=1):
            train_values = pipeline.predict(frames["train"][input_columns])
            values = pipeline.predict(frames["validation"][input_columns])
        saved = ml_predictions.loc[ml_predictions["model"] == name]
        np.testing.assert_array_equal(saved["source_row"], frames["validation"]["source_row"])
        np.testing.assert_array_equal(saved["actual_price"], y_validation)
        np.testing.assert_allclose(values, saved["predicted_price"], rtol=1e-10, atol=1e-7)
        model_bytes = model_path.stat().st_size
    row = {"model": name, "model_bytes": model_bytes}
    for partition, actual, predicted in [("train", y_train, train_values), ("validation", y_validation, values)]:
        for metric, result in [("mae_usd", mean_absolute_error(actual, predicted)),
                               ("rmse_usd", root_mean_squared_error(actual, predicted)), ("r2", r2_score(actual, predicted))]:
            row[f"{partition}_{metric}"] = result
            np.testing.assert_allclose(result, expected[f"{partition}_{metric}"], rtol=1e-6, atol=1e-5)
    row["mae_gap_usd"] = row["validation_mae_usd"] - row["train_mae_usd"]
    comparison_rows.append(row)
    price_frames.append(pd.DataFrame({"model": name, "source_row": frames["validation"]["source_row"],
                                     "actual_price": y_validation, "predicted_price": values}))
    for batch_size in [1, 256]:
        raw_batch = frames["validation"][input_columns].iloc[:batch_size]
        times = []
        with threadpool_limits(limits=1):
            for repeat in range(18):
                started = perf_counter()
                if name == ann_name:
                    output = ann(preprocessor.transform(raw_batch).astype("float32"), training=False).numpy()
                else:
                    output = pipeline.predict(raw_batch)
                elapsed = 1000 * (perf_counter() - started)
                assert np.isfinite(output).all()
                if repeat >= 3:
                    times.append(elapsed)
        benchmark_rows.append({"model": name, "batch_rows": batch_size,
            "median_milliseconds": np.median(times), "p90_milliseconds": np.quantile(times, 0.9), "repeats": len(times)})
comparison = pd.DataFrame(comparison_rows).sort_values("validation_mae_usd")
comparison.to_csv(output_dir / "model_comparison.csv", index=False)
pd.DataFrame(benchmark_rows).to_csv(output_dir / "inference_benchmark.csv", index=False)

errors = pd.concat(price_frames, ignore_index=True)
errors["absolute_error_usd"] = (errors["actual_price"] - errors["predicted_price"]).abs()
errors["price_range"] = pd.cut(errors["actual_price"], [0, 1000, 3000, 5000, 10000, np.inf],
    labels=["0-999", "1000-2999", "3000-4999", "5000-9999", "10000+"], right=False)
assert errors["price_range"].notna().all()
price_summary = errors.groupby(["model", "price_range"], observed=True).agg(
    count=("actual_price", "size"), mae_usd=("absolute_error_usd", "mean"))
price_summary.to_csv(output_dir / "price_range_mae.csv")
per_seed_ranges = price_summary.loc[[f"refined_seed_{seed}" for seed in seeds]].reset_index()
range_stability = per_seed_ranges.groupby("price_range", observed=True)["mae_usd"].agg(["mean", "std"])
range_stability.to_csv(output_dir / "price_range_seed_stability.csv")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
for index, seed in enumerate(seeds):
    history = pd.read_csv(output_dir / f"seed_{seed}_history.csv")
    epoch = history["epoch"] + 1
    axes[index, 0].plot(epoch, history["loss"], label="Training MSE")
    axes[index, 0].plot(epoch, history["val_loss"], label="Validation MSE")
    axes[index, 0].set(yscale="log", title=f"Seed {seed}: Loss", xlabel="Epoch", ylabel="MSE (USD squared)")
    axes[index, 1].plot(epoch, history["mae_usd"], label="Training MAE")
    axes[index, 1].plot(epoch, history["val_mae_usd"], label="Validation MAE")
    axes[index, 1].set(yscale="log", title=f"Seed {seed}: MAE and Learning Rate", xlabel="Epoch", ylabel="MAE (USD)")
    lr_axis = axes[index, 1].twinx()
    lr_axis.step(epoch, history["learning_rate"], where="post", color="gray", alpha=0.6)
    lr_axis.set(yscale="log", ylabel="Learning Rate")
    for ax in axes[index]:
        ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(output_dir / "training_stability.png", dpi=180)
plt.close(fig)
fig, ax = plt.subplots(figsize=(11, 6))
price_summary.loc[["HistGradientBoostingRegressor", "RandomForestRegressor", ann_name], "mae_usd"].unstack("model").plot.bar(ax=ax, rot=0)
ax.set(title="Selected Candidates: Validation MAE by Actual Price", xlabel="Actual Price Range (USD)", ylabel="MAE (USD)")
fig.tight_layout()
fig.savefig(output_dir / "price_range_comparison.png", dpi=180)
plt.close(fig)

best_ml = ml_results.sort_values("validation_mae_usd").iloc[0]
ann_eligible = (not use_refinement or stability.loc["validation_mae_usd", "mean"] < best_ml["validation_mae_usd"])
prefer_ann = ann_eligible and ann_row["validation_mae_usd"] < best_ml["validation_mae_usd"]
production = ann_name if prefer_ann else best_ml["model"]
for path in protected_paths:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == protected_hashes[str(path)]
selection = {
    "provisional_production_model": production,
    "production_artifact": str(output_dir / "selected_ann.keras") if prefer_ann else f"artifacts/tuning/{production}_pipeline.joblib",
    "strongest_ann": ann_name, "ann_artifact": str(output_dir / "selected_ann.keras"),
    "ann_source_artifact": str(ann_path), "ann_representative_seed": 42,
    "refinement_accepted": bool(use_refinement),
    "three_seed_mean_mae_usd": float(stability.loc["validation_mae_usd", "mean"]),
    "three_seed_std_mae_usd": float(stability.loc["validation_mae_usd", "std"]),
    "rationale": "Applied the rules recorded before training. Seed 42 is preassigned, not the best seed. No significance claim; seed variability does not measure uncertainty over datasets.",
    "operational_notes": "ANN requires existing preprocessing plus TensorFlow CPU and float32 conversion. Saved ML pipelines include preprocessing and use scikit-learn/joblib. Benchmarks are warm-process local measurements, not service latency or cold-start guarantees. No model deployed.",
    "warning_messages": warning_messages,
    "runtime_notices": "TensorFlow may report native Windows CPU-only execution and Keras compatibility deprecations; Python warnings are retained above.",
    "checks": "Passed: original artifacts/data/test hashes unchanged; feature and target alignment; 26 features; no target/group leakage; train-only preprocessing; finite histories and predictions; all seed and selected model reloads; selected checkpoint MAE; regenerated ML train/validation metrics match stored results. No test arrays loaded.",
    "limitations": configuration["limitations"],
}
(output_dir / "candidate_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
print("\nPer-seed metrics:\n" + seed_metrics.round(5).to_string(index=False), flush=True)
print("\nSeed stability (sample SD):\n" + stability.round(6).to_string(), flush=True)
print("\nCandidate comparison:\n" + comparison.round(5).to_string(index=False), flush=True)
print("\nPrice range MAE:\n" + price_summary.round(3).to_string(), flush=True)
print("\nWarm-process inference:\n" + pd.DataFrame(benchmark_rows).round(3).to_string(index=False), flush=True)
print(f"\nProduction candidate: {production}; strongest ANN: {ann_name}. All checks passed. No test evaluation.", flush=True)
