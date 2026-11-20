import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import warnings

# CPU execution, one thread and deterministic kernels reduce numerical variation.
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


tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
output_dir = Path("artifacts/deep_learning")
output_dir.mkdir(parents=True, exist_ok=True)

# A per-observation metric is necessary to express log-target curves in USD.
@keras.utils.register_keras_serializable(package="CaratMind")
def log_price_absolute_error(y_true, y_pred):
    return tf.abs(tf.math.expm1(y_true) - tf.math.expm1(y_pred))


preparation = json.loads(Path("artifacts/preparation/preparation_report.json").read_text())
ml_metadata = json.loads(Path("artifacts/tuning/experiment_metadata.json").read_text())
preprocessor = joblib.load("artifacts/preparation/preprocessor.joblib")
input_columns = preprocessor.feature_names_in_.tolist()
feature_names = preprocessor.get_feature_names_out().tolist()
assert input_columns == ml_metadata["input_columns"]
assert not {"price", "source_row", "Unnamed: 0"}.intersection(input_columns)
assert feature_names == preparation["transformed_features"] and len(feature_names) == 26
assert all(name.split("__", 1)[-1] not in {"price", "source_row", "Unnamed: 0"} for name in feature_names)

# Hash existing artifacts, including test files, without loading test data.
protected_paths = [Path("data/raw/10-diamonds (1).csv")]
protected_paths += sorted(Path("src").rglob("*.py"))
for folder in ["data/processed", "artifacts/preparation", "artifacts/baselines", "artifacts/tuning", "artifacts/figures"]:
    protected_paths += sorted(path for path in Path(folder).rglob("*") if path.is_file())
protected_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected_paths}
for name, expected in ml_metadata["protected_sha256"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
assert protected_hashes[str(protected_paths[0])] == preparation["raw_sha256"]

splits = {}
frames = {}
for name in ["train", "validation"]:
    frame = pd.read_csv(f"data/processed/{name}.csv")
    assert set(frame.columns) == {"source_row", "price", *input_columns}
    assert not frame.isna().any().any()
    with np.load(f"data/processed/{name}_transformed.npz", allow_pickle=False) as saved:
        features = saved["X"].copy()
        target = saved["y"].copy()
        np.testing.assert_array_equal(saved["source_row"], frame["source_row"])
        np.testing.assert_array_equal(target, frame["price"])
        assert saved["feature_names"].tolist() == feature_names
    assert features.shape == tuple(preparation["splits"][name]["transformed_shape"])
    assert target.shape == (len(frame),)
    assert np.isfinite(features).all() and np.isfinite(target).all() and (target > 0).all()
    np.testing.assert_allclose(features, preprocessor.transform(frame[input_columns]), rtol=1e-12, atol=1e-12)
    # Keras uses float32; this changes numerical precision, not the features or scaling.
    splits[name] = {"X": features.astype("float32"), "y": target.astype("float64")}
    frames[name] = frame
    print(f"Verified {name}: X={features.shape}, y={target.shape}", flush=True)
assert set(frames["train"]["source_row"]).isdisjoint(frames["validation"]["source_row"])
assert set(frames["train"][input_columns].itertuples(index=False, name=None)).isdisjoint(
    set(frames["validation"][input_columns].itertuples(index=False, name=None)))
X_train, y_train = splits["train"]["X"], splits["train"]["y"]
X_validation, y_validation = splits["validation"]["X"], splits["validation"]["y"]
scaler = preprocessor.named_steps["columns"].named_transformers_["numeric"]
assert scaler.n_samples_seen_ == len(y_train)
np.testing.assert_allclose(scaler.mean_, frames["train"][preparation["numeric_columns"]].mean())

# Validate the saved ML metrics against their row-aligned validation predictions.
ml_results = pd.read_csv("artifacts/tuning/validation_comparison.csv")
ml_results = ml_results.loc[ml_results["experiment"] == "tuned"].copy()
ml_errors = pd.read_csv("artifacts/tuning/validation_errors.csv")
for name in ml_results["model"]:
    error_rows = ml_errors.loc[ml_errors["model"] == name]
    np.testing.assert_array_equal(error_rows["source_row"], frames["validation"]["source_row"])
    np.testing.assert_array_equal(error_rows["actual_price"], y_validation)
    row = ml_results.loc[ml_results["model"] == name].iloc[0]
    np.testing.assert_allclose(mean_absolute_error(y_validation, error_rows["predicted_price"]), row["validation_mae_usd"])
    np.testing.assert_allclose(root_mean_squared_error(y_validation, error_rows["predicted_price"]), row["validation_rmse_usd"])
    np.testing.assert_allclose(r2_score(y_validation, error_rows["predicted_price"]), row["validation_r2"])

experiments = {
    "baseline_raw": {"units": [64, 32], "dropout": 0.0, "target": "price"},
    "deep_raw": {"units": [128, 64, 32], "dropout": 0.0, "target": "price"},
    "deep_dropout_raw": {"units": [128, 64, 32], "dropout": 0.1, "target": "price"},
    "baseline_log": {"units": [64, 32], "dropout": 0.0, "target": "log1p(price)"},
}
configuration = {
    "experiments": experiments, "seed": 42, "optimizer": "Adam", "learning_rate": 0.001,
    "loss": "MSE on the experiment target scale", "batch_size": 64, "maximum_epochs": 150,
    "early_stopping": {"monitor": "val_mae_usd", "mode": "min", "patience": 15,
                       "min_delta": 0, "restore_best_weights": True},
    "regularization_reason": "Controlled mild Dropout comparison tests whether deep-network co-adaptation contributes to a training/validation gap; it is not assumed beneficial.",
    "target_loss_reason": "Raw-price MSE emphasizes large dollar errors. Log1p MSE emphasizes proportional differences; expm1 does not apply a lognormal bias correction.",
    "curve_note": "MAE curves are measured during each epoch, with Dropout active for training only. Final metrics use restored weights in inference mode. Log curves explicitly invert both targets and outputs in float32.",
    "reload_tolerance": {"rtol": 1e-6, "atol_usd": 0.01},
    "reproducibility": "Keras/Python/NumPy seeds=42; deterministic TensorFlow CPU kernels; one inter/intra-op thread; seeded dataset shuffling. Exact results across different hardware/package versions are not guaranteed; full independent retraining variability is not measured.",
    "test_policy": "Test files hashed only; no test arrays loaded, predictions, evaluation or selection.",
    "feature_names": feature_names, "input_columns": input_columns,
    "protected_sha256": protected_hashes, "python": platform.python_version(),
    "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    "tensorflow_devices": [device.name for device in tf.config.list_physical_devices()],
    "ml_versions": ml_metadata["versions"],
    "limitations": "Four fixed experiments, one initialization seed and repeatedly used validation partition. Validation is used for early stopping and model selection, not independent generalization assessment.",
}
(output_dir / "experiment_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
results = []
error_frames = []
warning_messages = []
experiment_started = perf_counter()
for name, settings in experiments.items():
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    model = keras.Sequential([keras.Input(shape=(26,))], name=name)
    for position, units in enumerate(settings["units"]):
        model.add(keras.layers.Dense(units, activation="relu"))
        # Regularize the first two hidden layers; keep the final representation intact.
        if settings["dropout"] and position < len(settings["units"]) - 1:
            model.add(keras.layers.Dropout(settings["dropout"]))
    model.add(keras.layers.Dense(1, activation="linear"))
    log_target = settings["target"] == "log1p(price)"
    metric = (keras.metrics.MeanMetricWrapper(log_price_absolute_error, name="mae_usd")
              if log_target else keras.metrics.MeanAbsoluteError(name="mae_usd"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=[metric])
    train_target = np.log1p(y_train) if log_target else y_train
    validation_target = np.log1p(y_validation) if log_target else y_validation
    train_target = train_target.astype("float32").reshape(-1, 1)
    validation_target = validation_target.astype("float32").reshape(-1, 1)
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    train_data = tf.data.Dataset.from_tensor_slices((X_train, train_target))
    train_data = train_data.shuffle(len(X_train), seed=42, reshuffle_each_iteration=True)
    train_data = train_data.batch(64).with_options(options).prefetch(1)
    validation_data = tf.data.Dataset.from_tensor_slices((X_validation, validation_target))
    validation_data = validation_data.batch(64).with_options(options).prefetch(1)
    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    (output_dir / f"{name}_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    parameters = int(sum(np.prod(weight.shape) for weight in model.trainable_weights))
    print(f"\nTraining {name}: {settings}; {parameters:,} trainable parameters", flush=True)
    started = perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            history = model.fit(train_data, validation_data=validation_data, epochs=150, verbose=0, shuffle=False,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_mae_usd", mode="min", patience=15,
                                                   restore_best_weights=True),
                    keras.callbacks.CSVLogger(str(output_dir / f"{name}_history.csv")),
                    keras.callbacks.TerminateOnNaN(),
                ])
        duration = perf_counter() - started
        for warning in caught:
            warning_messages.append(f"{name}: {warning.category.__name__}: {warning.message}")
        history_frame = pd.read_csv(output_dir / f"{name}_history.csv")
        assert np.isfinite(history_frame.to_numpy()).all()
        assert len(history_frame) == len(history.history["loss"])
        best_index = int(np.argmin(history.history["val_mae_usd"]))
        train_output = model(X_train, training=False).numpy().ravel().astype("float64")
        validation_output = model(X_validation, training=False).numpy().ravel().astype("float64")
        train_prediction = np.expm1(train_output) if log_target else train_output
        prediction = np.expm1(validation_output) if log_target else validation_output
        assert np.isfinite(train_prediction).all() and np.isfinite(prediction).all()
        model_path = output_dir / f"{name}.keras"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.save(model_path)
            # Inference loading needs no custom metric or compilation.
            restored = keras.models.load_model(model_path, compile=False)
        for warning in caught:
            warning_messages.append(f"{name}: {warning.category.__name__}: {warning.message}")
        restored_output = restored(X_validation, training=False).numpy().ravel().astype("float64")
        restored_prediction = np.expm1(restored_output) if log_target else restored_output
        np.testing.assert_allclose(prediction, restored_prediction, rtol=1e-6, atol=0.01)
        row = {"model": name, "target": settings["target"], "hidden_layers": str(settings["units"]),
               "dropout": settings["dropout"], "trainable_parameters": parameters,
               "optimizer": "Adam", "learning_rate": 0.001, "batch_size": 64,
               "epochs_executed": len(history_frame), "best_validation_epoch": best_index + 1,
               "lowest_validation_loss_epoch": int(np.argmin(history.history["val_loss"])) + 1,
               "training_seconds": duration, "model_bytes": model_path.stat().st_size,
               "train_loss_at_best_epoch": history.history["loss"][best_index],
               "validation_loss_at_best_epoch": history.history["val_loss"][best_index],
               "final_epoch_train_loss": history.history["loss"][-1],
               "final_epoch_validation_loss": history.history["val_loss"][-1]}
        for partition, actual, predicted in [("train", y_train, train_prediction),
                                             ("validation", y_validation, prediction)]:
            row[f"{partition}_mae_usd"] = mean_absolute_error(actual, predicted)
            row[f"{partition}_rmse_usd"] = root_mean_squared_error(actual, predicted)
            row[f"{partition}_r2"] = r2_score(actual, predicted)
        np.testing.assert_allclose(row["validation_mae_usd"], history.history["val_mae_usd"][best_index],
                                   rtol=1e-5, atol=0.02)
        row["mae_gap_usd"] = row["validation_mae_usd"] - row["train_mae_usd"]
        row["rmse_gap_usd"] = row["validation_rmse_usd"] - row["train_rmse_usd"]
        row["r2_drop"] = row["train_r2"] - row["validation_r2"]
        row["validation_negative_predictions"] = int((prediction < 0).sum())
        results.append(row)
        pd.DataFrame(results).sort_values("validation_mae_usd").to_csv(output_dir / "experiment_comparison.csv", index=False)
    except Exception as error:
        failure = {"experiment": name, "error": repr(error), "completed_experiments": [row["model"] for row in results]}
        (output_dir / "failed_experiment.json").write_text(json.dumps(failure, indent=2))
        raise

    errors = pd.DataFrame({"model": name, "source_row": frames["validation"]["source_row"],
                           "actual_price": y_validation, "predicted_price": prediction})
    errors["residual_usd"] = errors["actual_price"] - errors["predicted_price"]
    errors["absolute_error_usd"] = errors["residual_usd"].abs()
    errors["squared_error"] = errors["residual_usd"] ** 2
    errors["price_range"] = pd.cut(errors["actual_price"], bins=[0, 1000, 3000, 5000, 10000, np.inf],
        labels=["0-999", "1000-2999", "3000-4999", "5000-9999", "10000+"], right=False)
    error_frames.append(errors)
    pd.concat(error_frames, ignore_index=True).drop(columns="squared_error").to_csv(
        output_dir / "validation_predictions.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = history_frame["epoch"] + 1
    axes[0, 0].plot(epochs, history_frame["loss"], label="Training")
    axes[0, 0].plot(epochs, history_frame["val_loss"], label="Validation")
    axes[0, 0].set_yscale("log")
    loss_label = "log1p(price) MSE" if log_target else "Price MSE (USD squared)"
    axes[0, 0].set(title="Loss (logarithmic axis)", xlabel="Epoch", ylabel=loss_label)
    axes[0, 1].plot(epochs, history_frame["mae_usd"], label="Training")
    axes[0, 1].plot(epochs, history_frame["val_mae_usd"], label="Validation")
    axes[0, 1].set(title="MAE in Original USD", xlabel="Epoch", ylabel="MAE (USD)")
    for ax in axes[0]:
        ax.axvline(best_index + 1, color="gray", linestyle="--", label="Restored epoch")
        ax.legend()
    if log_target:
        # The first log-target epoch has very large USD errors; retain its full range.
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_title("MAE in Original USD (logarithmic axis)")
    axes[1, 0].scatter(y_validation, prediction, s=10, alpha=0.25)
    limits = [min(y_validation.min(), prediction.min()), max(y_validation.max(), prediction.max())]
    axes[1, 0].plot(limits, limits, "--", color="darkred")
    axes[1, 0].set(title="Validation Predictions", xlabel="Actual Price (USD)", ylabel="Predicted Price (USD)")
    axes[1, 1].hist(errors["residual_usd"], bins=70, color="steelblue", edgecolor="white")
    axes[1, 1].axvline(0, color="darkred", linestyle="--")
    axes[1, 1].set(title="Validation Residuals", xlabel="Actual - Predicted Price (USD)", ylabel="Number of Diamonds")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_diagnostics.png", dpi=200)
    plt.close(fig)
    print(f"Completed {name}: epochs={len(history_frame)}, best={best_index + 1}, "
          f"seconds={duration:.1f}, validation MAE=${row['validation_mae_usd']:.2f}", flush=True)

comparison = pd.DataFrame(results).sort_values("validation_mae_usd").reset_index(drop=True)
errors = pd.concat(error_frames, ignore_index=True)
grouped = errors.groupby(["model", "price_range"], observed=True)
error_summary = grouped.agg(count=("actual_price", "size"), mae_usd=("absolute_error_usd", "mean"),
    median_absolute_error_usd=("absolute_error_usd", "median"), mean_residual_usd=("residual_usd", "mean"),
    mean_squared_error=("squared_error", "mean"))
error_summary["rmse_usd"] = np.sqrt(error_summary.pop("mean_squared_error"))
error_summary["p90_absolute_error_usd"] = grouped["absolute_error_usd"].quantile(0.9)
error_summary.reset_index().to_csv(output_dir / "validation_errors_by_price.csv", index=False)
fig, ax = plt.subplots(figsize=(11, 6))
error_summary["mae_usd"].unstack("model").plot.bar(ax=ax, rot=0)
ax.set(title="ANN Validation MAE by Actual Price", xlabel="Actual Price Range (USD)", ylabel="MAE (USD)")
ax.legend(title="Experiment")
fig.tight_layout()
fig.savefig(output_dir / "validation_mae_by_price.png", dpi=200)
plt.close(fig)

best_ann = comparison.iloc[0]
comparison_columns = ["model", "train_mae_usd", "train_rmse_usd", "train_r2", "validation_mae_usd",
                      "validation_rmse_usd", "validation_r2", "mae_gap_usd"]
ml_dl = pd.concat([ml_results[comparison_columns], comparison.loc[:0, comparison_columns]], ignore_index=True)
for index, name in enumerate(ml_dl["model"]):
    if name in experiments:
        ml_dl.loc[index, "high_price_mae_usd"] = error_summary.loc[(name, "10000+"), "mae_usd"]
        ml_dl.loc[index, "complexity"] = f"{int(best_ann['trainable_parameters'])} trainable parameters"
        ml_dl.loc[index, "model_bytes"] = best_ann["model_bytes"]
    else:
        high = ml_errors.loc[(ml_errors["model"] == name) & (ml_errors["actual_price"] >= 10000)]
        ml_dl.loc[index, "high_price_mae_usd"] = (high["actual_price"] - high["predicted_price"]).abs().mean()
        ml_dl.loc[index, "complexity"] = ("100 trees, unrestricted depth" if name == "RandomForestRegressor"
                                         else "250 boosting iterations, at most 63 leaves per tree")
        ml_dl.loc[index, "model_bytes"] = Path(f"artifacts/tuning/{name}_pipeline.joblib").stat().st_size
ml_dl.sort_values("validation_mae_usd").to_csv(output_dir / "ml_dl_comparison.csv", index=False)
for path in protected_paths:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == protected_hashes[str(path)]
verification = {
    "best_ann": best_ann["model"], "completed_experiments": comparison["model"].tolist(),
    "warnings": warning_messages, "failed_experiments": [],
    "total_training_and_diagnostics_seconds": perf_counter() - experiment_started,
    "checks": "Passed: 26 prepared features, finite data, positive targets, CSV/NPZ/pipeline alignment, no forbidden inputs, no train/validation row or feature-group overlap, train-only scaler, ML metrics reproduced from aligned saved errors, histories finite, restored epoch metrics match, all Keras reload USD predictions match at rtol=1e-6/atol=$0.01, all protected hashes unchanged.",
    "test_evaluation": False,
}
(output_dir / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
print("\n" + comparison.round(4).to_string(index=False), flush=True)
print("\nML vs strongest ANN:\n" + ml_dl.round(4).to_string(index=False), flush=True)
print("\nValidation errors by price:\n" + error_summary.round(2).to_string(), flush=True)
print(f"\nWarnings captured: {len(warning_messages)}. All checks passed. Phase 4A complete; no test evaluation.", flush=True)
for message in sorted(set(warning_messages)):
    print(message, flush=True)
