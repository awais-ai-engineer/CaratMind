import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
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


tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
folder = Path("artifacts/deep_learning/final")
selection = json.loads((folder / "candidate_selection.json").read_text())
configuration = json.loads((folder / "experiment_config.json").read_text())
for path, expected in configuration["protected_sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
preprocessor = joblib.load("artifacts/preparation/preprocessor.joblib")
input_columns = preprocessor.feature_names_in_.tolist()
assert input_columns == configuration["input_columns"]
assert preprocessor.get_feature_names_out().tolist() == configuration["feature_names"]
assert not {"price", "source_row", "Unnamed: 0"}.intersection(input_columns)
frames = {}
arrays = {}
for name in ["train", "validation"]:
    frame = pd.read_csv(f"data/processed/{name}.csv")
    assert not frame.isna().any().any()
    with np.load(f"data/processed/{name}_transformed.npz", allow_pickle=False) as saved:
        X, y = saved["X"].copy(), saved["y"].astype("float64")
        np.testing.assert_array_equal(saved["source_row"], frame["source_row"])
        np.testing.assert_array_equal(y, frame["price"])
        assert saved["feature_names"].tolist() == configuration["feature_names"]
    assert X.shape == (len(y), 26) and np.isfinite(X).all() and np.isfinite(y).all()
    np.testing.assert_allclose(X, preprocessor.transform(frame[input_columns]), rtol=1e-12, atol=1e-12)
    frames[name], arrays[name] = frame, {"X": X.astype("float32"), "y": y}
assert set(frames["train"]["source_row"]).isdisjoint(frames["validation"]["source_row"])
assert set(frames["train"][input_columns].itertuples(index=False, name=None)).isdisjoint(
    set(frames["validation"][input_columns].itertuples(index=False, name=None)))
X_train, y_train = arrays["train"]["X"], arrays["train"]["y"]
X_validation, y_validation = arrays["validation"]["X"], arrays["validation"]["y"]

# If refinement was rejected, verify the retained configuration rather than
# presenting the rejected protocol's seed variability as the final ANN's stability.
phase4a = pd.read_csv("artifacts/deep_learning/experiment_comparison.csv")
reference = phase4a.loc[phase4a["model"] == "deep_raw"].iloc[0]
new_seeds = [] if selection["refinement_accepted"] else [7, 2026]
records = []
prediction_frames = []
warning_messages = []
robustness_config = {
    "reason": "Complete three-seed verification of the selected ANN, including the retained Phase 4A configuration when refinement is rejected.",
    "seeds": [42, 7, 2026], "representative_seed": 42,
    "refinement_accepted": selection["refinement_accepted"],
    "retained_protocol": {"architecture": [26, 128, 64, 32, 1], "optimizer": "Adam", "learning_rate": 0.001,
        "loss": "original-price MSE", "batch_size": 64, "maximum_epochs": 150,
        "early_stopping_patience": 15, "monitor": "val_mae_usd", "restore_best_weights": True,
        "learning_rate_schedule": None},
    "reuse_policy": "Reuse verified Phase 4A seed 42; train only missing seeds 7 and 2026. If refinement is accepted, reuse all three already verified refined runs.",
    "selection_policy": configuration["ann_rule"],
    "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
}
(folder / "selected_ann_robustness_config.json").write_text(json.dumps(robustness_config, indent=2), encoding="utf-8")
for seed in [42, 7, 2026]:
    started = perf_counter()
    if seed in new_seeds:
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        model = keras.Sequential([keras.Input(shape=(26,)), keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"), keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="linear")], name=f"retained_ann_seed_{seed}")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse",
            metrics=[keras.metrics.MeanAbsoluteError(name="mae_usd")])
        options = tf.data.Options()
        options.deterministic = True
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train.astype("float32").reshape(-1, 1)))
        train_data = train_data.shuffle(len(X_train), seed=seed, reshuffle_each_iteration=True).batch(64).with_options(options).prefetch(1)
        val_data = tf.data.Dataset.from_tensor_slices((X_validation, y_validation.astype("float32").reshape(-1, 1)))
        val_data = val_data.batch(64).with_options(options).prefetch(1)
        history_path = folder / f"retained_seed_{seed}_history.csv"
        model_path = folder / f"retained_seed_{seed}.keras"
        print(f"Training retained ANN configuration, seed {seed}; no new tuning", flush=True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(train_data, validation_data=val_data, epochs=150, shuffle=False, verbose=0,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="val_mae_usd", mode="min", patience=15,
                        restore_best_weights=True), keras.callbacks.CSVLogger(str(history_path)),
                        keras.callbacks.TerminateOnNaN()])
                duration = perf_counter() - started
                model.save(model_path)
            warning_messages.extend(f"retained seed {seed}: {item.category.__name__}: {item.message}" for item in caught)
        except Exception as error:
            (folder / "robustness_failure.json").write_text(json.dumps({"seed": seed, "error": repr(error)}, indent=2))
            raise
    else:
        if selection["refinement_accepted"]:
            model_path, history_path = folder / f"seed_{seed}.keras", folder / f"seed_{seed}_history.csv"
            duration = float(pd.read_csv(folder / "seed_metrics.csv").set_index("seed").loc[seed, "training_seconds"])
        else:
            model_path = Path("artifacts/deep_learning/deep_raw.keras")
            history_path = Path("artifacts/deep_learning/deep_raw_history.csv")
            duration = float(reference["training_seconds"])
        model = keras.models.load_model(model_path, compile=False)
    history = pd.read_csv(history_path)
    assert model.count_params() == 13825 and np.isfinite(history.to_numpy()).all()
    prediction = model(X_validation, training=False).numpy().ravel().astype("float64")
    train_prediction = model(X_train, training=False).numpy().ravel().astype("float64")
    assert np.isfinite(prediction).all() and np.isfinite(train_prediction).all()
    restored = keras.models.load_model(model_path, compile=False)
    np.testing.assert_allclose(prediction, restored(preprocessor.transform(frames["validation"][input_columns]).astype("float32"), training=False).numpy().ravel(), rtol=1e-6, atol=0.01)
    row = {"seed": seed, "model_path": str(model_path), "history_path": str(history_path),
        "epochs": len(history), "restored_epoch": int(history["val_mae_usd"].idxmin()) + 1,
        "training_seconds": duration}
    for partition, actual, predicted in [("train", y_train, train_prediction), ("validation", y_validation, prediction)]:
        row[f"{partition}_mae_usd"] = mean_absolute_error(actual, predicted)
        row[f"{partition}_rmse_usd"] = root_mean_squared_error(actual, predicted)
        row[f"{partition}_r2"] = r2_score(actual, predicted)
    np.testing.assert_allclose(row["validation_mae_usd"], history["val_mae_usd"].min(), rtol=1e-5, atol=0.02)
    row["mae_gap_usd"] = row["validation_mae_usd"] - row["train_mae_usd"]
    records.append(row)
    pd.DataFrame(records).to_csv(folder / "selected_ann_seed_metrics.csv", index=False)
    prediction_frames.append(pd.DataFrame({"seed": seed, "source_row": frames["validation"]["source_row"],
        "actual_price": y_validation, "predicted_price": prediction}))
    pd.concat(prediction_frames, ignore_index=True).to_csv(folder / "selected_ann_predictions.csv", index=False)
    print(f"Selected configuration seed {seed}: validation MAE=${row['validation_mae_usd']:.2f}", flush=True)

metrics = pd.DataFrame(records)
columns = ["train_mae_usd", "train_rmse_usd", "train_r2", "validation_mae_usd", "validation_rmse_usd", "validation_r2", "mae_gap_usd"]
stability = metrics[columns].agg(["mean", "std"]).T
stability.index.name = "metric"
stability.to_csv(folder / "selected_ann_stability.csv")
errors = pd.concat(prediction_frames, ignore_index=True)
errors["absolute_error_usd"] = (errors["actual_price"] - errors["predicted_price"]).abs()
errors["price_range"] = pd.cut(errors["actual_price"], [0, 1000, 3000, 5000, 10000, np.inf],
    labels=["0-999", "1000-2999", "3000-4999", "5000-9999", "10000+"], right=False)
by_range = errors.groupby(["seed", "price_range"], observed=True).agg(count=("actual_price", "size"), mae_usd=("absolute_error_usd", "mean"))
by_range.to_csv(folder / "selected_ann_price_range_mae.csv")
by_range.reset_index().groupby("price_range", observed=True)["mae_usd"].agg(["mean", "std"]).to_csv(folder / "selected_ann_price_range_stability.csv")
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
for index, row in enumerate(records):
    history = pd.read_csv(row["history_path"])
    for column, fields, ylabel in [(0, ["loss", "val_loss"], "MSE (USD squared)"), (1, ["mae_usd", "val_mae_usd"], "MAE (USD)")]:
        ax = axes[index, column]
        ax.plot(history["epoch"] + 1, history[fields[0]], label="Training")
        ax.plot(history["epoch"] + 1, history[fields[1]], label="Validation")
        ax.axvline(row["restored_epoch"], linestyle="--", color="gray", label="Restored epoch")
        ax.set(yscale="log", title=f"Selected ANN: seed {row['seed']}", xlabel="Epoch", ylabel=ylabel)
        ax.legend()
fig.tight_layout()
fig.savefig(folder / "selected_ann_training_stability.png", dpi=180)
plt.close(fig)
for path, expected in configuration["protected_sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
selection["selected_ann_three_seed_stability"] = stability.to_dict(orient="index")
selection["selected_ann_stability_policy"] = "Three predefined seeds; sample SD (ddof=1); same selected configuration for all seeds; seed 42 remains the representative even if another seed scores lower. No dataset-level significance claim."
selection["selected_ann_robustness_complete"] = True
(folder / "candidate_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
verification = {"checks": "Selected ANN configuration verified across seeds 42, 7 and 2026. Feature/target alignment, no group overlap, finite histories/predictions, restored epoch metrics and reload through existing preprocessing passed. Protected files unchanged. No test evaluation.",
    "warning_messages": warning_messages, "seed_metrics": records,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(folder / "robustness_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
print("\nSelected ANN stability (sample SD):\n" + stability.round(6).to_string(), flush=True)
print("Selected ANN three-seed robustness complete. Representative remains seed 42; no test evaluation.", flush=True)
