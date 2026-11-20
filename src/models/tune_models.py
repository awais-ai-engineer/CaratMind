import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


output_dir = Path("artifacts/tuning")
output_dir.mkdir(parents=True, exist_ok=True)
preparation = json.loads(Path("artifacts/preparation/preparation_report.json").read_text())
baseline_config = json.loads(Path("artifacts/baselines/experiment_config.json").read_text())
baseline_results = pd.read_csv("artifacts/baselines/model_comparison.csv")
preprocessor = joblib.load("artifacts/preparation/preprocessor.joblib")
input_columns = preprocessor.feature_names_in_.tolist()
assert input_columns == baseline_config["input_columns"]
assert not {"price", "Unnamed: 0", "source_row"}.intersection(input_columns)
assert preprocessor.get_feature_names_out().tolist() == preparation["transformed_features"]

# Protect all existing inputs, including test files and baseline artifacts, by hash.
protected_paths = [Path("data/raw/10-diamonds (1).csv"),
                   Path("src/data/prepare_data.py"), Path("src/models/train_baselines.py")]
for folder in ["data/processed", "artifacts/preparation", "artifacts/baselines"]:
    protected_paths += sorted(path for path in Path(folder).rglob("*") if path.is_file())
protected_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected_paths}
for name, expected in baseline_config["input_sha256"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
assert protected_hashes[str(protected_paths[0])] == preparation["raw_sha256"]

frames = {}
for name in ["train", "validation"]:
    frame = pd.read_csv(f"data/processed/{name}.csv")
    assert set(frame.columns) == {"source_row", "price", *input_columns}
    assert not frame.isna().any().any()
    assert np.isfinite(frame.select_dtypes(include="number")).all().all()
    with np.load(f"data/processed/{name}_transformed.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_row"], frame["source_row"])
        np.testing.assert_array_equal(saved["y"], frame["price"])
        assert saved["feature_names"].tolist() == preparation["transformed_features"]
        assert saved["X"].shape == tuple(preparation["splits"][name]["transformed_shape"])
        np.testing.assert_allclose(saved["X"], preprocessor.transform(frame[input_columns]),
                                   rtol=1e-12, atol=1e-12)
    frames[name] = frame
    print(f"{name}: {len(frame):,} rows; prepared arrays and feature order verified", flush=True)
X_train = frames["train"][input_columns]
y_train = frames["train"]["price"]
X_validation = frames["validation"][input_columns]
y_validation = frames["validation"]["price"]
assert set(frames["train"]["source_row"]).isdisjoint(frames["validation"]["source_row"])
assert set(X_train.itertuples(index=False, name=None)).isdisjoint(
    set(X_validation.itertuples(index=False, name=None)))

# Reconstruct the Phase 2 grouping policy from inputs only, never from price.
groups = X_train.groupby(input_columns, sort=False).ngroup()
assert groups.nunique() == preparation["splits"]["train"]["groups"]
cv = GroupKFold(n_splits=3, shuffle=True, random_state=42)
folds = list(cv.split(X_train, y_train, groups))
repeated_folds = list(cv.split(X_train, y_train, groups))
fold_assignment = np.full(len(X_train), -1)
fold_sizes = []
for fold, ((fit_rows, score_rows), (fit_again, score_again)) in enumerate(zip(folds, repeated_folds)):
    assert set(groups.iloc[fit_rows]).isdisjoint(groups.iloc[score_rows])
    assert set(fit_rows).isdisjoint(score_rows)
    assert len(fit_rows) + len(score_rows) == len(X_train)
    np.testing.assert_array_equal(fit_rows, fit_again)
    np.testing.assert_array_equal(score_rows, score_again)
    assert (fold_assignment[score_rows] == -1).all()
    fold_assignment[score_rows] = fold
    fold_sizes.append({"fold": fold, "fit_rows": len(fit_rows), "score_rows": len(score_rows)})
assert (fold_assignment >= 0).all()
pd.DataFrame({"source_row": frames["train"]["source_row"], "group": groups,
              "score_fold": fold_assignment}).to_csv(output_dir / "cv_fold_assignments.csv", index=False)

# Five focused candidates per family, not a large Cartesian grid.
# Keep 100 forest trees for direct comparability; vary capacity and feature sampling.
forest_candidates = [
    {"max_depth": None, "min_samples_leaf": 2, "min_samples_split": 2, "max_features": 1.0},
    {"max_depth": None, "min_samples_leaf": 1, "min_samples_split": 2, "max_features": 0.8},
    {"max_depth": 20, "min_samples_leaf": 2, "min_samples_split": 5, "max_features": 0.8},
    {"max_depth": 15, "min_samples_leaf": 4, "min_samples_split": 10, "max_features": 1.0},
    {"max_depth": None, "min_samples_leaf": 2, "min_samples_split": 5, "max_features": 0.6},
]
# Longer boosting runs test whether the 100-iteration baseline was too limited.
boosting_candidates = [
    {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31, "l2_regularization": 0.0},
    {"learning_rate": 0.1, "max_iter": 250, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    {"learning_rate": 0.1, "max_iter": 250, "max_leaf_nodes": 63, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 63, "l2_regularization": 5.0},
]
models = {
    "RandomForestRegressor": RandomForestRegressor(**baseline_config["configurations"]["RandomForestRegressor"]),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
        **baseline_config["configurations"]["HistGradientBoostingRegressor"]),
}
candidates = {"RandomForestRegressor": forest_candidates,
              "HistGradientBoostingRegressor": boosting_candidates}
selected = {}
cv_tables = []
best_configurations = {}
cv_summaries = {}
started = perf_counter()
for name, model in models.items():
    pipeline = Pipeline([("preprocessing", clone(preprocessor)), ("model", model)])
    assert not hasattr(pipeline.named_steps["preprocessing"], "feature_names_in_")
    parameter_grid = [{f"model__{key}": [value] for key, value in candidate.items()}
                      for candidate in candidates[name]]
    search = GridSearchCV(pipeline, parameter_grid, scoring="neg_mean_absolute_error",
                          cv=folds, refit=True, n_jobs=1, error_score="raise",
                          return_train_score=True, verbose=2)
    print(f"\n{name}: 5 candidates x 3 group-aware folds", flush=True)
    with threadpool_limits(limits=1):
        # Only raw training features and training targets enter any search or fit.
        search.fit(X_train, y_train)
        selected[name] = search.best_estimator_
        # Repeat the selected configuration on every fold to verify CV scores and
        # inspect the fitted fold preprocessors, rather than reusing the full-train one.
        repeated = cross_validate(clone(search.best_estimator_), X_train, y_train,
                                  cv=folds, scoring="neg_mean_absolute_error",
                                  return_estimator=True, n_jobs=1, error_score="raise")
    best_row = search.best_index_
    expected_scores = np.array([search.cv_results_[f"split{fold}_test_score"][best_row]
                                for fold in range(len(folds))])
    np.testing.assert_allclose(repeated["test_score"], expected_scores, rtol=1e-12, atol=1e-9)
    for estimator, (fit_rows, score_rows) in zip(repeated["estimator"], folds):
        fold_columns = estimator.named_steps["preprocessing"].named_steps["columns"]
        fold_scaler = fold_columns.named_transformers_["numeric"]
        assert fold_scaler.n_samples_seen_ == len(fit_rows)
        np.testing.assert_allclose(fold_scaler.mean_,
            X_train.iloc[fit_rows][preparation["numeric_columns"]].mean())
        encoder = fold_columns.named_transformers_["categorical"]
        for column, categories in zip(preparation["categorical_columns"], encoder.categories_):
            assert set(categories) == set(X_train.iloc[fit_rows][column])
    del repeated
    full_scaler = selected[name].named_steps["preprocessing"].named_steps["columns"].named_transformers_["numeric"]
    assert full_scaler.n_samples_seen_ == len(X_train)
    np.testing.assert_allclose(full_scaler.mean_, X_train[preparation["numeric_columns"]].mean())
    best_configurations[name] = selected[name].named_steps["model"].get_params()
    cv_summaries[name] = {"mean_mae_usd": float(-search.best_score_),
                          "std_mae_usd": float(search.cv_results_["std_test_score"][best_row]),
                          "fold_mae_usd": (-expected_scores).tolist()}
    table = pd.DataFrame(search.cv_results_)
    table.insert(0, "model", name)
    table["cv_mae_usd"] = -table["mean_test_score"]
    table["cv_train_mae_usd"] = -table["mean_train_score"]
    cv_tables.append(table)
    # Save progress before starting the next family.
    pd.concat(cv_tables, ignore_index=True).to_csv(output_dir / "tuning_results.csv", index=False)
    joblib.dump(selected[name], output_dir / f"{name}_pipeline.joblib", compress=3)
    print(f"Selected {name}: CV MAE={-search.best_score_:.2f}; {search.best_params_}", flush=True)

# Both searches are complete before validation is used to compare candidates.
comparison_rows = []
error_frames = []
for name, pipeline in selected.items():
    with threadpool_limits(limits=1):
        train_prediction = pipeline.predict(X_train)
        prediction = pipeline.predict(X_validation)
        restored = joblib.load(output_dir / f"{name}_pipeline.joblib")
        np.testing.assert_allclose(prediction, restored.predict(X_validation), rtol=1e-12, atol=1e-9)
        repeated = clone(pipeline)
        repeated.fit(X_train, y_train)
        np.testing.assert_allclose(train_prediction, repeated.predict(X_train), rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(prediction, repeated.predict(X_validation), rtol=1e-12, atol=1e-9)
        del repeated
    assert np.isfinite(prediction).all() and np.isfinite(train_prediction).all()
    row = {"model": name, "experiment": "tuned"}
    for partition, actual, predicted in [("train", y_train, train_prediction),
                                          ("validation", y_validation, prediction)]:
        row[f"{partition}_mae_usd"] = mean_absolute_error(actual, predicted)
        row[f"{partition}_rmse_usd"] = root_mean_squared_error(actual, predicted)
        row[f"{partition}_r2"] = r2_score(actual, predicted)
    row["mae_gap_usd"] = row["validation_mae_usd"] - row["train_mae_usd"]
    row["rmse_gap_usd"] = row["validation_rmse_usd"] - row["train_rmse_usd"]
    row["r2_drop"] = row["train_r2"] - row["validation_r2"]
    comparison_rows.append(row)
    baseline_row = baseline_results.loc[baseline_results["model"] == name].iloc[0].to_dict()
    baseline_row.pop("seconds_including_repeat_check")
    baseline_row["experiment"] = "baseline"
    comparison_rows.append(baseline_row)
    errors = pd.DataFrame({"model": name, "source_row": frames["validation"]["source_row"],
                           "actual_price": y_validation, "predicted_price": prediction})
    errors["residual_usd"] = errors["actual_price"] - errors["predicted_price"]
    errors["absolute_error_usd"] = errors["residual_usd"].abs()
    errors["squared_error"] = errors["residual_usd"] ** 2
    errors["absolute_percentage_error"] = 100 * errors["absolute_error_usd"] / errors["actual_price"]
    # Fixed interpretable USD bins are for diagnostics only, not feature engineering.
    errors["price_range"] = pd.cut(errors["actual_price"], bins=[0, 1000, 3000, 5000, 10000, np.inf],
                                    labels=["0-999", "1000-2999", "3000-4999", "5000-9999", "10000+"],
                                    right=False)
    assert errors["price_range"].notna().all()
    error_frames.append(errors)

comparison = pd.DataFrame(comparison_rows).sort_values("validation_mae_usd").reset_index(drop=True)
comparison.to_csv(output_dir / "validation_comparison.csv", index=False)
errors = pd.concat(error_frames, ignore_index=True)
errors.drop(columns="squared_error").to_csv(output_dir / "validation_errors.csv", index=False)
grouped = errors.groupby(["model", "price_range"], observed=True)
error_summary = grouped.agg(count=("actual_price", "size"), mae_usd=("absolute_error_usd", "mean"),
                            median_absolute_error_usd=("absolute_error_usd", "median"),
                            mean_residual_usd=("residual_usd", "mean"),
                            mean_squared_error=("squared_error", "mean"),
                            mean_absolute_percentage_error=("absolute_percentage_error", "mean"))
error_summary["rmse_usd"] = np.sqrt(error_summary.pop("mean_squared_error"))
error_summary["p90_absolute_error_usd"] = grouped["absolute_error_usd"].quantile(0.9)
error_summary.reset_index().to_csv(output_dir / "validation_errors_by_price.csv", index=False)
errors.sort_values("absolute_error_usd", ascending=False).groupby("model", sort=False).head(10).drop(
    columns="squared_error").to_csv(output_dir / "largest_validation_errors.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for column, name in enumerate(selected):
    model_errors = errors.loc[errors["model"] == name]
    ax = axes[0, column]
    ax.scatter(model_errors["actual_price"], model_errors["predicted_price"], s=10, alpha=0.25)
    limits = [min(model_errors["actual_price"].min(), model_errors["predicted_price"].min()),
              max(model_errors["actual_price"].max(), model_errors["predicted_price"].max())]
    ax.plot(limits, limits, "--", color="darkred", label="Perfect prediction")
    ax.set(title=name, xlabel="Actual Price (USD)", ylabel="Predicted Price (USD)")
    ax.legend()
    ax = axes[1, column]
    ax.hist(model_errors["residual_usd"], bins=70, color="steelblue", edgecolor="white")
    ax.axvline(0, color="darkred", linestyle="--")
    ax.set(xlabel="Actual - Predicted Price (USD)", ylabel="Number of Diamonds")
fig.suptitle("Tuned Models: Validation Predictions and Residuals")
fig.tight_layout()
fig.savefig(output_dir / "validation_diagnostics.png", dpi=200)
plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 6))
error_summary["mae_usd"].unstack("model").plot.bar(ax=ax, rot=0)
ax.set(title="Validation MAE by Actual Price Range", xlabel="Actual Price Range (USD)", ylabel="MAE (USD)")
ax.legend(title="Tuned Model")
fig.tight_layout()
fig.savefig(output_dir / "validation_mae_by_price.png", dpi=200)
plt.close(fig)

for path in protected_paths:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == protected_hashes[str(path)]
metadata = {
    "phase": "3B", "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__, "matplotlib": matplotlib.__version__},
    "selection_policy": "Each family selected by minimum mean training-only group CV MAE; provisional preference among baseline and tuned models by existing validation MAE.",
    "preferred_model": comparison.loc[0, ["model", "experiment"]].to_dict(),
    "candidates": candidates, "best_configurations": best_configurations, "cv_results": cv_summaries,
    "cv": {"strategy": "GroupKFold", "n_splits": 3, "shuffle": True, "random_state": 42,
           "groups": int(groups.nunique()), "fold_sizes": fold_sizes,
           "group_policy": "Exact raw input tuples, excluding price, identifier and source_row."},
    "input_columns": input_columns, "protected_sha256": protected_hashes,
    "price_bins_usd": "[0,1000), [1000,3000), [3000,5000), [5000,10000), [10000,infinity)",
    "verification": "Passed: baseline input hashes match; artifacts compatible; no target/group leakage; fold scaler means, sample counts and encoder categories checked; CV scores repeated; selected full-train fits repeated; pipeline reload predictions match; protected files unchanged.",
    "test_policy": "Test files hashed only; no test features or targets loaded for predictions, scores or tuning.",
    "warning_policy": "Run with -W error; any runtime warning stops execution.",
    "elapsed_seconds": perf_counter() - started,
    "limitations": "Small joint candidate search, not exhaustive or causal parameter attribution. CV selection scores have selection optimism. Validation has been reused across phases; test generalization remains unmeasured.",
}
(output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("\n" + comparison.round(4).to_string(index=False), flush=True)
print("\nErrors by actual price range:\n" + error_summary.round(2).to_string(), flush=True)
print("\nCV results:", json.dumps(cv_summaries, indent=2), flush=True)
print("\nPreferred:", metadata["preferred_model"], flush=True)
print("All checks passed. No test evaluation. Phase 3B complete.", flush=True)
