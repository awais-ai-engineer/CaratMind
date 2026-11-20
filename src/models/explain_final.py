"""Reproduce final diagnostics from saved predictions; explain the frozen booster."""
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error
from threadpoolctl import threadpool_limits

folder = Path("artifacts/final_evaluation")
manifest = json.loads((folder / "frozen_selection.json").read_text())
evaluation = json.loads((folder / "evaluation_verification.json").read_text())
assert evaluation["test_prediction_passes_per_model"] == 1
columns = manifest["input_columns"]
bands = [0, 1000, 3000, 5000, 10000, np.inf]
labels = ["0-999", "1000-2999", "3000-4999", "5000-9999", "10000+"]
test = pd.read_csv("data/processed/test.csv")
metrics, range_metrics, predictions = [], [], {}
validation = pd.DataFrame(manifest["validation_results"]).set_index("model")
for name in manifest["model_paths"]:
    saved = pd.read_csv(folder / f"{name}_test_predictions.csv", float_precision="round_trip")
    assert len(saved) == 8089
    np.testing.assert_array_equal(saved["source_row"], test["source_row"])
    np.testing.assert_array_equal(saved["actual_price"], test["price"])
    actual, predicted = saved["actual_price"].to_numpy(), saved["predicted_price"].to_numpy()
    assert np.isfinite(predicted).all() and (actual > 0).all()
    error = predicted - actual  # Positive bias means overprediction, in every report.
    absolute = np.abs(error)
    mae, rmse = mean_absolute_error(actual, predicted), root_mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    # Independent NumPy calculation checks the library metrics without another prediction pass.
    np.testing.assert_allclose([mae, rmse, r2, median_absolute_error(actual, predicted)],
        [absolute.mean(), np.sqrt(np.mean(error ** 2)),
         1 - np.sum(error ** 2) / np.sum((actual - actual.mean()) ** 2), np.median(absolute)], rtol=1e-12)
    metrics.append({"model": name, "sample_count": len(actual), "mae_usd": mae,
        "rmse_usd": rmse, "r2": r2, "median_absolute_error_usd": float(np.median(absolute)),
        "p90_absolute_error_usd": float(np.quantile(absolute, .9)),
        "mean_prediction_bias_usd": float(error.mean()), "mape_percent": float(100 * np.mean(absolute / actual)),
        "wape_percent": float(100 * absolute.sum() / actual.sum()),
        "validation_mae_usd": float(validation.loc[name, "validation_mae_usd"]),
        "validation_rmse_usd": float(validation.loc[name, "validation_rmse_usd"]),
        "validation_r2": float(validation.loc[name, "validation_r2"]),
        "test_minus_validation_mae_usd": float(mae - validation.loc[name, "validation_mae_usd"])})
    saved["signed_error_usd"], saved["absolute_error_usd"] = error, absolute
    saved["price_range"] = pd.cut(actual, bands, labels=labels, right=False)
    for label, group in saved.groupby("price_range", observed=True):
        range_metrics.append({"model": name, "price_range": str(label), "sample_count": len(group),
            "mae_usd": float(group["absolute_error_usd"].mean()),
            "rmse_usd": float(np.sqrt(np.mean(group["signed_error_usd"] ** 2))),
            "median_absolute_error_usd": float(group["absolute_error_usd"].median()),
            "mean_signed_error_usd": float(group["signed_error_usd"].mean())})
    predictions[name] = saved
results, ranges = pd.DataFrame(metrics), pd.DataFrame(range_metrics)
results.to_csv(folder / "test_metrics.csv", index=False)
(folder / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
ranges.to_csv(folder / "price_range_errors.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, saved) in zip(axes, predictions.items()):
    ax.scatter(saved["actual_price"], saved["predicted_price"], s=5, alpha=.22)
    limits = [0, max(saved["actual_price"].max(), saved["predicted_price"].max())]
    ax.plot(limits, limits, color="black", linestyle="--")
    ax.set(title=name.replace("Regressor", "").replace("phase4a_deep_raw", "ANN seed 42"),
           xlabel="Actual price (USD)", ylabel="Predicted price (USD)")
fig.tight_layout()
fig.savefig(folder / "actual_vs_predicted.png", dpi=160)
plt.close(fig)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for column, (name, saved) in enumerate(predictions.items()):
    axes[0, column].scatter(saved["actual_price"], saved["signed_error_usd"], s=5, alpha=.22)
    axes[0, column].axhline(0, color="black", linestyle="--")
    axes[0, column].set(title=name.replace("Regressor", ""), xlabel="Actual price (USD)", ylabel="Prediction - actual (USD)")
    axes[1, column].hist(saved["signed_error_usd"], bins=80)
    axes[1, column].set(xlabel="Prediction - actual (USD)", ylabel="Count")
fig.tight_layout()
fig.savefig(folder / "residuals.png", dpi=160)
plt.close(fig)

# Explain original input columns through the complete persisted pipeline.
# Validation labels are used here; no test labels enter importance or PDP.
booster = joblib.load(manifest["model_paths"][manifest["primary_model"]])
reference = pd.read_csv("data/processed/validation.csv")
training = pd.read_csv("data/processed/train.csv")
importance_path = folder / "permutation_importance.csv"
if not importance_path.exists():
    print("Computing validation permutation importance: 9 original features, 5 repeats.", flush=True)
    with threadpool_limits(limits=1):
        importance = permutation_importance(booster, reference[columns], reference["price"],
            scoring="neg_mean_absolute_error", n_repeats=5, random_state=42, n_jobs=1)
    importance_table = pd.DataFrame({"feature": columns, "mean_mae_increase_usd": importance.importances_mean,
                                      "repeat_sd_usd": importance.importances_std})
    importance_table = importance_table.sort_values("mean_mae_increase_usd", ascending=False)
    importance_table.to_csv(importance_path, index=False)
    pd.DataFrame(importance.importances.T, columns=columns).to_csv(folder / "permutation_repeats.csv", index=False)
else:
    importance_table = pd.read_csv(importance_path)
fig, ax = plt.subplots(figsize=(9, 5))
ordered = importance_table.iloc[::-1]
ax.barh(ordered["feature"], ordered["mean_mae_increase_usd"], xerr=ordered["repeat_sd_usd"])
ax.set(xlabel="Validation MAE increase after permutation (USD; mean +/- repeat SD)", title="Frozen boosting model: predictive importance")
fig.tight_layout()
fig.savefig(folder / "permutation_importance.png", dpi=160)
plt.close(fig)
numeric = ["carat", "depth", "table", "x", "y", "z"]
pdp_features = importance_table.loc[importance_table["feature"].isin(numeric), "feature"].head(3).tolist()
pdp_reference = training[columns].sample(n=2000, random_state=42)
pdp_rows = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, feature in zip(axes, pdp_features):
    with threadpool_limits(limits=1):
        dependence = partial_dependence(booster, pdp_reference, [feature], method="brute",
                                       grid_resolution=25, percentiles=(.05, .95))
    grid, average = dependence["grid_values"][0], dependence["average"][0]
    pdp_rows.extend({"feature": feature, "feature_value": float(x), "mean_predicted_price_usd": float(y)} for x, y in zip(grid, average))
    ax.plot(grid, average)
    ax.set(xlabel=feature, ylabel="Average predicted price (USD)", title=f"Partial dependence: {feature}")
fig.tight_layout()
fig.savefig(folder / "partial_dependence.png", dpi=160)
plt.close(fig)
pd.DataFrame(pdp_rows).to_csv(folder / "partial_dependence.csv", index=False)
mapping = []
for encoded in manifest["transformed_features"]:
    original = encoded.split("__", 1)[1]
    if encoded.startswith("categorical__"):
        original = original.split("_", 1)[0]
    assert original in columns
    mapping.append({"transformed_feature": encoded, "original_feature": original})
pd.DataFrame(mapping).to_csv(folder / "feature_mapping.csv", index=False)
explanation = {
    "model": manifest["primary_model"], "importance_reference": "all 8,097 validation rows; MAE; five permutations, seed 42",
    "permutation_unit": "Original input column, through persisted preprocessing; all encoded levels move together.",
    "pdp_reference": "2,000 training rows sampled with seed 42, 25 grid points between 5th and 95th percentiles; brute method",
    "pdp_features": pdp_features, "feature_selection_basis": "Validation permutation importance only",
    "limitations": "Importance measures predictive reliance, not correlation or causation. Correlated carat/x/y/z may share or mask importance. Permutation and PDP may create unrealistic combinations. Repeat SD is permutation variability, not a population confidence interval. No causal interpretation.",
    "shap": "Not required; permutation importance and PDP use established installed libraries without new dependencies.",
    "relative_error_policy": "MAPE = mean(abs(prediction-actual)/actual)*100; positive targets verified. WAPE = sum(abs(error))/sum(actual)*100. MAPE emphasizes lower prices; neither replaces USD errors or proves market accuracy.",
    "signed_error_convention": "prediction minus actual; positive means overprediction",
}
(folder / "explainability.json").write_text(json.dumps(explanation, indent=2), encoding="utf-8")

metric_text = results.to_string(index=False, float_format=lambda value: f"{value:.4f}")
range_text = ranges.to_string(index=False, float_format=lambda value: f"{value:.4f}")
card = f"""# CaratMind model card

## Intended use
Educational dataset-based diamond price estimation in USD. The frozen primary model is tuned HistGradientBoosting, chosen using validation MAE before final testing. It is not a guaranteed real-time market valuation engine. No deployment is performed.

## Dataset and input schema
Original dataset: 53,940 records, 11 columns, data/raw/10-diamonds (1).csv. Historical provenance, market date, currency representativeness and independent external validity are not established by these experiments. Inputs, in order: {', '.join(columns)}. carat is weight; x/y/z are dimensions in mm; depth and table are percentage measurements; cut/color/clarity are categorical grades. Target: price, USD. source_row and Unnamed: 0 are audit identifiers, excluded from inputs.

## Cleaning and methodology
Remove nonpositive dimensions or y/z >20 mm: 23 removed, 53,917 retained. Keep 48 depth mismatches exceeding 2 percentage points for investigation. Keep duplicate records; exact nine-input tuples are grouped independently of price. Seed-42 group split: 37,731 train / 8,097 validation / 8,089 test. No identical input groups cross partitions. StandardScaler and safe one-hot encoding are fitted on training only; 26 transformed inputs. Unknown categories encode as zeros and need monitoring. All numerical features are retained, including correlated size measurements.

ML tuning used training-only three-fold group-aware CV with fold-specific preprocessing. Final models remain fitted on the original training set, without train/validation refitting. ANN: 128/64/32 ReLU, linear output, Adam .001, original-price MSE, batch 64, maximum 150 epochs, early stopping patience 15 restoring weights; fixed representative seed 42. Selected ANN three-seed validation MAE: 280.6551 +/- 2.9162 USD (sample SD). This measures seed stability on one split, not population uncertainty.

## Frozen configurations and versions
Full model parameters, ANN architecture, selected artifact paths, SHA256 hashes, package versions and freeze timestamp are in frozen_selection.json. Artifacts are unchanged; models and preprocessing use the exact persisted versions. Comparators never replace the validation-selected primary based on test results.

## Validation and final test performance
Bias means prediction minus actual. All three candidates were predicted once. Reports are reproducible from saved predictions without further test inference.

```text
{metric_text}
```

## Price ranges and error behavior
```text
{range_text}
```
Higher-priced examples have larger absolute errors; signed errors in the table identify systematic underprediction (negative) or overprediction (positive). These are dataset price bands, not demographic fairness groups. Rare grades and extreme sizes may have limited representation. Aggregate metrics can conceal substantial individual errors. Relative errors use positive actual prices; MAPE weights lower prices more heavily, and WAPE reflects price composition. Neither provides an accuracy guarantee.

## Explainability
Permutation importance uses validation labels and original columns through the complete pipeline. PDP uses 2,000 training reference rows, and its numerical features are selected from validation importance. Encoded-to-original mappings are in feature_mapping.csv. Predictive importance does not establish causality. Correlated size variables can mask or share importance; permutation and PDP can create implausible feature combinations. PDP is an averaged model response, not a promised price change for an individual diamond. No test labels guide feature selection or explanation settings.

## Appropriate and inappropriate use
Appropriate: assessment demonstrations, analysis of comparable in-distribution records, and human-reviewed exploratory estimates. Inappropriate: certified appraisal, guaranteed transaction prices, financial decisions without independent expertise, unverified out-of-distribution diamonds, or claims of current-market accuracy. Condition, provenance, certification, location, market changes and other unobserved factors may matter. Exact duplicate grouping does not guarantee all near-duplicates or hidden dependencies are separated. No confidence intervals or statistical superiority claims are made. The final test set is now consumed and cannot be reused as an untouched holdout for future tuning.
"""
(folder / "model_card.md").write_text(card, encoding="utf-8")

for path, expected in manifest["protected_sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
figures = list(folder.glob("*.png"))
assert len(figures) == 4
for path in figures:
    with Image.open(path) as image:
        image.verify()
outputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()
           if p.is_file() and p.name != "verification.json"}
(folder / "verification.json").write_text(json.dumps({
    "checks_passed": ["saved predictions aligned with 8,089 test targets", "all predictions finite",
        "NumPy independently reproduces sklearn MAE/RMSE/R2/median", "positive denominators for relative metrics",
        "all earlier data, scripts and artifacts unchanged", "four valid PNG images",
        "explanations use training/validation only", "primary model remains frozen"],
    "failed_checks": [], "evaluation": evaluation, "output_sha256": outputs,
    "warnings_and_limitations": ["Test set now consumed; do not use for future tuning.",
        "No repeated test prediction passes; reproducibility checks recompute metrics from saved outputs.",
        "Correlated features limit permutation/PDP interpretation; no causal or statistical significance claims."],
}, indent=2), encoding="utf-8")
print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print("\nPrice-range diagnostics:\n", range_text)
print("\nValidation permutation importance:\n", importance_table.to_string(index=False))
print("\nAll checks passed; no model fitting, test retuning, or new test predictions.")
