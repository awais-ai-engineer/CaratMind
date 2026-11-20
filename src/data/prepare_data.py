import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


raw_path = Path("data/raw/10-diamonds (1).csv")
raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
df = pd.read_csv(raw_path)
print(f"Before cleaning: {df.shape}")
print("Repeated records excluding identifier:", df.drop(columns="Unnamed: 0").duplicated().sum())

# Fail explicitly if new data needs an additional missing/non-finite policy.
if df.isna().any().any():
    raise ValueError("Missing values found; review the data before preparing splits.")
if not np.isfinite(df.select_dtypes(include="number")).all().all():
    raise ValueError("Non-finite numerical values found; review the data.")

# Fixed rules from the earlier audit, not thresholds fitted to this split.
invalid_dimensions = (df[["x", "y", "z"]] <= 0).any(axis=1)
extreme_dimensions = (df[["y", "z"]] > 20).any(axis=1)
clean = df.loc[~(invalid_dimensions | extreme_dimensions)].copy()
print("Non-positive dimension rows:", invalid_dimensions.sum())
print("Rows with y or z > 20 mm:", extreme_dimensions.sum())
print(f"After cleaning: {clean.shape}")

calculated_depth = 200 * clean["z"] / (clean["x"] + clean["y"])
depth_difference = (clean["depth"] - calculated_depth).abs()
depth_audit = clean[["Unnamed: 0", "depth", "x", "y", "z"]].copy()
depth_audit["calculated_depth"] = calculated_depth
depth_audit["difference_pp"] = depth_difference
print("\nDepth mismatches > 2 percentage points retained:", (depth_difference > 2).sum())
print(depth_audit.loc[depth_difference > 2].sort_values("difference_pp", ascending=False)
      .head(10).round(4).to_string(index=False))

X = clean.drop(columns=["Unnamed: 0", "price"])
y = clean["price"]
assert "price" not in X.columns and "Unnamed: 0" not in X.columns
numeric_columns = X.select_dtypes(include="number").columns.tolist()
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
print("\nInput features:", X.columns.tolist())

# Exact input tuples define groups, independently of the target and identifier.
groups = X.groupby(X.columns.tolist(), sort=False).ngroup()
train_split = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_index, remaining_index = next(train_split.split(X, groups=groups))
evaluation_split = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_relative, test_relative = next(evaluation_split.split(
    X.iloc[remaining_index], groups=groups.iloc[remaining_index]))
validation_index = remaining_index[val_relative]
test_index = remaining_index[test_relative]
partitions = {"train": train_index, "validation": validation_index, "test": test_index}

# Re-run the seeded split to check reproducibility and complete coverage.
train_again, remaining_again = next(train_split.split(X, groups=groups))
val_again, test_again = next(evaluation_split.split(
    X.iloc[remaining_again], groups=groups.iloc[remaining_again]))
assert np.array_equal(train_index, train_again)
assert np.array_equal(validation_index, remaining_again[val_again])
assert np.array_equal(test_index, remaining_again[test_again])
assert np.array_equal(np.sort(np.concatenate(list(partitions.values()))), np.arange(len(X)))
group_sets = [set(groups.iloc[index]) for index in partitions.values()]
assert group_sets[0].isdisjoint(group_sets[1])
assert group_sets[0].isdisjoint(group_sets[2])
assert group_sets[1].isdisjoint(group_sets[2])

preprocessor = Pipeline([
    ("columns", ColumnTransformer([
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical_columns),
    ], remainder="drop")),
])
preprocessor.fit(X.iloc[train_index])
feature_names = preprocessor.get_feature_names_out().tolist()
assert preprocessor.feature_names_in_.tolist() == X.columns.tolist()
scaler = preprocessor.named_steps["columns"].named_transformers_["numeric"]
assert scaler.n_samples_seen_ == len(train_index)
np.testing.assert_allclose(scaler.mean_, X.iloc[train_index][numeric_columns].mean())

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)
preparation_dir = Path("artifacts/preparation")
preparation_dir.mkdir(parents=True, exist_ok=True)
split_sizes = {}
for name, index in partitions.items():
    transformed = preprocessor.transform(X.iloc[index])
    assert np.isfinite(transformed).all()
    assert transformed.shape == (len(index), len(feature_names))
    assert not y.iloc[index].isna().any()
    print(f"{name}: {len(index):,} rows ({len(index) / len(X):.3%}); transformed {transformed.shape}")
    split_sizes[name] = {"rows": len(index), "groups": int(groups.iloc[index].nunique()),
                         "transformed_shape": list(transformed.shape)}
    # Source row is metadata for auditing, never a modeling input.
    split_data = clean.iloc[index].drop(columns="Unnamed: 0").copy()
    split_data.insert(0, "source_row", clean.index[index])
    split_data.to_csv(processed_dir / f"{name}.csv", index=False)
    np.savez_compressed(processed_dir / f"{name}_transformed.npz",
                        X=transformed, y=y.iloc[index].to_numpy(),
                        source_row=clean.index[index].to_numpy(),
                        feature_names=np.array(feature_names))

# An unseen category must transform without fitting the encoder again.
unseen = X.iloc[train_index[:1]].copy()
unseen[categorical_columns[0]] = "__unseen_category_check__"
assert np.isfinite(preprocessor.transform(unseen)).all()
joblib.dump(preprocessor, preparation_dir / "preprocessor.joblib")
restored = joblib.load(preparation_dir / "preprocessor.joblib")
np.testing.assert_allclose(restored.transform(X.iloc[test_index[:5]]),
                           preprocessor.transform(X.iloc[test_index[:5]]))
assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_hash

report = {
    "raw_sha256": raw_hash,
    "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                 "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    "random_state": 42,
    "raw_rows": len(df), "clean_rows": len(clean),
    "removed_rows": int((invalid_dimensions | extreme_dimensions).sum()),
    "depth_mismatches_retained": int((depth_difference > 2).sum()),
    "cleaning_policy": "Remove any x/y/z <= 0 or y/z > 20 mm; retain depth mismatches > 2 pp.",
    "duplicate_policy": "Retain repeated rows; group by exact input tuples, excluding price and identifier.",
    "split_policy": "70/15/15 of input groups using two GroupShuffleSplit calls; row proportions are approximate.",
    "preprocessing_policy": "StandardScaler and one-hot categories fitted only on train; unknown categories map to all zeros.",
    "artifact_policy": "CSV source_row is raw zero-based row metadata, not an input; NPZ X/y are separate arrays.",
    "numeric_columns": numeric_columns, "categorical_columns": categorical_columns,
    "transformed_features": feature_names, "splits": split_sizes,
    "checks": "Passed: no missing/non-finite values, no target/identifier inputs, no group overlap, complete coverage, seeded reproducibility, train-only scaling, unseen category, pipeline reload, raw hash.",
}
(preparation_dir / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\nAll checks passed. Saved split CSV/NPZ files, fitted preprocessor and preparation report.")
print("No predictive model trained. Phase 2 complete.")
