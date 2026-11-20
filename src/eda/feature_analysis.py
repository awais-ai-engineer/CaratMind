from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


df = pd.read_csv("data/raw/10-diamonds (1).csv")
figures_dir = Path("artifacts/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

numeric_columns = df.select_dtypes(include="number").columns.drop("Unnamed: 0")
numeric_data = df[numeric_columns]
finite_rows = np.isfinite(numeric_data).all(axis=1)
print(f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Rows with missing or infinite numerical values: {(~finite_rows).sum()}")

# Use one common finite sample for comparable Pearson correlations.
correlation = numeric_data.loc[finite_rows].corr(method="pearson")
print("\nPearson correlation matrix (identifier excluded):")
print(correlation.round(6).to_string())
price_correlation = correlation["price"].drop("price")
price_order = price_correlation.abs().sort_values(ascending=False).index
print("\nCorrelations with price, sorted by absolute strength:")
print(price_correlation.loc[price_order].round(6).to_string())
print("\nCarat and dimension correlations:")
print(correlation.loc[["carat", "x", "y", "z"], ["carat", "x", "y", "z"]].round(6).to_string())

invalid_dimensions = (df[["x", "y", "z"]] <= 0).any(axis=1)
extreme_dimensions = (df[["y", "z"]] > 20).any(axis=1)
print(f"\nRows with non-positive dimensions: {invalid_dimensions.sum()}")
print(f"Rows with y or z above 20 mm: {extreme_dimensions.sum()}")
print(f"Repeated records excluding identifier: {df.drop(columns='Unnamed: 0').duplicated().sum()}")

# VIF uses only numerical inputs; price and the volume proxy are not included.
inputs = numeric_data.drop(columns="price")
vif_rows = np.isfinite(inputs).all(axis=1)
vif_data = inputs.loc[vif_rows].copy()
print(f"\nVIF rows: {len(vif_data):,}; non-finite rows excluded: {(~vif_rows).sum()}")
print("Finite zero/extreme measurements remain in this diagnostic; no physical cleaning applied.")
constant_columns = vif_data.columns[vif_data.nunique() <= 1].tolist()
print(f"Constant columns excluded (VIF undefined): {constant_columns}")
vif_data = vif_data.drop(columns=constant_columns)
if vif_data.shape[1] < 2 or len(vif_data) <= vif_data.shape[1]:
    raise ValueError("VIF needs at least two varying inputs and more rows than inputs.")

# Centering/scaling improves numerical conditioning without changing intercept-based VIF.
standardized = (vif_data - vif_data.mean()) / vif_data.std(ddof=0)
vif_values = []
for column in standardized.columns:
    response = standardized[column].to_numpy()
    other_inputs = standardized.drop(columns=column).to_numpy()
    design = np.column_stack([np.ones(len(standardized)), other_inputs])
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    residuals = response - design @ coefficients
    residual_sum = np.sum(residuals ** 2)
    total_sum = np.sum((response - response.mean()) ** 2)
    # VIF = 1 / (1 - R squared); perfect linear dependence gives infinity.
    vif = np.inf if residual_sum <= np.finfo(float).eps * total_sum else total_sum / residual_sum
    vif_values.append(vif)

vif_report = pd.Series(vif_values, index=standardized.columns, name="VIF")
print("\nVIF (auxiliary linear regressions with an intercept):")
print(vif_report.sort_values(ascending=False).round(6).to_string())

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, square=True,
            cbar_kws={"label": "Pearson Correlation"})
plt.title("Feature Relationship Investigation: Pearson Correlations")
plt.tight_layout()
plt.savefig(figures_dir / "feature_analysis_correlation_heatmap.png", dpi=300)
plt.close()

# x*y*z approximates a bounding-box volume, not a diamond's true volume.
volume_proxy = df["x"] * df["y"] * df["z"]
volume_rows = np.isfinite(volume_proxy) & np.isfinite(df["carat"])
volume_correlation = df.loc[volume_rows, "carat"].corr(volume_proxy.loc[volume_rows])
print(f"\nCarat vs approximate x*y*z volume Pearson correlation: {volume_correlation:.6f}")
print(f"Non-finite carat/volume rows excluded from scatter: {(~volume_rows).sum()}")
flagged = invalid_dimensions | extreme_dimensions
plt.figure(figsize=(10, 6))
plt.scatter(df.loc[volume_rows & ~flagged, "carat"],
            volume_proxy.loc[volume_rows & ~flagged], s=12, alpha=0.25,
            label="Other records")
plt.scatter(df.loc[volume_rows & flagged, "carat"],
            volume_proxy.loc[volume_rows & flagged], s=35, color="darkorange",
            marker="x", label="Non-positive dimension or y/z > 20 mm")
plt.title(f"Carat vs Approximate Geometric Volume Proxy (r = {volume_correlation:.3f})")
plt.xlabel("Carat Weight")
plt.ylabel("Approximate Volume Proxy: x*y*z (mm³)")
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / "carat_vs_volume_proxy.png", dpi=300)
plt.close()

print("\nInterpretation:")
print("Pearson correlation measures linear association; correlation does not prove causation.")
print("High VIF indicates linear redundancy among these inputs, not feature uselessness.")
print("Outliers, repeated records and non-linear size relationships can affect these diagnostics.")
print("The volume proxy uses dimensions only, with no target information.")
print("No feature-selection decision is made before train/validation splitting.")
print("Raw data unchanged; no predictive model or preprocessing pipeline created.")
print(f"Saved 2 charts to {figures_dir}.")
