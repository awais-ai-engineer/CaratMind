import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/10-diamonds (1).csv")

# Dataset dimensions
print("Dataset Shape:", df.shape)

# Preview dataset
print(df.head()) 

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Inspect data types and non-null counts
print("\nDataset Information:")
df.info()

# Numerical statistics
print("\nNumerical Summary:")
print(df.describe().T)

# Categorical statistics
print("\nCategorical Summary:")
print(df.describe(include="object").T)

# Unique categories
for column in df.select_dtypes(include="object").columns:
    print(f"\n{column} categories:")
    print(df[column].value_counts())

# 1. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Duplicate records
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# 3. Zero dimension counts
print("\nZero Dimensions:")
print((df[["x", "y", "z"]] == 0).sum())

# 4. Suspicious dimension records
print("\nSuspicious Dimensions:")
suspicious = df[
    (df["x"] == 0) |
    (df["y"] == 0) |
    (df["z"] == 0) |
    (df["y"] > 20) |
    (df["z"] > 20)
]

print(suspicious)

# 5. Skewness
print("\nNumerical Skewness:")
print(df.select_dtypes(include="number").skew())

# Duplicate records excluding identifier
features = df.drop(columns=["Unnamed: 0"])

duplicates = features.duplicated().sum()

print("\nDuplicate Records:", duplicates)


# Invalid physical dimensions
invalid_dimensions = (
    (df[["x", "y", "z"]] <= 0).any(axis=1)
)

print("\nInvalid Dimension Rows:")
print(invalid_dimensions.sum())


# Extremely large measurements
extreme_dimensions = (
    (df["y"] > 20) |
    (df["z"] > 20)
)

print("\nExtreme Dimension Rows:")
print(extreme_dimensions.sum())


# Combined suspicious records
suspicious = invalid_dimensions | extreme_dimensions

print("\nTotal Suspicious Rows:")
print(suspicious.sum())


# Verify depth percentage consistency
valid = df.loc[
    ~invalid_dimensions &
    ~extreme_dimensions
].copy()

calculated_depth = (
    2 * valid["z"] /
    (valid["x"] + valid["y"])
) * 100

depth_difference = (
    valid["depth"] - calculated_depth
).abs()

print("\nDepth Difference Statistics:")
print(depth_difference.describe())

print("\nRows with depth difference > 2:")
print((depth_difference > 2).sum())

# Investigate inconsistent depth measurements
depth_audit = valid.copy()

depth_audit["calculated_depth"] = calculated_depth
depth_audit["depth_difference"] = depth_difference

inconsistent = depth_audit[
    depth_audit["depth_difference"] > 2
].copy()

inconsistent = inconsistent.sort_values(
    by="depth_difference",
    ascending=False
)

print("\nTop 15 Depth Inconsistencies:")

print(
    inconsistent[
        [
            "carat",
            "price",
            "depth",
            "calculated_depth",
            "depth_difference",
            "x",
            "y",
            "z"
        ]
    ].head(15).to_string(index=False)
)
