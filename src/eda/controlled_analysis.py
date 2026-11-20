import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/10-diamonds (1).csv")

# Select diamonds between 0.95 and 1.05 carats
diamonds_1ct = df[df["carat"].between(0.95, 1.05)].copy()

print("Selected Diamonds:", diamonds_1ct.shape[0])

# Overall price distribution
print("\nPrice Statistics:")
print(diamonds_1ct["price"].describe())

# Price comparison by cut
print("\nPrice by Cut:")
print(
    diamonds_1ct.groupby("cut")["price"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

# Price comparison by color
print("\nPrice by Color:")
print(
    diamonds_1ct.groupby("color")["price"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

# Price comparison by clarity
print("\nPrice by Clarity:")
print(
    diamonds_1ct.groupby("clarity")["price"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)
# Compare cut within the same color and clarity grades
controlled_group = diamonds_1ct[
    (diamonds_1ct["color"] == "G") &
    (diamonds_1ct["clarity"] == "VS1")
]

print("\nControlled Comparison: Color G, Clarity VS1")

print(
    controlled_group.groupby("cut")["price"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

# Verify actual carat variation within groups
print("\nCarat Range by Cut:")

print(
    controlled_group.groupby("cut")["carat"]
    .agg(["min", "max", "mean"])
    .round(3)
)