import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/raw/10-diamonds (1).csv")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="price",
    bins=60,
    kde=True
)

plt.title("Diamond Price Distribution")
plt.xlabel("Price (USD)")
plt.ylabel("Number of Diamonds")

plt.tight_layout()
plt.savefig(
    "artifacts/figures/price_distribution.png",
    dpi=300
)
plt.show()

# Carat vs Price relationship

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="carat",
    y="price",
    alpha=0.3,
    s=15
)

plt.title("Diamond Carat vs Price")
plt.xlabel("Carat Weight")
plt.ylabel("Price (USD)")

plt.tight_layout()

plt.savefig(
    "artifacts/figures/carat_vs_price.png",
    dpi=300
)

plt.show()
correlation = df["carat"].corr(df["price"])

print(f"Carat vs Price Correlation: {correlation:.4f}")
