import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
real = pd.read_csv("dataset.csv")
fake = pd.read_csv("synthetic_data.csv")

# Professional theme
sns.set_style("whitegrid")

# Color palette
colors = ["#1f77b4", "#ff7f0e"]  # blue & orange

for col in real.columns:
    plt.figure(figsize=(8,5))

    sns.kdeplot(real[col], label="Real Data", fill=True, color=colors[0], linewidth=2)
    sns.kdeplot(fake[col], label="Synthetic Data", fill=True, color=colors[1], linewidth=2)

    plt.title(f"{col} Distribution Comparison", fontsize=14, fontweight='bold')
    plt.xlabel(col)
    plt.ylabel("Density")

    plt.legend()
    plt.grid(alpha=0.3)

    # Save image
    plt.savefig(f"{col}_professional.png")

    plt.show()