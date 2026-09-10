"""
WEEK 2 - Exploratory Data Analysis and Visualization

Dataset:
Breast Cancer Wisconsin (Diagnostic) dataset
Public source: UCI Machine Learning Repository
Loaded through scikit-learn's packaged copy.

Outputs:
- descriptive_statistics.csv
- target_distribution.csv
- dataset_profile.csv
- breast_cancer_eda_dataset.csv
- PNG visualizations
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

OUTPUT = Path("week2_outputs")
OUTPUT.mkdir(exist_ok=True)

# 1. Load public dataset
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df["target_name"] = df["target"].map({0: "malignant", 1: "benign"})

# 2. Initial EDA
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(10))
print("\nDuplicate rows:", df.duplicated().sum())
print("\nDescriptive statistics:")
print(df.describe().T)

# 3. Save statistics
df.describe().T.to_csv(OUTPUT / "descriptive_statistics.csv")
df["target_name"].value_counts().to_csv(OUTPUT / "target_distribution.csv")
df.to_csv(OUTPUT / "breast_cancer_eda_dataset.csv", index=False)

# 4. Target distribution
counts = df["target_name"].value_counts()
plt.figure(figsize=(7, 5))
counts.plot(kind="bar")
plt.title("Distribution of Breast Cancer Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT / "01_target_distribution.png", dpi=200)
plt.close()

# 5. Histograms
for col in ["mean radius", "mean texture", "mean perimeter", "mean area"]:
    plt.figure(figsize=(7, 5))
    plt.hist(df[col], bins=25)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT / f"hist_{col.replace(' ', '_')}.png", dpi=200)
    plt.close()

# 6. Group comparison
grouped = df.groupby("target_name")["mean radius"].mean().sort_values()
plt.figure(figsize=(7, 5))
grouped.plot(kind="bar")
plt.title("Average Mean Radius by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Average Mean Radius")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT / "02_mean_radius_by_class.png", dpi=200)
plt.close()

# 7. Correlation analysis
cols = ["mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity", "mean symmetry"]
corr = df[cols].corr()
plt.figure(figsize=(9, 7))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(cols)), cols, rotation=60, ha="right")
plt.yticks(range(len(cols)), cols)
plt.title("Correlation Matrix of Selected Features")
plt.tight_layout()
plt.savefig(OUTPUT / "03_correlation_heatmap.png", dpi=200)
plt.close()

# 8. Relationship between two variables
plt.figure(figsize=(7, 5))
for label, marker in [("malignant", "o"), ("benign", "x")]:
    subset = df[df["target_name"] == label]
    plt.scatter(subset["mean radius"], subset["mean texture"],
                alpha=0.55, s=18, marker=marker, label=label)
plt.title("Mean Radius vs Mean Texture")
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "04_radius_vs_texture.png", dpi=200)
plt.close()

# 9. Box plot
groups = [df.loc[df["target_name"] == label, "mean area"]
          for label in ["malignant", "benign"]]
plt.figure(figsize=(7, 5))
plt.boxplot(groups, tick_labels=["malignant", "benign"])
plt.title("Mean Area by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Mean Area")
plt.tight_layout()
plt.savefig(OUTPUT / "05_mean_area_boxplot.png", dpi=200)
plt.close()

# 10. Profile
profile = pd.Series({
    "rows": len(df),
    "columns": len(df.columns),
    "missing_cells": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "malignant": int((df["target_name"] == "malignant").sum()),
    "benign": int((df["target_name"] == "benign").sum()),
})
profile.to_csv(OUTPUT / "dataset_profile.csv")

print("\nEDA complete. Files saved to:", OUTPUT.resolve())
