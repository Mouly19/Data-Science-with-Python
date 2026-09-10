"""
Week 1 - Data Acquisition, Cleaning and Preprocessing
Dataset: Titanic passenger dataset
Source: Seaborn public dataset

Run:
    pip install pandas numpy seaborn matplotlib
    python week1_data_cleaning.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("week1_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Acquire public dataset
df = sns.load_dataset("titanic")

print("Original shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isna().sum())

# Save initial profile
with open(OUTPUT_DIR / "initial_profile.txt", "w", encoding="utf-8") as f:
    f.write(f"Shape: {df.shape}\n\n")
    f.write("Data types:\n")
    f.write(df.dtypes.to_string())
    f.write("\n\nMissing values:\n")
    f.write(df.isna().sum().to_string())
    f.write("\n\nDescriptive statistics:\n")
    f.write(df.describe(include="all").to_string())

# 2. Duplicate check
duplicate_count = int(df.duplicated().sum())
print("\nDuplicate rows:", duplicate_count)

df = df.drop_duplicates().copy()

# 3. Missing-value treatment
# Numerical: median
if "age" in df.columns:
    df["age"] = df["age"].fillna(df["age"].median())

# Categorical: mode
for col in ["embarked", "embark_town"]:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

# deck has substantial missingness; remove it
if "deck" in df.columns:
    df = df.drop(columns=["deck"])

# 4. Consistency checks
for col in ["sex", "embarked", "embark_town", "class", "who", "adult_male", "alone"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Convert target to integer if present
if "survived" in df.columns:
    df["survived"] = pd.to_numeric(df["survived"], errors="coerce").astype("Int64")

# 5. Outlier flagging using IQR
outlier_summary = []

for col in ["age", "fare", "sibsp", "parch"]:
    if col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)

        outlier_summary.append({
            "column": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "potential_outliers": int(mask.sum())
        })

outlier_df = pd.DataFrame(outlier_summary)
outlier_df.to_csv(OUTPUT_DIR / "outlier_summary.csv", index=False)

# 6. One-hot encode selected categorical variables for a model-ready file
categorical_cols = [
    col for col in ["sex", "embarked", "class", "who"]
    if col in df.columns
]

model_ready = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

# 7. Save cleaned data
df.to_csv(OUTPUT_DIR / "titanic_cleaned.csv", index=False)
model_ready.to_csv(OUTPUT_DIR / "titanic_model_ready.csv", index=False)

# 8. Generate simple before/after missing-value chart
before_missing = sns.load_dataset("titanic").isna().sum()
after_missing = df.isna().sum()

missing_compare = pd.DataFrame({
    "Before cleaning": before_missing,
    "After cleaning": after_missing
}).fillna(0)

missing_compare.plot(kind="bar", figsize=(12, 6))
plt.title("Missing Values Before and After Cleaning")
plt.xlabel("Column")
plt.ylabel("Number of Missing Values")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "missing_values_before_after.png", dpi=200)
plt.close()

# 9. Final report
print("\nCleaned shape:", df.shape)
print("\nRemaining missing values:")
print(df.isna().sum())
print("\nOutputs saved in:", OUTPUT_DIR.resolve())
