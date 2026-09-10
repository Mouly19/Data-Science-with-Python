"""
WEEK 6 - INTEGRATIVE CAPSTONE PROJECT AND EVALUATION
Project: End-to-End Wine Classification, Clustering and Neural Network Analysis

This project integrates:
1. Data acquisition
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Unsupervised learning (K-Means)
5. Supervised learning (Random Forest)
6. Neural-network classification (MLP)
7. Model evaluation and comparison
8. Insights and recommendations

Dataset: UCI Wine dataset, accessed through scikit-learn's public loader.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score, adjusted_rand_score
)

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. DATA ACQUISITION
# ---------------------------------------------------------
wine = load_wine(as_frame=True)
df = wine.frame.copy()

target_name = "target"
feature_names = wine.feature_names

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive statistics:\n", df.describe())

# ---------------------------------------------------------
# 2. DATA CLEANING / PREPROCESSING
# ---------------------------------------------------------
# Remove exact duplicate rows if present.
df = df.drop_duplicates().reset_index(drop=True)

X = df[feature_names]
y = df[target_name]

# Median imputation + standardization are included in pipelines.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
X_all_scaled = preprocessor.fit_transform(X)

# ---------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------
class_counts = y.value_counts().sort_index()
plt.figure(figsize=(7, 5))
plt.bar(class_counts.index.astype(str), class_counts.values)
plt.title("Wine Class Distribution")
plt.xlabel("Wine class")
plt.ylabel("Number of samples")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=200)
plt.close()

corr = df[feature_names].corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(feature_names)), feature_names, rotation=90)
plt.yticks(range(len(feature_names)), feature_names)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"), dpi=200)
plt.close()

# PCA projection for visualization
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_all_scaled)

plt.figure(figsize=(8, 6))
for cls in sorted(y.unique()):
    mask = y.to_numpy() == cls
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Class {cls}")
plt.title("PCA Projection of Wine Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_classes.png"), dpi=200)
plt.close()

# ---------------------------------------------------------
# 4. UNSUPERVISED LEARNING - K-MEANS
# ---------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=20)
clusters = kmeans.fit_predict(X_all_scaled)

silhouette = silhouette_score(X_all_scaled, clusters)
ari = adjusted_rand_score(y, clusters)

print(f"\nK-Means silhouette score: {silhouette:.4f}")
print(f"K-Means adjusted Rand index: {ari:.4f}")

plt.figure(figsize=(8, 6))
for cluster_id in sorted(np.unique(clusters)):
    mask = clusters == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {cluster_id}")
plt.title("K-Means Clusters in PCA Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_clusters.png"), dpi=200)
plt.close()

# ---------------------------------------------------------
# 5. SUPERVISED LEARNING - RANDOM FOREST
# ---------------------------------------------------------
rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=300, random_state=SEED, class_weight="balanced"
    ))
])

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

rf_cv = cross_val_score(
    rf_pipeline, X, y, cv=5, scoring="accuracy"
)

print(f"\nRandom Forest test accuracy: {rf_accuracy:.4f}")
print(f"Random Forest 5-fold CV accuracy: {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}")
print("\nRandom Forest classification report:\n",
      classification_report(y_test, rf_pred, digits=4))

rf_cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6, 5))
plt.imshow(rf_cm)
plt.colorbar()
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(rf_cm.shape[0]):
    for j in range(rf_cm.shape[1]):
        plt.text(j, i, rf_cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "random_forest_confusion_matrix.png"), dpi=200)
plt.close()

# ---------------------------------------------------------
# 6. NEURAL NETWORK - MLP
# ---------------------------------------------------------
mlp = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=SEED
    ))
])

mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)

mlp_cv = cross_val_score(
    mlp, X, y, cv=5, scoring="accuracy"
)

print(f"\nMLP test accuracy: {mlp_accuracy:.4f}")
print(f"MLP 5-fold CV accuracy: {mlp_cv.mean():.4f} +/- {mlp_cv.std():.4f}")
print("\nMLP classification report:\n",
      classification_report(y_test, mlp_pred, digits=4))

mlp_cm = confusion_matrix(y_test, mlp_pred)
plt.figure(figsize=(6, 5))
plt.imshow(mlp_cm)
plt.colorbar()
plt.title("MLP Neural Network Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(mlp_cm.shape[0]):
    for j in range(mlp_cm.shape[1]):
        plt.text(j, i, mlp_cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mlp_confusion_matrix.png"), dpi=200)
plt.close()

# ---------------------------------------------------------
# 7. MODEL COMPARISON
# ---------------------------------------------------------
comparison = pd.DataFrame({
    "Model": ["Random Forest", "MLP Neural Network"],
    "Test Accuracy": [rf_accuracy, mlp_accuracy],
    "CV Mean Accuracy": [rf_cv.mean(), mlp_cv.mean()],
    "CV Std": [rf_cv.std(), mlp_cv.std()]
})
print("\nModel comparison:\n", comparison)
comparison.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.bar(comparison["Model"], comparison["Test Accuracy"])
plt.ylim(0, 1.05)
plt.title("Supervised Model Test Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=200)
plt.close()

# ---------------------------------------------------------
# 8. FEATURE IMPORTANCE
# ---------------------------------------------------------
rf_model = rf_pipeline.named_steps["model"]
importance = pd.Series(
    rf_model.feature_importances_, index=feature_names
).sort_values(ascending=False).head(10)

plt.figure(figsize=(9, 6))
plt.barh(importance.index[::-1], importance.values[::-1])
plt.title("Top Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=200)
plt.close()

# ---------------------------------------------------------
# 9. SAVE CLEANED DATA + SUMMARY
# ---------------------------------------------------------
output_df = df.copy()
output_df["cluster"] = clusters
output_df.to_csv(os.path.join(OUTPUT_DIR, "wine_analysis_dataset.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "results_summary.txt"), "w", encoding="utf-8") as f:
    f.write(f"Dataset shape after duplicate removal: {df.shape}\n")
    f.write(f"K-Means silhouette score: {silhouette:.4f}\n")
    f.write(f"K-Means adjusted Rand index: {ari:.4f}\n")
    f.write(f"Random Forest test accuracy: {rf_accuracy:.4f}\n")
    f.write(f"Random Forest CV mean accuracy: {rf_cv.mean():.4f}\n")
    f.write(f"MLP test accuracy: {mlp_accuracy:.4f}\n")
    f.write(f"MLP CV mean accuracy: {mlp_cv.mean():.4f}\n")

print("\nComplete. Check the outputs/ directory for figures, metrics and data.")
