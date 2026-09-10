"""
Week 3 - Unsupervised Learning and Clustering Analysis
Dataset: Wine recognition dataset (UCI Wine dataset), accessed through scikit-learn.

This script:
1. Loads the public Wine dataset.
2. Explores and validates the data.
3. Standardizes numerical features.
4. Selects K for K-Means using inertia and silhouette score.
5. Fits K-Means and Agglomerative (Ward) clustering.
6. Evaluates cluster quality.
7. Uses PCA only for 2-D visualization.
8. Profiles the resulting clusters.
9. Saves figures, metrics, and cluster assignments.

Run:
    python week3_clustering_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from scipy.cluster.hierarchy import linkage, dendrogram


RANDOM_STATE = 42
OUTPUT_DIR = Path("week3_outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    # ------------------------------------------------------------
    # 1. Load public dataset
    # ------------------------------------------------------------
    wine = load_wine(as_frame=True)
    df = wine.frame.copy()
    feature_names = list(wine.feature_names)

    # The target is retained only for descriptive/external validation.
    # It is NOT used to train the unsupervised clustering model.
    df["target_name"] = df["target"].map(
        dict(enumerate(wine.target_names))
    )

    X = df[feature_names]

    print("Dataset shape:", X.shape)
    print("\nMissing values:")
    print(X.isna().sum())
    print("\nDescriptive statistics:")
    print(X.describe().round(2))

    # ------------------------------------------------------------
    # 2. Standardize features
    # ------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------
    # 3. Select number of K-Means clusters
    # ------------------------------------------------------------
    rows = []
    k_models = {}

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20
        )
        cluster_labels = model.fit_predict(X_scaled)

        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X_scaled, cluster_labels),
            "calinski_harabasz": calinski_harabasz_score(
                X_scaled, cluster_labels
            ),
            "davies_bouldin": davies_bouldin_score(
                X_scaled, cluster_labels
            ),
        })
        k_models[k] = model

    selection = pd.DataFrame(rows)
    selection.to_csv(OUTPUT_DIR / "k_selection_metrics.csv", index=False)

    # Elbow plot
    plt.figure(figsize=(7, 5))
    plt.plot(selection["k"], selection["inertia"], marker="o")
    plt.title("K-Means Elbow Curve")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Within-cluster SSE / Inertia")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "01_elbow.png", dpi=180)
    plt.close()

    # Silhouette plot
    plt.figure(figsize=(7, 5))
    plt.plot(selection["k"], selection["silhouette"], marker="o")
    plt.title("Silhouette Score by Number of Clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "02_silhouette.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # 4. Final K-Means model
    # ------------------------------------------------------------
    # k=3 provides the highest silhouette score among the tested
    # values and gives a compact, interpretable segmentation.
    best_k = 3
    kmeans = k_models[best_k]
    kmeans_labels = kmeans.labels_

    # ------------------------------------------------------------
    # 5. Evaluate K-Means
    # ------------------------------------------------------------
    km_silhouette = silhouette_score(X_scaled, kmeans_labels)
    km_ch = calinski_harabasz_score(X_scaled, kmeans_labels)
    km_db = davies_bouldin_score(X_scaled, kmeans_labels)

    print("\nK-Means evaluation")
    print("Silhouette:", round(km_silhouette, 4))
    print("Calinski-Harabasz:", round(km_ch, 4))
    print("Davies-Bouldin:", round(km_db, 4))
    print("Cluster sizes:", np.bincount(kmeans_labels))

    # ------------------------------------------------------------
    # 6. Hierarchical clustering for comparison
    # ------------------------------------------------------------
    agglomerative = AgglomerativeClustering(
        n_clusters=best_k,
        linkage="ward"
    )
    agg_labels = agglomerative.fit_predict(X_scaled)

    agg_silhouette = silhouette_score(X_scaled, agg_labels)
    agg_ch = calinski_harabasz_score(X_scaled, agg_labels)
    agg_db = davies_bouldin_score(X_scaled, agg_labels)

    print("\nAgglomerative evaluation")
    print("Silhouette:", round(agg_silhouette, 4))
    print("Calinski-Harabasz:", round(agg_ch, 4))
    print("Davies-Bouldin:", round(agg_db, 4))
    print("Cluster sizes:", np.bincount(agg_labels))

    # ------------------------------------------------------------
    # 7. PCA for visualization only
    # ------------------------------------------------------------
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    print("\nPCA explained variance:",
          np.round(pca.explained_variance_ratio_, 4))
    print("Total explained variance:",
          round(pca.explained_variance_ratio_.sum(), 4))

    # K-Means PCA plot
    plt.figure(figsize=(8, 6))
    for cluster_id in range(best_k):
        mask = kmeans_labels == cluster_id
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.75
        )

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        marker="X",
        s=180,
        label="Centroids"
    )
    plt.title("K-Means Clusters in PCA Space")
    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)"
    )
    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "03_pca_clusters.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # 8. Cluster profiling
    # ------------------------------------------------------------
    profile = (
        df.assign(cluster=kmeans_labels)
        .groupby("cluster")[feature_names]
        .mean()
        .T
    )
    profile.to_csv(OUTPUT_DIR / "cluster_profile_means.csv")

    plt.figure(figsize=(12, 7))
    sns.heatmap(
        profile,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=profile.values.mean()
    )
    plt.title("Mean Feature Values by K-Means Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "04_cluster_profile.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # 9. Hierarchical dendrogram
    # ------------------------------------------------------------
    linkage_matrix = linkage(X_scaled, method="ward")

    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, no_labels=True, color_threshold=0)
    plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    plt.xlabel("Wine observations")
    plt.ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "05_dendrogram.png", dpi=180)
    plt.close()

    # Agglomerative PCA plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=agg_labels,
        s=55
    )
    plt.title("Agglomerative Clusters in PCA Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "06_agglomerative.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # 10. Optional external validation
    # ------------------------------------------------------------
    # The original class labels are not used to build the clusters.
    # They are used only after clustering to assess agreement with
    # known wine cultivars.
    ari_km = adjusted_rand_score(df["target"], kmeans_labels)
    nmi_km = normalized_mutual_info_score(df["target"], kmeans_labels)
    ari_agg = adjusted_rand_score(df["target"], agg_labels)
    nmi_agg = normalized_mutual_info_score(df["target"], agg_labels)

    print("\nExternal validation (labels NOT used for training)")
    print("K-Means ARI:", round(ari_km, 4))
    print("K-Means NMI:", round(nmi_km, 4))
    print("Agglomerative ARI:", round(ari_agg, 4))
    print("Agglomerative NMI:", round(nmi_agg, 4))

    # ------------------------------------------------------------
    # 11. Save assignments
    # ------------------------------------------------------------
    result = df.copy()
    result["kmeans_cluster"] = kmeans_labels
    result["agglomerative_cluster"] = agg_labels
    result.to_csv(
        OUTPUT_DIR / "wine_cluster_assignments.csv",
        index=False
    )

    print("\nAnalysis completed.")
    print("Outputs saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
