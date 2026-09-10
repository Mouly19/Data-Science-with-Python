# Week 3 – Unsupervised Learning and Clustering Analysis

## Project overview
This project completes the Week 3 internship task on **Unsupervised Learning and Clustering Analysis**. It uses the public Wine recognition dataset distributed through scikit-learn and applies K-Means clustering, with Agglomerative (hierarchical) clustering included as a comparison.

The workflow covers dataset acquisition, inspection, feature standardization, cluster-number selection, model fitting, cluster-quality evaluation, PCA visualization, hierarchical visualization, and cluster profiling.

## Dataset
- **Dataset:** Wine recognition dataset
- **Source/access method:** `sklearn.datasets.load_wine()`
- **Observations:** 178
- **Predictor features:** 13 continuous chemical measurements
- **Known target:** three wine cultivars; this target is **not used to fit the unsupervised models** and is used only for optional external validation after clustering.

## Main findings
K-Means was tested for k=2 through k=8. The silhouette score was highest at **k=3 (0.2849)** among the tested values, so three clusters were selected. The final K-Means solution produced cluster sizes of **65, 51, and 62**.

The three clusters show meaningful differences in variables such as flavanoids, total phenols, proline, color intensity, malic acid, and OD280/OD315. This supports the interpretation that the clustering is capturing distinct chemical profiles rather than arbitrary partitions.

Agglomerative clustering with Ward linkage and three clusters was also evaluated as a methodological comparison. K-Means produced stronger agreement with the dataset's known cultivar labels under ARI/NMI, although those labels were never supplied to either clustering algorithm.

## How to run

1. Create a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run:

```bash
python week3_clustering_analysis.py
```

The script creates a `week3_outputs/` directory containing:
- cluster-selection metrics
- cluster assignments
- cluster profile means
- six figures used in the report

## Files
- `week3_clustering_analysis.py` – complete Python implementation
- `requirements.txt` – Python dependencies
- `README.md` – project documentation
- `Week_3_Unsupervised_Learning_Clustering_Report.docx` – detailed report
- `figures/` – report figures
- `k_selection_metrics.csv` – K-Means selection metrics
- `wine_cluster_assignments.csv` – observation-level cluster assignments

## Reproducibility
A fixed random state of 42 is used for K-Means and PCA. K-Means uses `n_init=20` to reduce dependence on a single centroid initialization.

## Limitations
The silhouette score is moderate rather than extremely high, indicating overlap between some observations. PCA is used for visualization and does not replace the full 13-dimensional feature space used for clustering. Standardization is important because the original features have different measurement scales.

## Learning outcomes
This project demonstrates practical understanding of:
- unsupervised learning
- K-Means clustering
- hierarchical clustering
- feature scaling
- model selection with internal validation metrics
- PCA-based visualization
- cluster profiling
- responsible use of external validation labels
