# Week 6 – Integrative Capstone Project

## Project
**End-to-End Wine Classification, Clustering and Neural Network Analysis**

This project fulfills the Week 6 requirement to combine data acquisition, preprocessing, exploratory analysis, supervised learning, unsupervised learning, evaluation, insights and recommendations into one reproducible Python data-science pipeline.

## Dataset
The project uses the public **UCI Wine dataset**, loaded through `sklearn.datasets.load_wine`. It contains chemical measurements of wine samples belonging to three classes.

## Pipeline
1. Acquire the public dataset.
2. Inspect structure, descriptive statistics and missing values.
3. Remove exact duplicates.
4. Use median imputation and standardization.
5. Perform EDA using class distribution, correlation analysis and PCA.
6. Apply K-Means clustering with 3 clusters.
7. Evaluate clustering with silhouette score and adjusted Rand index.
8. Train a Random Forest classifier.
9. Validate the Random Forest using 5-fold cross-validation.
10. Train an MLP neural-network classifier.
11. Validate the MLP using 5-fold cross-validation.
12. Compare models and inspect feature importance.
13. Save figures, metrics and the processed dataset.

## Models

### K-Means
- Number of clusters: 3
- Initialization: k-means++
- `n_init=20`
- Standardized features

### Random Forest
- 300 trees
- Balanced class weights
- Fixed random state of 42
- 5-fold cross-validation

### MLP Neural Network
- Hidden layers: 64 and 32 neurons
- ReLU activation
- Adam optimizer
- Early stopping
- Validation fraction: 15%
- Maximum 500 iterations

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python week6_capstone.py
```

The script creates an `outputs/` directory.

## Outputs
- `class_distribution.png`
- `correlation_matrix.png`
- `pca_classes.png`
- `kmeans_clusters.png`
- `random_forest_confusion_matrix.png`
- `mlp_confusion_matrix.png`
- `model_comparison.png`
- `feature_importance.png`
- `model_comparison.csv`
- `wine_analysis_dataset.csv`
- `results_summary.txt`

## Reproducibility
The random seed is fixed at 42. Cross-validation and stratified train/test splitting are used for more reliable evaluation.

## Limitations
The Wine dataset is relatively small and clean compared with real-world business data. Results therefore should not be generalized to larger production datasets without additional validation. Future work could include hyperparameter optimization, more advanced clustering methods, feature-selection experiments, ensemble models, explainability and deployment.

## Submission Files
- `Week_6_Integrative_Capstone_Report.docx`
- `week6_capstone.py`
- `requirements.txt`
- `README.md`
