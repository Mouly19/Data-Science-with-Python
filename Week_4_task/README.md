# Week 4 - Supervised Learning Model Implementation

## Project
Multiclass classification of the Wine Recognition dataset using supervised machine learning.

## Objective
The Week 4 task requires a supervised learning problem using a public dataset, preprocessing/feature engineering, model implementation, validation, evaluation metrics, and discussion of strengths, limitations and improvements.

## Dataset
The project uses the Wine Recognition dataset available through `sklearn.datasets.load_wine()`. It contains 178 observations, 13 numeric chemical measurements, and 3 wine classes.

## Models
Four models are compared:
- Logistic Regression
- Support Vector Machine (RBF kernel)
- Random Forest
- K-Nearest Neighbors

Models that are sensitive to feature scale use `StandardScaler` inside a scikit-learn Pipeline. This prevents data leakage because scaling is learned separately inside each cross-validation training fold.

## Validation
- Stratified 80/20 train-test split
- 5-fold Stratified Cross-Validation on the training set
- Model selection by mean CV accuracy
- Final evaluation on the untouched test set

## Metrics
Accuracy, macro precision, macro recall, macro F1, multiclass one-vs-rest ROC-AUC, confusion matrix and classification report are generated.

## Run
```bash
pip install -r requirements.txt
python week4_supervised_learning.py
```

The script creates:
- `model_comparison_cv.csv`
- `test_metrics.csv`
- `classification_report.csv`
- `test_predictions.csv`
- `run_summary.json`
- figures in `figures/`

## Reproducibility
Random state is fixed at 42. The dataset is loaded from scikit-learn, so no external CSV download is required.

## Submission contents
The folder includes the Word report, complete Python source code, requirements file, README, figures and generated evaluation tables.
