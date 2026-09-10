
"""
Week 4 - Supervised Learning Model Implementation
Wine Classification Project

Dataset: UCI Wine Recognition dataset, accessed through scikit-learn's built-in
load_wine() dataset. The task is multiclass classification of wine cultivars.

The script:
1. Loads and inspects the public dataset
2. Splits data into stratified train/test sets
3. Builds preprocessing pipelines
4. Trains Logistic Regression, SVM, Random Forest and KNN
5. Uses 5-fold stratified cross-validation on the training set
6. Selects the best model by mean CV accuracy
7. Evaluates the selected model on the held-out test set
8. Produces confusion matrix, classification report, ROC curves and feature importance
9. Saves metrics and predictions to CSV files
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, roc_auc_score
)

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

def main():
    # -----------------------------
    # 1. Load public dataset
    # -----------------------------
    data = load_wine(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()
    target_names = list(data.target_names)

    print("Dataset shape:", X.shape)
    print("Missing values:", int(X.isna().sum().sum()))
    print("Class distribution:")
    print(y.value_counts().sort_index())

    # -----------------------------
    # 2. Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # -----------------------------
    # 3. Models + preprocessing
    # -----------------------------
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
            ))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted_models = {}

    # -----------------------------
    # 4. Cross-validation
    # -----------------------------
    for name, model in models.items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
            n_jobs=-1
        )
        rows.append({
            "Model": name,
            "CV Accuracy Mean": scores["test_accuracy"].mean(),
            "CV Accuracy Std": scores["test_accuracy"].std(),
            "CV Precision Macro": scores["test_precision_macro"].mean(),
            "CV Recall Macro": scores["test_recall_macro"].mean(),
            "CV F1 Macro": scores["test_f1_macro"].mean()
        })
        fitted_models[name] = model.fit(X_train, y_train)

    cv_results = pd.DataFrame(rows).sort_values("CV Accuracy Mean", ascending=False)
    cv_results.to_csv(BASE_DIR / "model_comparison_cv.csv", index=False)
    print("\nCross-validation results:\n", cv_results.to_string(index=False))

    best_name = cv_results.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    print("\nSelected model:", best_name)

    # -----------------------------
    # 5. Held-out test evaluation
    # -----------------------------
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    metrics = {
        "Model": best_name,
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Test Precision Macro": precision_score(y_test, y_pred, average="macro"),
        "Test Recall Macro": recall_score(y_test, y_pred, average="macro"),
        "Test F1 Macro": f1_score(y_test, y_pred, average="macro"),
        "Test ROC-AUC OvR": roc_auc_score(y_test, y_prob, multi_class="ovr")
    }
    pd.DataFrame([metrics]).to_csv(BASE_DIR / "test_metrics.csv", index=False)

    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )
    pd.DataFrame(report).T.to_csv(BASE_DIR / "classification_report.csv")

    predictions = X_test.copy()
    predictions["Actual Class"] = y_test.values
    predictions["Predicted Class"] = y_pred
    predictions.to_csv(BASE_DIR / "test_predictions.csv", index=False)

    print("\nTest metrics:")
    for k, v in metrics.items():
        if k != "Model":
            print(f"{k}: {v:.4f}")

    # -----------------------------
    # 6. Confusion matrix
    # -----------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=target_names, cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # -----------------------------
    # 7. Multiclass ROC curves
    # -----------------------------
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, class_name in enumerate(target_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"One-vs-Rest ROC Curves - {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves.png", dpi=200)
    plt.close(fig)

    # -----------------------------
    # 8. Feature importance when available
    # -----------------------------
    if best_name == "Random Forest":
        estimator = best_model.named_steps["model"]
        importance = pd.Series(
            estimator.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        importance.to_csv(BASE_DIR / "feature_importance.csv", header=["Importance"])

        fig, ax = plt.subplots(figsize=(9, 7))
        importance.head(10).sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Top 10 Random Forest Feature Importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "feature_importance.png", dpi=200)
        plt.close(fig)

    # -----------------------------
    # 9. Save a reproducibility summary
    # -----------------------------
    summary = {
        "dataset": "Wine Recognition Dataset (UCI; scikit-learn load_wine)",
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "cv_folds": 5,
        "selected_model": best_name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(y.nunique())
    }
    pd.DataFrame([summary]).to_json(BASE_DIR / "run_summary.json", orient="records", indent=2)

if __name__ == "__main__":
    main()
