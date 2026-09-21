"""
Loan Approval Prediction System
Module: src/evaluate_model.py
Academic Alignment: Unit 3 (Evaluation Metrics, Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC) & Unit 4 (Explainability)

Academic Note:
Evaluates models predicting Loan Approval (1 = Approved, 0 = Rejected).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

sns.set_theme(style="whitegrid")


def evaluate_and_visualize(models_dir, results_dir, figures_dir):
    """
    Generates evaluation charts:
    1. confusion_matrices.png
    2. roc_curves.png
    3. feature_importance.png
    """
    os.makedirs(figures_dir, exist_ok=True)

    # Load artifacts and test split
    artifacts = joblib.load(os.path.join(models_dir, "preprocessing_artifacts.pkl"))
    X_test, y_test = joblib.load(os.path.join(models_dir, "test_split.pkl"))
    feature_columns = artifacts["feature_columns"]

    # Load models
    models = {
        "Logistic Regression": joblib.load(os.path.join(models_dir, "logistic_regression_model.pkl")),
        "Random Forest": joblib.load(os.path.join(models_dir, "random_forest_model.pkl")),
        "XGBoost": joblib.load(os.path.join(models_dir, "xgboost_model.pkl")),
    }

    # 1. Confusion Matrix Multi-plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    class_labels = ["Rejected (0)", "Approved (1)"]

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels, annot_kws={"size": 13, "weight": "bold"})
        ax.set_title(f"Confusion Matrix: {name}", weight='bold', pad=10)
        ax.set_xlabel("Predicted Approval Status")
        ax.set_ylabel("Actual Approval Status")
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "confusion_matrices.png")
    fig.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[SAVED] Confusion matrices figure saved to: {cm_path}")

    # 2. Multi-Model ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Logistic Regression': '#2980b9', 'Random Forest': '#27ae60', 'XGBoost': '#e67e22'}

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", color=colors.get(name, '#333333'), lw=2.2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance Baseline')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) — Loan Approval", weight='bold', pad=12)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_path = os.path.join(figures_dir, "roc_curves.png")
    fig.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[SAVED] ROC curves figure saved to: {roc_path}")

    # 3. Feature Importance Analysis
    rf_model = models["Random Forest"]
    xgb_model = models["XGBoost"]
    lr_model = models["Logistic Regression"]

    rf_imp = pd.Series(rf_model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    lr_coef = pd.Series(np.abs(lr_model.coef_[0]), index=feature_columns).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    top_rf = rf_imp.head(10)
    sns.barplot(x=top_rf.values, y=top_rf.index, ax=axes[0], palette="Greens_r", hue=top_rf.index, legend=False)
    axes[0].set_title("Random Forest Top Features (Gini Impurity)", weight='bold')
    axes[0].set_xlabel("Feature Importance Score")

    top_xgb = xgb_imp.head(10)
    sns.barplot(x=top_xgb.values, y=top_xgb.index, ax=axes[1], palette="Oranges_r", hue=top_xgb.index, legend=False)
    axes[1].set_title("XGBoost Top Features (Gain / Weight)", weight='bold')
    axes[1].set_xlabel("Feature Importance Score")

    top_lr = lr_coef.head(10)
    sns.barplot(x=top_lr.values, y=top_lr.index, ax=axes[2], palette="Blues_r", hue=top_lr.index, legend=False)
    axes[2].set_title("Logistic Regression (Absolute Coefficients)", weight='bold')
    axes[2].set_xlabel("|Standardized Coefficient|")

    plt.tight_layout()
    feat_path = os.path.join(figures_dir, "feature_importance.png")
    fig.savefig(feat_path, dpi=300)
    plt.close()
    print(f"[SAVED] Feature importance figure saved to: {feat_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "outputs", "results")
    figures_dir = os.path.join(base_dir, "outputs", "figures")
    evaluate_and_visualize(models_dir, results_dir, figures_dir)
