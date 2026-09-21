"""
Loan Approval Prediction System
Module: src/train_model.py
Academic Alignment: Unit 3 (Predictive Analytics, Classification, Training/Testing, Cross-Validation)

Academic Note:
Predicts loan approval eligibility (1 = Approved, 0 = Rejected / Not Approved).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from preprocessing import load_data, prepare_datasets_for_modeling


def train_and_evaluate_models(data_path, models_dir, results_dir):
    """
    Trains Logistic Regression, Random Forest, and XGBoost on Loan Approval prediction.
    Evaluates on unseen test data, runs 5-fold Stratified CV,
    saves the best model, artifacts, and comparative empirical metrics.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and preprocess data
    raw_df = load_data(data_path)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_resampled,
        y_train_resampled,
        artifacts
    ) = prepare_datasets_for_modeling(raw_df, random_state=42)
    
    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[INFO] SMOTE Resampled Train shape: {X_train_resampled.shape}")
    print(f"[INFO] Unseen Test Class distribution:\n{y_test.value_counts().to_dict()} (1: Approved, 0: Rejected)")

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    results = []
    trained_models = {}
    test_predictions = {}
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n--- Training {name} on SMOTE-balanced training data ---")
        # Train model on SMOTE-balanced training data
        model.fit(X_train_resampled, y_train_resampled)
        trained_models[name] = model
        
        # 5-Fold Stratified Cross-Validation on SMOTE training fold
        cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv_kfold, scoring='f1')
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Predict on Unseen Test Data
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate exact empirical metrics for Loan Approval (class 1) and overall
        acc = float(accuracy_score(y_test, y_pred))
        prec_app = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        rec_app = float(recall_score(y_test, y_pred, pos_label=1))
        f1_app = float(f1_score(y_test, y_pred, pos_label=1))
        
        prec_rej = float(precision_score(y_test, y_pred, pos_label=0, zero_division=0))
        rec_rej = float(recall_score(y_test, y_pred, pos_label=0))
        f1_rej = float(f1_score(y_test, y_pred, pos_label=0))
        
        roc_auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0
        cm = confusion_matrix(y_test, y_pred).tolist()

        test_predictions[name] = {
            "y_pred": y_pred.tolist(),
            "y_proba": y_proba.tolist() if y_proba is not None else None,
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"], output_dict=True)
        }

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision (Approved)": round(prec_app, 4),
            "Recall (Approved)": round(rec_app, 4),
            "F1-Score (Approved)": round(f1_app, 4),
            "F1-Score (Rejected)": round(f1_rej, 4),
            "ROC-AUC": round(roc_auc, 4),
            "CV 5-Fold F1 (Mean)": round(cv_mean, 4),
            "CV 5-Fold F1 (Std)": round(cv_std, 4)
        })

        print(f"Accuracy: {acc:.4f} | Prec (Approved): {prec_app:.4f} | Rec (Approved): {rec_app:.4f} | F1 (Approved): {f1_app:.4f} | F1 (Rejected): {f1_rej:.4f} | ROC-AUC: {roc_auc:.4f} | CV-F1: {cv_mean:.4f}")

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_path = os.path.join(results_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n[SAVED] Empirical model comparison saved to: {comparison_path}")

    # Model Selection Technical Criteria:
    # Top model selected based on composite balance of Test Accuracy, F1-Score (Approved & Rejected), and ROC-AUC
    comparison_df['Selection_Score'] = (
        0.35 * comparison_df['Accuracy'] +
        0.35 * comparison_df['F1-Score (Approved)'] +
        0.30 * comparison_df['ROC-AUC']
    )
    best_idx = comparison_df['Selection_Score'].idxmax()
    best_model_name = comparison_df.loc[best_idx, 'Model']
    best_model = trained_models[best_model_name]
    print(f"\n[SELECTION] Best Model selected based on technical criteria: {best_model_name}")

    # Save Best Model and all trained models
    joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))
    for name, m in trained_models.items():
        fname = name.lower().replace(" ", "_") + "_model.pkl"
        joblib.dump(m, os.path.join(models_dir, fname))

    # Save Preprocessing Artifacts
    joblib.dump(artifacts, os.path.join(models_dir, "preprocessing_artifacts.pkl"))

    # Save detailed evaluation metadata and test predictions
    eval_payload = {
        "best_model_name": best_model_name,
        "feature_columns": artifacts["feature_columns"],
        "test_ground_truth": y_test.tolist(),
        "predictions": test_predictions,
        "metrics_summary": results
    }
    with open(os.path.join(results_dir, "evaluation_details.json"), "w") as f:
        json.dump(eval_payload, f, indent=4)

    # Save test dataset splits for evaluation script
    joblib.dump((X_test, y_test), os.path.join(models_dir, "test_split.pkl"))
    joblib.dump((X_train, y_train), os.path.join(models_dir, "train_split.pkl"))

    print(f"[SAVED] Models & preprocessing artifacts successfully saved to: {models_dir}")
    return comparison_df, best_model_name


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "loan_data.csv")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "outputs", "results")
    
    train_and_evaluate_models(data_path, models_dir, results_dir)
