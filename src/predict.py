"""
Loan Approval Prediction System
Module: src/predict.py
Academic Alignment: Unit 3 (Predictive Inference) & Unit 4 (Underwriting Decision Support & Eligibility Tiers)

Academic Note:
This prediction engine estimates the likelihood of Loan Approval (1 = Approved, 0 = Rejected).
It does not predict post-disbursement loan default, as the dataset tracks credit approval decisions.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

# Safe import for local or package execution
try:
    from preprocessing import clean_data, engineer_features
except ImportError:
    from src.preprocessing import clean_data, engineer_features


def load_inference_artifacts(models_dir=None):
    """Loads the trained winning model and preprocessing artifacts."""
    if models_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models")
        
    model_path = os.path.join(models_dir, "best_model.pkl")
    artifacts_path = os.path.join(models_dir, "preprocessing_artifacts.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Model or preprocessing artifacts not found in {models_dir}. Please run src/train_model.py first.")
        
    model = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)
    return model, artifacts


def predict_loan_approval(applicant_data: dict, model=None, artifacts=None):
    """
    Predicts loan approval eligibility for a loan applicant.
    
    Parameters:
    - applicant_data: dictionary containing applicant fields:
      'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
      'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
      'Credit_History', 'Property_Area'
    - model: pre-loaded sklearn/xgboost model (optional)
    - artifacts: pre-loaded artifacts dictionary (optional)
    
    Returns:
    - Dictionary with Approval Probability, Rejection Probability,
      Predicted Class, Eligibility Category, and Underwriting Advisory.
    """
    if model is None or artifacts is None:
        model, artifacts = load_inference_artifacts()
        
    feature_columns = artifacts["feature_columns"]
    scaler = artifacts["scaler"]
    imputation_vals = artifacts["imputation_values"]
    
    # Convert input to DataFrame
    df = pd.DataFrame([applicant_data])
    
    # Fill any missing keys with training medians/modes
    for col, default_val in imputation_vals.items():
        if col not in df.columns or pd.isna(df[col].iloc[0]):
            df[col] = default_val
            
    # Clean and standardize
    df = clean_data(df)
    
    # Engineer domain features
    df_fe = engineer_features(df)
    
    # Explicit deterministic one-hot dummy generation (avoids single-row drop_first dropping bug)
    df_fe['Gender_Male'] = (df_fe['Gender'] == 'Male').astype(int)
    df_fe['Married_Yes'] = (df_fe['Married'] == 'Yes').astype(int)
    df_fe['Education_Not Graduate'] = (df_fe['Education'] == 'Not Graduate').astype(int)
    df_fe['Self_Employed_Yes'] = (df_fe['Self_Employed'] == 'Yes').astype(int)
    df_fe['Property_Area_Semiurban'] = (df_fe['Property_Area'] == 'Semiurban').astype(int)
    df_fe['Property_Area_Urban'] = (df_fe['Property_Area'] == 'Urban').astype(int)
    
    # Reindex exactly to the trained feature columns
    df_encoded = df_fe[feature_columns]
    
    # Scale numerical features using saved StandardScaler
    df_scaled = pd.DataFrame(scaler.transform(df_encoded), columns=feature_columns)
    
    # Predict probabilities (class 1 is Approved, class 0 is Rejected)
    probabilities = model.predict_proba(df_scaled)[0]
    rejection_prob = float(probabilities[0])
    approval_prob = float(probabilities[1])
    predicted_class = int(model.predict(df_scaled)[0])
    
    # Transparent Eligibility Categorization Thresholds:
    # - High Approval Likelihood: Approval probability >= 70%
    # - Moderate Approval Likelihood: Approval probability between 50% and 70%
    # - Low Approval Likelihood (High Rejection Likelihood): Approval probability < 50%
    if approval_prob >= 0.70:
        eligibility_tier = "High Approval Likelihood"
        recommendation = "Applicant strongly satisfies credit guidelines. Recommended for standard loan approval."
    elif approval_prob >= 0.50:
        eligibility_tier = "Moderate Approval Likelihood"
        recommendation = "Applicant meets baseline threshold. Conditional approval advised (verify co-signer or adjust term)."
    else:
        eligibility_tier = "Low Approval Likelihood"
        recommendation = "Applicant does not satisfy underwriting criteria. Application rejection or collateral pledge advised."

    predicted_label = "Loan Approved" if predicted_class == 1 else "Loan Rejected (Not Approved)"

    return {
        "approval_probability_pct": round(approval_prob * 100, 2),
        "rejection_probability_pct": round(rejection_prob * 100, 2),
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "eligibility_tier": eligibility_tier,
        "recommendation": recommendation,
        "notice": "Disclaimer: Prediction indicates statistical likelihood of underwriting approval based on historical patterns, not a binding commitment."
    }


# Backwards compatibility alias
predict_applicant_risk = predict_loan_approval


def format_approval_report(assessment: dict):
    """Formats assessment into a clean terminal report."""
    report = f"""
=====================================================
         Loan Approval Eligibility Report
=====================================================
Approval Probability:      {assessment['approval_probability_pct']}%
Rejection Probability:     {assessment['rejection_probability_pct']}%
Predicted Outcome:         {assessment['predicted_label']}
Eligibility Category:      {assessment['eligibility_tier']}
-----------------------------------------------------
Underwriting Recommendation:
{assessment['recommendation']}
-----------------------------------------------------
{assessment['notice']}
=====================================================
"""
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Loan Approval Eligibility for a Loan Applicant")
    parser.add_argument("--gender", type=str, default="Male", choices=["Male", "Female"])
    parser.add_argument("--married", type=str, default="Yes", choices=["Yes", "No"])
    parser.add_argument("--dependents", type=str, default="0", choices=["0", "1", "2", "3+"])
    parser.add_argument("--education", type=str, default="Graduate", choices=["Graduate", "Not Graduate"])
    parser.add_argument("--self_employed", type=str, default="No", choices=["Yes", "No"])
    parser.add_argument("--income", type=float, default=6000.0, help="Monthly applicant income in $")
    parser.add_argument("--coapplicant_income", type=float, default=2000.0, help="Monthly coapplicant income in $")
    parser.add_argument("--loan_amount", type=float, default=120.0, help="Loan amount in thousands ($000s)")
    parser.add_argument("--term", type=float, default=360.0, help="Loan term in months (e.g. 360)")
    parser.add_argument("--credit_history", type=float, default=1.0, choices=[1.0, 0.0], help="1.0 for meets guidelines, 0.0 for deficient")
    parser.add_argument("--property_area", type=str, default="Semiurban", choices=["Urban", "Semiurban", "Rural"])
    
    args = parser.parse_args()
    
    applicant = {
        "Gender": args.gender,
        "Married": args.married,
        "Dependents": args.dependents,
        "Education": args.education,
        "Self_Employed": args.self_employed,
        "ApplicantIncome": args.income,
        "CoapplicantIncome": args.coapplicant_income,
        "LoanAmount": args.loan_amount,
        "Loan_Amount_Term": args.term,
        "Credit_History": args.credit_history,
        "Property_Area": args.property_area
    }
    
    result = predict_loan_approval(applicant)
    print(format_approval_report(result))
