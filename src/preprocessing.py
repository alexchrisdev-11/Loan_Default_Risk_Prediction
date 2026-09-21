"""
Loan Approval Prediction System
Module: src/preprocessing.py
Academic Alignment: Unit 1 (Data Quality, Cleaning) & Unit 2 (Preprocessing, Imputation, Transformation)

Important Academic Note:
The target column in this dataset is `Loan_Status`:
  - 'Y' = Loan Approved (Eligible) -> Encoded as 1
  - 'N' = Loan Not Approved / Rejected (Ineligible) -> Encoded as 0
This represents an underwriting approval decision prior to loan disbursement,
NOT a post-disbursement default/repayment event.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(filepath):
    """Loads dataset from CSV file path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")
    df = pd.read_csv(filepath)
    return df


def inspect_dataset(df):
    """
    Dynamically inspects the dataset and returns a summary dictionary.
    """
    summary = {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_counts": df.isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated().sum()),
    }
    
    # Identify target column dynamically
    candidate_targets = [c for c in df.columns if c.lower() in ['loan_status', 'approval_status', 'approved', 'status']]
    summary["target_column"] = candidate_targets[0] if candidate_targets else df.columns[-1]
    
    if summary["target_column"] in df.columns:
        summary["target_distribution"] = df[summary["target_column"]].value_counts().to_dict()
        summary["target_proportions"] = (df[summary["target_column"]].value_counts(normalize=True) * 100).to_dict()
    
    return summary


def clean_data(df):
    """
    Performs data cleaning:
    - Drops non-predictive identifier columns (e.g. Loan_ID)
    - Replaces '3+' in Dependents with 3 and converts to numeric
    - Imputes missing values using domain-appropriate statistical strategies:
      * Median for skewed numerical columns (LoanAmount, Loan_Amount_Term)
      * Mode for categorical columns (Gender, Married, Dependents, Self_Employed, Credit_History)
    """
    data = df.copy()
    
    # Drop identifier if present
    if 'Loan_ID' in data.columns:
        data = data.drop(columns=['Loan_ID'])
        
    # Standardize Dependents: replace '3+' with '3'
    if 'Dependents' in data.columns:
        data['Dependents'] = data['Dependents'].replace({'3+': '3'})
        
    # Impute categorical variables with mode
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    for col in cat_cols:
        if col in data.columns:
            mode_val = data[col].mode()[0]
            data[col] = data[col].fillna(mode_val)
            
    # Impute numerical variables with median (robust against outliers)
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    for col in num_cols:
        if col in data.columns:
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            
    # Ensure Dependents is integer
    if 'Dependents' in data.columns:
        data['Dependents'] = data['Dependents'].astype(int)
        
    # Ensure Credit_History is integer (0 or 1)
    if 'Credit_History' in data.columns:
        data['Credit_History'] = data['Credit_History'].astype(int)
        
    return data


def engineer_features(df):
    """
    Creates meaningful financial/credit features justified by banking domain:
    1. Total_Income = ApplicantIncome + CoapplicantIncome
    2. Loan_to_Income_Ratio = LoanAmount / (Total_Income / 1000 + 1e-5)
    3. Monthly_EMI_Proxy = (LoanAmount * 1000) / (Loan_Amount_Term + 1e-5)
    4. EMI_to_Income_Ratio = Monthly_EMI_Proxy / ((Total_Income / 12) + 1e-5)
    5. Income_Per_Dependent = Total_Income / (Dependents + 1)
    """
    data = df.copy()
    
    # Total household income
    data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    
    # Loan amount is in thousands in this dataset, convert scale for ratio
    total_inc_k = data['Total_Income'] / 1000.0 + 1e-5
    data['Loan_to_Income_Ratio'] = data['LoanAmount'] / total_inc_k
    
    # Monthly EMI proxy (LoanAmount * 1000 / Loan_Amount_Term)
    data['Monthly_EMI_Proxy'] = (data['LoanAmount'] * 1000.0) / (data['Loan_Amount_Term'] + 1e-5)
    
    # Ratio of EMI obligation to total monthly income
    monthly_income = (data['Total_Income'] / 12.0) + 1e-5
    data['EMI_to_Income_Ratio'] = data['Monthly_EMI_Proxy'] / monthly_income
    
    # Household income per dependent
    deps = data['Dependents'] if 'Dependents' in data.columns else 0
    data['Income_Per_Dependent'] = data['Total_Income'] / (deps + 1)
    
    return data


def prepare_datasets_for_modeling(raw_df, target_col='Loan_Status', test_size=0.2, random_state=42):
    """
    End-to-end preprocessing orchestration with strict data leakage prevention:
    1. Clean raw data
    2. Engineer domain features
    3. Encode target variable:
       - 'Y' -> 1 (Loan Approved)
       - 'N' -> 0 (Loan Not Approved / Rejected)
    4. Perform Stratified Train/Test split BEFORE any scaling or oversampling
    5. Fit encoders and scaler ONLY on training data
    6. Apply SMOTE ONLY on training split to balance Approved vs Rejected classes
    7. Return train, test splits, SMOTE-balanced train split, and preprocessing artifacts
    """
    # 1. Clean
    cleaned_df = clean_data(raw_df)
    
    # 2. Engineer features
    fe_df = engineer_features(cleaned_df)
    
    # 3. Identify and isolate target
    if target_col not in fe_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(fe_df.columns)}")
        
    y_raw = fe_df[target_col]
    X_raw = fe_df.drop(columns=[target_col])
    
    # Map target:
    # In Loan Approval prediction:
    # 1 = Loan Approved ('Y')
    # 0 = Loan Not Approved / Rejected ('N')
    if y_raw.dtype == 'O' or y_raw.dtype == 'object':
        y = y_raw.map({'Y': 1, 'N': 0, '1': 1, '0': 0}).fillna(0).astype(int)
    else:
        y = y_raw.astype(int)
        
    # Identify categorical and numerical features
    cat_columns = [col for col in X_raw.columns if X_raw[col].dtype == 'object' or col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']]
    num_columns = [col for col in X_raw.columns if col not in cat_columns]
    
    # 4. Train-Test Split FIRST (Strict Data Leakage Prevention)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Compute one-hot dummy variables fitted on train
    X_train_encoded = pd.get_dummies(X_train_raw, columns=cat_columns, drop_first=True)
    feature_columns = list(X_train_encoded.columns)
    
    # Transform test set using the exact same columns
    X_test_encoded = pd.get_dummies(X_test_raw, columns=cat_columns, drop_first=True)
    X_test_encoded = X_test_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # 5. Fit Scaler ONLY on Training data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_encoded), columns=feature_columns, index=X_train_encoded.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), columns=feature_columns, index=X_test_encoded.index)
    
    # 6. Apply SMOTE ONLY on Training data to balance minority Rejected class (0) with Approved class (1)
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    X_train_resampled = pd.DataFrame(X_train_resampled, columns=feature_columns)
    
    # 7. Package preprocessing artifacts for deployment/predict
    imputation_values = {}
    for col in cleaned_df.columns:
        if col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']:
            imputation_values[col] = cleaned_df[col].mode()[0]
        elif col != target_col:
            imputation_values[col] = cleaned_df[col].median()
            
    artifacts = {
        "feature_columns": feature_columns,
        "cat_columns": cat_columns,
        "num_columns": num_columns,
        "scaler": scaler,
        "imputation_values": imputation_values,
        "target_col": target_col,
        "class_mapping": {"Rejected (Not Approved)": 0, "Approved": 1}
    }
    
    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X_train_resampled,
        y_train_resampled,
        artifacts
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "loan_data.csv")
    df = load_data(data_path)
    summary = inspect_dataset(df)
    print("=== INSPECTION SUMMARY ===")
    print(f"Shape: {summary['num_rows']} x {summary['num_cols']}")
    print(f"Target Column: {summary['target_column']}")
    print(f"Target Distribution (Raw): {summary['target_distribution']}")
    print(f"Target Proportions (%): {summary['target_proportions']}")
    print("\nTarget Semantics: 'Y' = Loan Approved (1), 'N' = Loan Rejected / Not Approved (0)")
    
    X_tr, X_te, y_tr, y_te, X_tr_smote, y_tr_smote, art = prepare_datasets_for_modeling(df)
    print("\n=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
    print(f"Original Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"Train Class counts: {y_tr.value_counts().to_dict()} (0: Rejected, 1: Approved)")
    print(f"SMOTE Train Class counts: {pd.Series(y_tr_smote).value_counts().to_dict()}")
    print(f"Total encoded features: {len(art['feature_columns'])}")
