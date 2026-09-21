# Loan Approval Prediction System for Banking and Financial Analytics

**Silver Oak University — BCA Semester 7: Computational Analytics Capstone**  
An end-to-end predictive machine learning and underwriting analytics system for retail banking loan eligibility assessment.

---

## 📑 Table of Contents
1. [Project Overview & Problem Statement](#project-overview--problem-statement)
2. [Methodological Rigor: Why Loan Approval vs Loan Default](#methodological-rigor-why-loan-approval-vs-loan-default)
3. [Academic Alignment to Computational Analytics Syllabus](#academic-alignment-to-computational-analytics-syllabus)
4. [Dataset Description & Feature Dictionary](#dataset-description--feature-dictionary)
5. [Technology Stack](#technology-stack)
6. [System Architecture & Directory Structure](#system-architecture--directory-structure)
7. [Computational Analytics Workflow](#computational-analytics-workflow)
   - Phase 1: Data Acquisition & Dynamic Inspection
   - Phase 2: Data Quality & Cleaning Audit
   - Phase 3: Exploratory Data Analysis (EDA)
   - Phase 4: Preprocessing & Strict Data Leakage Prevention
   - Phase 5: Domain Feature Engineering
   - Phase 6: Supervised Classification & 5-Fold Stratified CV
   - Phase 7: Model Evaluation (Accuracy, F1, ROC-AUC, Confusion Matrix)
   - Phase 8: Feature Importance & Explainability
   - Phase 9: Model Selection & Artifact Serialization
   - Phase 10: Standalone Inference & Eligibility Tiers
   - Phase 11: Interactive Streamlit Dashboard
8. [Measured Empirical Results](#measured-empirical-results)
9. [Installation & Setup Guide](#installation--setup-guide)
10. [How to Run the Project](#how-to-run-the-project)
11. [Viva Voce & Technical Defense Guide](#viva-voce--technical-defense-guide)
12. [Project Limitations & Future Scope](#project-limitations--future-scope)

---

## 🏦 Project Overview & Problem Statement
In retail banking and credit analytics, credit underwriting is the frontline decision-making process determining whether a loan applicant qualifies for credit:
- **Approving creditworthy applicants** drives interest earnings and customer growth.
- **Rejecting ineligible applicants** prevents capital misallocation and preserves institutional solvency.
- **Manual underwriting** is slow, subjective, and prone to inconsistency.

This project delivers an automated, transparent **Loan Approval Prediction System**. By combining statistical data auditing, domain-justified financial ratio engineering, class-imbalance resolution via SMOTE, and supervised machine learning algorithms (**Logistic Regression**, **Random Forest**, and **XGBoost**), the system estimates the likelihood of loan approval and assigns applicants to clear eligibility tiers (**High Approval Likelihood**, **Moderate Approval Likelihood**, or **Low Approval Likelihood**) with actionable underwriting recommendations.

---

## ⚠️ Methodological Rigor: Why Loan Approval vs Loan Default

### The Target Variable: `Loan_Status`
Inspection of the actual dataset confirms that the target variable is **`Loan_Status`**, with observed values:
- **`Y` (422 records, 68.73%):** Loan Approved
- **`N` (192 records, 31.27%):** Loan Not Approved / Rejected

### Why This is NOT a Loan Default Dataset:
1. **Approval vs. Repayment:** `Loan_Status` represents an **underwriting decision made prior to disbursement**, not an observed repayment outcome.
2. **The "Defaulter" Fallacy:** An applicant whose loan was rejected (`N`) was never issued funds. A borrower who never received a loan cannot default on repayments.
3. **No Observed Default Target:** The dataset contains no post-disbursement metrics (e.g. days past due, delinquency counters, or charge-off flags).

> [!IMPORTANT]
> **Academic Integrity Statement:**  
> To maintain strict scientific and computational rigor, this project **does NOT fabricate or re-label rejection outcomes as defaults**. It models **Loan Approval Eligibility** directly from actual data. If a project specifically requires loan default prediction, a historical dataset tracking post-disbursement repayment behavior (such as LendingClub default records or credit risk sets with observed defaults) is required.

---

## 📚 Academic Alignment to Computational Analytics Syllabus

This project is explicitly mapped to the four core pedagogical units of the **Silver Oak University BCA Semester 7 Computational Analytics** syllabus:

| Unit | Syllabus Topics Covered | Project Implementation & Evidence |
| :--- | :--- | :--- |
| **Unit 1** | **Data Collection, Data Quality, Cleaning & Statistical Foundations** | Dynamic ingestion of `data/loan_data.csv` (614 rows × 13 columns), missing value quantification, duplicate audits, statistical summary (`describe()`), and Pearson correlation analysis. |
| **Unit 2** | **Data Preprocessing, Missing Values, Transformation, EDA & Visualization** | Median imputation for skewed numerical features, mode imputation for categoricals, outlier boxplots, log-income transformations, standard scaling, and 8 high-resolution Matplotlib/Seaborn figures in `outputs/figures/`. |
| **Unit 3** | **Predictive Analytics, Feature Engineering, Classification & Validation** | Derived debt ratios (`Total_Income`, `Loan_to_Income_Ratio`, `Monthly_EMI_Proxy`, `EMI_to_Income_Ratio`), **SMOTE** applied strictly on training split (no leakage), supervised classification (Logistic Regression, Random Forest, XGBoost), 5-Fold Stratified CV, Accuracy, Precision, Recall, F1-Score, and ROC-AUC. |
| **Unit 4** | **Business Intelligence, Dashboards & Real-World Financial AI** | Multi-tab interactive Streamlit dashboard (`dashboard/app.py`), feature importance ranking (Gini Impurity, Gain, Coefficients), transparent underwriting eligibility tiers, and real-time inference simulator. |

---

## 📊 Dataset Description & Feature Dictionary

### Dataset Metadata
- **Source:** Kaggle / Banking Analytics Loan Eligibility Dataset
- **Total Records:** 614 applicant entries
- **Total Attributes:** 13 variables (Demographic, Financial, Credit History, Target)
- **Target Column:** `Loan_Status` (`Y` = Approved [1], `N` = Rejected [0])

### Feature Dictionary
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Loan_ID` | String | Unique applicant identifier (dropped prior to modeling) |
| `Gender` | Categorical | Gender of primary applicant (`Male`, `Female`) |
| `Married` | Categorical | Marital status (`Yes`, `No`) |
| `Dependents` | Categorical/Numeric | Number of financial dependents (`0`, `1`, `2`, `3+` standardized to `3`) |
| `Education` | Categorical | Educational attainment (`Graduate`, `Not Graduate`) |
| `Self_Employed` | Categorical | Employment status (`Yes`, `No`) |
| `ApplicantIncome` | Numeric (Int) | Monthly income of primary applicant ($) |
| `CoapplicantIncome`| Numeric (Float) | Monthly income of secondary coapplicant ($) |
| `LoanAmount` | Numeric (Float) | Requested loan amount (in thousands) |
| `Loan_Amount_Term` | Numeric (Float) | Loan repayment period (in months, typically 360) |
| `Credit_History` | Numeric/Binary | Credit bureau guideline compliance (`1.0` = Satisfactory, `0.0` = Deficient) |
| `Property_Area` | Categorical | Geographic classification of collateral (`Urban`, `Semiurban`, `Rural`) |
| `Loan_Status` | Categorical (Target)| Underwriting eligibility outcome (`Y` = Approved [1], `N` = Rejected [0]) |

---

## 🛠️ Technology Stack
- **Programming Language:** Python 3.11+
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning & Preprocessing:** Scikit-Learn
- **Gradient Boosting:** XGBoost
- **Class Imbalance Handling:** Imbalanced-Learn (SMOTE)
- **Model Serialization:** Joblib
- **Interactive Web Dashboard:** Streamlit
- **Pedagogical Notebooks:** Jupyter Notebook, Nbconvert

---

## 📁 System Architecture & Directory Structure

```
Loan_Default_Risk_Prediction/
│
├── data/
│   └── loan_data.csv                    # Ingested dataset (614 rows × 13 columns)
│
├── notebooks/
│   └── loan_default_analysis.ipynb      # Fully executed pedagogical notebook (457 KB)
│
├── src/
│   ├── preprocessing.py                 # Ingestion, data quality, cleaning, FE, SMOTE
│   ├── eda_visualizations.py            # High-resolution figure generator
│   ├── train_model.py                   # Model training (LogReg, RF, XGBoost) & 5-Fold CV
│   ├── evaluate_model.py                # ROC curves, confusion matrices, feature importance
│   └── predict.py                       # Standalone inference with underwriting tiers
│
├── models/
│   ├── best_model.pkl                   # Winning model artifact (Random Forest)
│   ├── random_forest_model.pkl          # Individual trained Random Forest
│   ├── xgboost_model.pkl                # Individual trained XGBoost
│   ├── logistic_regression_model.pkl    # Individual trained Logistic Regression
│   ├── preprocessing_artifacts.pkl       # Scaler, feature names, imputation values
│   ├── train_split.pkl                  # Training partition
│   └── test_split.pkl                   # Unseen test partition
│
├── outputs/
│   ├── figures/                         # Generated 300-DPI visual charts
│   │   ├── approval_distribution.png
│   │   ├── income_distribution.png
│   │   ├── loan_amount_distribution.png
│   │   ├── credit_history_distribution.png
│   │   ├── approval_rate_by_categorical.png
│   │   ├── correlation_heatmap.png
│   │   ├── numerical_boxplots.png
│   │   ├── loan_to_income_approval.png
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   └── feature_importance.png
│   └── results/
│       ├── model_comparison.csv         # Measured empirical metrics table
│       └── evaluation_details.json      # Ground truth, predictions, classification reports
│
├── dashboard/
│   └── app.py                           # Full-featured Streamlit interactive web application
│
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Comprehensive technical & academic documentation
```

---

## 🔄 Computational Analytics Workflow

### Phase 1: Data Acquisition & Inspection
- Dynamic ingestion of `data/loan_data.csv`.
- Inspects dataset dimensions, data types, missing counts, and verifies target distribution.

### Phase 2: Data Quality & Cleaning Audit
- `Loan_ID` dropped (zero predictive power).
- `Dependents` standardized (`'3+'` -> `3`).
- **Median Imputation** for skewed numerical variables (`LoanAmount`, `Loan_Amount_Term`).
- **Mode Imputation** for categorical variables (`Credit_History`, `Self_Employed`, `Gender`, `Married`).
- Verified zero duplicate rows.

### Phase 3: Exploratory Data Analysis (EDA)
Generates 8 high-resolution 300-DPI charts in `outputs/figures/`:
1. **Target Distribution:** Analyzes approval outcomes (68.73% Approved vs 31.27% Rejected).
2. **Income Distributions:** Examines extreme right-skewness and validates log-transformation.
3. **Loan Amount Distribution:** Evaluates requested debt size across approval outcomes.
4. **Credit History Cross-tabulation:** Shows applicants with `Credit_History = 1.0` have an ~80% approval rate, while deficient applicants face an ~90% rejection rate.
5. **Approval Rates by Demographics:** Compares approval percentages across education, property location, and marital status.
6. **Correlation Heatmap:** Computes Pearson correlations with loan approval.
7. **Outlier Boxplots:** Highlights high-income outliers exceeding \$50,000/month.
8. **Bivariate Scatter:** Examines combined effects of Total Household Income and Loan Amount.

### Phase 4: Preprocessing & Strict Data Leakage Prevention
1. Categorical variables encoded via One-Hot Encoding.
2. **Train/Test Split First:** Stratified 80/20 split (491 train rows, 123 unseen test rows).
3. **Scaling:** `StandardScaler` fitted strictly on `X_train` and applied to `X_test`.
4. **SMOTE Imbalance Treatment:** Applied strictly to the training fold (resampling from 337:154 to 337:337). The test set remains completely untouched (85 Approved, 38 Rejected) for authentic evaluation.

### Phase 5: Domain Feature Engineering
Creates 5 banking-justified indicators:
1. `Total_Income` = $\text{ApplicantIncome} + \text{CoapplicantIncome}$
2. `Loan_to_Income_Ratio` = $\frac{\text{LoanAmount}}{\text{Total\_Income} / 1000}$
3. `Monthly_EMI_Proxy` = $\frac{\text{LoanAmount} \times 1000}{\text{Loan\_Amount\_Term}}$
4. `EMI_to_Income_Ratio` = $\frac{\text{Monthly\_EMI\_Proxy}}{\text{Total\_Income} / 12}$
5. `Income_Per_Dependent` = $\frac{\text{Total\_Income}}{\text{Dependents} + 1}$

### Phase 6: Supervised Classification & 5-Fold Stratified CV
Trains three classifiers on SMOTE-balanced training folds using `random_state=42`:
- **Logistic Regression:** Linear decision boundary.
- **Random Forest:** Ensemble bagging (150 trees, max depth 6).
- **XGBoost:** Gradient boosted decision trees with regularized objective.
- Evaluated via 5-Fold Stratified Cross-Validation to guarantee generalization.

### Phase 7: Model Evaluation (Accuracy, F1, ROC-AUC, Confusion Matrix)
Evaluates performance across both the Approved class (1) and Rejected class (0).

### Phase 8: Feature Importance & Explainability
- Gini Impurity (Random Forest) and Gain (XGBoost) confirm that `Credit_History` is the single most decisive factor, followed by `Total_Income`, `Loan_to_Income_Ratio`, and `LoanAmount`.

### Phase 9: Model Selection & Artifact Serialization
Random Forest achieved the best overall performance (Accuracy 83.74%, Approval F1 0.8837, Rejection F1 0.7297, ROC-AUC 0.8622). Serialized via Joblib into `models/`.

### Phase 10: Standalone Inference & Eligibility Tiers
Inference engine (`src/predict.py`) maps approval probabilities into transparent underwriting tiers:
- **High Approval Likelihood ($\ge 70\%$):** Recommended for immediate loan approval.
- **Moderate Approval Likelihood ($50\% - 70\%$):** Conditional approval (verify co-signer or adjust term).
- **Low Approval Likelihood ($< 50\%$):** Application rejection or substantial collateral pledge advised.

### Phase 11: Interactive Streamlit Dashboard
Multi-tab web dashboard (`dashboard/app.py`) providing data overviews, live EDA galleries, model benchmarking charts, confusion matrix heatmaps, ROC curves, and an interactive loan approval simulator.

---

## 📈 Measured Empirical Results

All metrics below are **actual empirical numbers** evaluated on the untouched 20% test partition (123 applicants). **No numbers were fabricated.**

| Machine Learning Model | Test Accuracy | Precision (Approved) | Recall (Approved) | F1-Score (Approved) | F1-Score (Rejected) | Test ROC-AUC | 5-Fold Stratified CV F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 80.49% | 85.06% | 87.06% | 0.8605 | 0.6757 | 0.8613 | 0.7260 |
| **Random Forest (Winner)** | **83.74%** | **87.36%** | **89.41%** | **0.8837** | **0.7297** | **0.8622** | 0.7680 |
| **XGBoost Classifier** | 79.67% | 87.50% | 82.35% | 0.8485 | 0.6914 | 0.8418 | **0.7743** |

---

## ⚙️ Installation & Setup Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed on Windows, macOS, or Linux.
- Terminal / PowerShell.

### 2. Clone / Open the Project Folder
```powershell
cd "c:\Users\Varsha Macwan\OneDrive\Desktop\Computational Analytics\Loan_Default_Risk_Prediction"
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### 1. Run Preprocessing & Feature Engineering
```powershell
python src/preprocessing.py
```

### 2. Generate All EDA Visualizations
```powershell
python src/eda_visualizations.py
```
*Generated plots will appear in `outputs/figures/`.*

### 3. Train All Models & Execute Cross-Validation
```powershell
python src/train_model.py
```
*Trained models will be saved in `models/` and metrics in `outputs/results/model_comparison.csv`.*

### 4. Generate Diagnostic Model Evaluation Charts
```powershell
python src/evaluate_model.py
```
*ROC curves, confusion matrices, and feature importance charts will be generated in `outputs/figures/`.*

### 5. Run Single Applicant Inference via CLI
```powershell
# Example 1: Strong Credit Profile (High Approval Likelihood)
python src/predict.py --income 6000 --coapplicant_income 2000 --loan_amount 120 --term 360 --credit_history 1.0

# Example 2: Deficient Credit Profile (Low Approval Likelihood / Rejection)
python src/predict.py --income 2500 --coapplicant_income 0 --loan_amount 200 --term 360 --credit_history 0.0
```

### 6. Start the Interactive Streamlit Dashboard
```powershell
streamlit run dashboard/app.py
```
*The web dashboard will automatically open in your default browser at `http://localhost:8501`.*

### 7. Run the Pedagogical Jupyter Notebook
```powershell
jupyter notebook notebooks/loan_default_analysis.ipynb
```

---

## 🎓 Viva Voce & Technical Defense Guide

1. **Q: What is the exact target column in your dataset, and what does it represent?**  
   *A:* The target column is `Loan_Status`, where `'Y'` represents Loan Approved (68.73%) and `'N'` represents Loan Not Approved / Rejected (31.27%). It represents an underwriting eligibility decision prior to loan disbursement.

2. **Q: Why cannot this dataset be legitimately called a Loan Default Prediction dataset?**  
   *A:* A loan default occurs after a loan has been disbursed and the borrower fails to make payments. In this dataset, applicants with `Loan_Status = 'N'` were rejected at the application phase and never received funds. Because they never held a loan, they cannot be labeled as "defaulters." Framing the project as **Loan Approval Prediction** is the methodologically correct and academically honest approach.

3. **Q: Why did you apply SMOTE only on the training set?**  
   *A:* Applying SMOTE before splitting causes **Data Leakage**, as synthetic samples would be generated from test observations, giving the model unfair foresight into test distributions. Splitting first preserves the test set as genuine, untouched real-world data.

4. **Q: Why did Random Forest outperform Logistic Regression and XGBoost?**  
   *A:* Random Forest handles non-linear interactions between financial ratios (such as loan amount relative to total income) without strict linear assumptions. Furthermore, on this 614-record dataset, Random Forest's bagged averaging provided superior variance control and achieved 83.74% Accuracy and 0.8837 Approval F1-score.

5. **Q: What was the most influential feature?**  
   *A:* Feature importance analysis demonstrated that `Credit_History` has the highest Gini Impurity reduction. Applicants with satisfactory credit compliance (`1.0`) experience an ~80% approval rate, whereas applicants with deficient history (`0.0`) face an ~90% rejection rate.

---

## 🔍 Project Limitations & Future Scope

### Limitations
1. **Sample Size:** 614 records represent a moderate sample size compared to enterprise banking portfolios.
2. **Pre-Disbursement Only:** The dataset does not track post-approval repayment schedules or charge-offs.
3. **Binary Credit History:** Credit history is recorded as a binary indicator (`0.0` or `1.0`) rather than continuous credit scores (e.g. FICO 300–850).

### Future Scope
1. **Multi-Stage Lending Pipeline:** Linking this Loan Approval engine with a post-disbursement Loan Default model trained on actual repayment transaction logs.
2. **Explainable AI (SHAP):** Providing local Shapley feature attributions to provide applicants with legally mandated adverse action reason codes.
3. **Real-time API Deployment:** Containerizing the inference pipeline using Docker and FastAPI for microservice banking integration.
