"""
Loan Approval Prediction System
Module: src/eda_visualizations.py
Academic Alignment: Unit 1 (Statistical Foundations, Correlation) & Unit 2 (EDA, Outlier Detection, Visualization)

Important Academic Note:
All visualizations examine Loan Approval Status ('Y' = Approved, 'N' = Rejected / Not Approved).
There are no claims of post-disbursement default, as this dataset tracks pre-disbursement underwriting decisions.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from preprocessing import load_data, clean_data, engineer_features

# Aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})


def generate_all_eda_visualizations(data_path, output_dir):
    """
    Generates and saves exploratory figures:
    1. approval_distribution.png (and default_distribution.png for backward compat)
    2. income_distribution.png
    3. loan_amount_distribution.png
    4. credit_history_distribution.png
    5. approval_rate_by_categorical.png
    6. correlation_heatmap.png
    7. numerical_boxplots.png
    8. loan_to_income_approval.png
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_df = load_data(data_path)
    df = engineer_features(clean_data(raw_df))
    
    # Semantics: Approved vs Rejected
    df['Approval_Status'] = df['Loan_Status'].map({'Y': 'Approved', 'N': 'Rejected (Not Approved)'})
    
    # 1. Target Class Distribution (Approved vs Rejected)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['Approval_Status'].value_counts()
    colors = ['#2b5c8f', '#c0392b']
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.45, edgecolor='black', linewidth=1.2)
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f'{height} ({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, weight='bold')
    ax.set_title("Target Distribution: Loan Approval Status (Loan_Status)", pad=15, weight='bold')
    ax.set_ylabel("Applicant Count")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "approval_distribution.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "default_distribution.png"), dpi=300)  # alias
    plt.close()
    
    # 2. Income Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df['ApplicantIncome'], kde=True, ax=axes[0], color='#2b5c8f', bins=30)
    axes[0].set_title("Applicant Income Distribution (Skewed)", weight='bold')
    axes[0].set_xlabel("Applicant Income ($)")
    axes[0].set_ylabel("Frequency")
    
    sns.histplot(np.log1p(df['Total_Income']), kde=True, ax=axes[1], color='#27ae60', bins=30)
    axes[1].set_title("Log-Transformed Total Household Income", weight='bold')
    axes[1].set_xlabel("Log(Total Income)")
    axes[1].set_ylabel("Density")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "income_distribution.png"), dpi=300)
    plt.close()

    # 3. Loan Amount Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df['LoanAmount'], kde=True, ax=axes[0], color='#e67e22', bins=30)
    axes[0].set_title("Loan Amount Distribution (in Thousands)", weight='bold')
    axes[0].set_xlabel("Loan Amount ($000s)")
    axes[0].set_ylabel("Frequency")
    
    sns.boxplot(x='Approval_Status', y='LoanAmount', data=df, ax=axes[1], palette=['#2b5c8f', '#c0392b'], hue='Approval_Status', legend=False)
    axes[1].set_title("Loan Amount by Approval Status", weight='bold')
    axes[1].set_xlabel("Loan Status")
    axes[1].set_ylabel("Loan Amount ($000s)")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "loan_amount_distribution.png"), dpi=300)
    plt.close()

    # 4. Credit History Impact on Approval
    fig, ax = plt.subplots(figsize=(8, 5))
    credit_cross = pd.crosstab(df['Credit_History'], df['Approval_Status'], normalize='index') * 100
    credit_cross.plot(kind='bar', stacked=True, color=['#2b5c8f', '#c0392b'], ax=ax, edgecolor='black')
    ax.set_title("Loan Approval & Rejection Rate by Credit History Status", pad=15, weight='bold')
    ax.set_xlabel("Credit History (1 = Meets Guidelines, 0 = Deficient History)")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(['Deficient (0.0)', 'Satisfactory (1.0)'], rotation=0)
    ax.legend(title="Approval Status", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "credit_history_distribution.png"), dpi=300)
    plt.close()

    # 5. Approval Rate across Key Categoricals
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cat_vars = ['Education', 'Property_Area', 'Married', 'Self_Employed']
    for idx, var in enumerate(cat_vars):
        ax = axes[idx // 2, idx % 2]
        ct = pd.crosstab(df[var], df['Approval_Status'], normalize='index') * 100
        ct.plot(kind='bar', stacked=True, color=['#2b5c8f', '#c0392b'], ax=ax, edgecolor='black')
        ax.set_title(f"Approval Distribution by {var}", weight='bold')
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel(var)
        ax.tick_params(axis='x', rotation=15)
        if idx == 0:
            ax.legend(title="Status")
        else:
            ax.get_legend().remove()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "approval_rate_by_categorical.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "default_rate_by_categorical.png"), dpi=300)  # alias
    plt.close()

    # 6. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    df_numeric = df.copy()
    df_numeric['Loan_Approved'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    num_cols = df_numeric.select_dtypes(include=[np.number]).columns
    corr_matrix = df_numeric[num_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='Blues', cbar=True,
                linewidths=0.5, linecolor='white', ax=ax, square=True)
    ax.set_title("Pearson Correlation Heatmap (Financial Features & Loan Approval)", pad=15, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()

    # 7. Numerical Boxplots (Outlier Detection)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.boxplot(y=df['ApplicantIncome'], ax=axes[0], color='#5dade2')
    axes[0].set_title("Applicant Income Outliers", weight='bold')
    axes[0].set_ylabel("Amount ($)")
    
    sns.boxplot(y=df['CoapplicantIncome'], ax=axes[1], color='#58d68d')
    axes[1].set_title("Coapplicant Income Outliers", weight='bold')
    axes[1].set_ylabel("Amount ($)")
    
    sns.boxplot(y=df['LoanAmount'], ax=axes[2], color='#f5b041')
    axes[2].set_title("Loan Amount Outliers", weight='bold')
    axes[2].set_ylabel("Amount ($000s)")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "numerical_boxplots.png"), dpi=300)
    plt.close()

    # 8. Loan to Income Approval Scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        x='Total_Income',
        y='LoanAmount',
        hue='Approval_Status',
        style='Approval_Status',
        data=df[df['Total_Income'] < 40000],
        palette=['#2b5c8f', '#c0392b'],
        alpha=0.8,
        s=60,
        ax=ax
    )
    ax.set_title("Bivariate Analysis: Total Income vs Loan Amount by Approval Status", pad=15, weight='bold')
    ax.set_xlabel("Total Household Income ($)")
    ax.set_ylabel("Loan Amount ($000s)")
    ax.legend(title="Approval Status")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "loan_to_income_approval.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "loan_to_income_risk.png"), dpi=300)  # alias
    plt.close()
    
    print(f"All EDA figures successfully regenerated and saved to: {output_dir}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "loan_data.csv")
    fig_dir = os.path.join(base_dir, "outputs", "figures")
    generate_all_eda_visualizations(data_path, fig_dir)
