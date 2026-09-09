import os
import sys
import numpy as np
import pandas as pd

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_dataset import prepare_dataset
from missing_data_handler import MissingDataDiagnostics, MissingDataImputer
from feature_engineering import FeatureEngineer
from feature_scaling import StandardScalerCustom, MinMaxScalerCustom, RobustScalerCustom
from cross_validation import StratifiedKFoldCustom, CrossValidationEngine
from model_evaluator import ModelMetrics, ThresholdOptimizer
from visualizer import DiagnosticVisualizer

# Scikit-Learn baseline for benchmark evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def run_applicant_risk_advisor(model, feature_engineer, scaler, feature_names, optimal_threshold: float):
    """Evaluates specific credit applicants to demonstrate real-world underwriting decisions."""
    print("\n" + "=" * 85)
    print("  [STEP 6] REAL-WORLD APPLICATION: CREDIT UNDERWRITING RISK ADVISOR")
    print("=" * 85)

    applicants = [
        {
            "name": "Applicant 1: Prime Salaried Homeowner",
            "profile": {
                "annual_income": 125000.0,
                "credit_score": 780.0,
                "employment_years": 8.5,
                "loan_amount": 15000.0,
                "interest_rate": 7.5,
                "revolving_balance": 4200.0,
                "revolving_utilization": 0.12,
                "total_credit_lines": 14,
                "delinquencies_2yrs": 0,
                "home_ownership": "MORTGAGE",
                "loan_purpose": "home_improvement",
                "application_type": "INDIVIDUAL"
            },
            "desc": "$125k income, 780 FICO score, 8.5 yrs job, $15k loan at 7.5%, 12% utilization, zero delinquencies."
        },
        {
            "name": "Applicant 2: Borderline Subprime Debt-Consolidation",
            "profile": {
                "annual_income": 48000.0,
                "credit_score": 640.0,
                "employment_years": 2.0,
                "loan_amount": 18000.0,
                "interest_rate": 16.8,
                "revolving_balance": 19500.0,
                "revolving_utilization": 0.72,
                "total_credit_lines": 11,
                "delinquencies_2yrs": 1,
                "home_ownership": "RENT",
                "loan_purpose": "debt_consolidation",
                "application_type": "INDIVIDUAL"
            },
            "desc": "$48k income, 640 FICO score, 2 yrs job, $18k loan at 16.8%, 72% utilization, 1 delinquency."
        },
        {
            "name": "Applicant 3: High-Risk Distressed Borrower",
            "profile": {
                "annual_income": 26000.0,
                "credit_score": 530.0,
                "employment_years": 0.5,
                "loan_amount": 25000.0,
                "interest_rate": 24.5,
                "revolving_balance": 34000.0,
                "revolving_utilization": 0.94,
                "total_credit_lines": 22,
                "delinquencies_2yrs": 3,
                "home_ownership": "RENT",
                "loan_purpose": "small_business",
                "application_type": "INDIVIDUAL"
            },
            "desc": "$26k income, 530 FICO score, $25k loan at 24.5%, 94% utilization, 3 recent delinquencies."
        }
    ]

    for app in applicants:
        df_single = pd.DataFrame([app["profile"]])
        
        # Transform through trained feature engineer & scaler
        df_eng = feature_engineer.transform(df_single)
        X_app = scaler.transform(df_eng[feature_names].values)
        
        prob_default = float(model.predict_proba(X_app)[0, 1]) * 100.0
        
        if prob_default < 15.0:
            decision = "INSTANT APPROVAL [PRIME BORROWER]"
            action = "Offer standard interest rate with automatic disbursement."
        elif prob_default < (optimal_threshold * 100.0):
            decision = "CONDITIONAL APPROVAL [MODERATE RISK]"
            action = "Require proof of income and co-signer or 1.5% rate premium."
        else:
            decision = "APPLICATION REJECTED [HIGH DEFAULT RISK]"
            action = f"Mitigates anticipated default loss of up to ${app['profile']['loan_amount']:,.0f}."

        print(f"\n  * {app['name']}:")
        print(f"    - Financial Profile    : {app['desc']}")
        print(f"    - Predicted Risk Score : {prob_default:5.1f}% Probability of Default")
        print(f"    - Underwriting Decision: {decision}")
        print(f"    - Prescribed Action    : {action}")

def main():
    print("=" * 85)
    print("  WEEK 9: ADVANCED MODEL EVALUATION, FEATURE ENGINEERING & SCALING SUITE")
    print("  Application: Financial Credit Underwriting & Default Risk Modeling")
    print("  Concepts: Missing Data, Imputation, Scalers, Stratified CV, ROC-AUC, Precision/Recall")
    print("=" * 85)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, 'reports')
    visualizer = DiagnosticVisualizer(output_dir=reports_dir)

    # -------------------------------------------------------------------------
    # STEP 0: Ingest Raw Dataset
    # -------------------------------------------------------------------------
    print("\n[STEP 0] Ingesting Credit Underwriting Dataset (10,000 records)...")
    csv_path = prepare_dataset()
    df_raw = pd.read_csv(csv_path)

    print(f"  [+] Records Ingested         : {len(df_raw):,} applicants")
    print(f"  [+] Feature Dimensions (d)   : {len(df_raw.columns) - 1} input attributes")
    default_count = int(df_raw['is_default'].sum())
    print(f"  [+] Target Default Rate      : {len(df_raw) - default_count:,} Fully Paid ({(1 - df_raw['is_default'].mean())*100:.1f}%) | {default_count:,} Defaulters ({df_raw['is_default'].mean()*100:.1f}%)")

    # -------------------------------------------------------------------------
    # STEP 1: Missing Data Diagnostics & Imputation
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Missing Data Diagnostics & Leakage-Free Imputation...")
    missing_summary = MissingDataDiagnostics.inspect_missingness(df_raw)
    print("  --- Detected Missing Data Prevalence ---")
    print(missing_summary.to_string(index=False))

    # Stratified Train/Test Split (80/20) BEFORE any preprocessing to prevent leakage
    np.random.seed(42)
    n_samples = len(df_raw)
    y_raw = df_raw['is_default'].values
    skf_init = StratifiedKFoldCustom(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = skf_init.split(df_raw.values, y_raw)[0]

    df_train_raw = df_raw.iloc[train_idx].copy().reset_index(drop=True)
    df_test_raw = df_raw.iloc[test_idx].copy().reset_index(drop=True)

    # Fit imputer ONLY on training set
    imputer = MissingDataImputer(strategy_num='median', strategy_cat='mode', add_indicators=True)
    imputer.fit(df_train_raw)

    df_train_imputed = imputer.transform(df_train_raw)
    df_test_imputed = imputer.transform(df_test_raw)

    print(f"  [+] Training Set Nulls Remaining : {df_train_imputed.isnull().sum().sum()} null values")
    print(f"  [+] Testing Set Nulls Remaining  : {df_test_imputed.isnull().sum().sum()} null values")
    print(f"  [+] Indicator Columns Added      : {[c for c in df_train_imputed.columns if '_was_missing' in c]}")

    visualizer.plot_missing_data_diagnostics(missing_summary, df_raw, df_train_imputed)

    # -------------------------------------------------------------------------
    # STEP 2: Feature Engineering, Non-Linear Transforms & Outlier Treatment
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Domain Feature Engineering, Log Transforms & Outlier Winsorization...")
    engineer = FeatureEngineer(
        log_transform_cols=['revolving_balance', 'annual_income'],
        outlier_cols=['revolving_balance', 'annual_income']
    )
    # Fit engineer strictly on training set
    engineer.fit(df_train_imputed)

    df_train_eng = engineer.transform(df_train_imputed)
    df_test_eng = engineer.transform(df_test_imputed)

    feature_cols = [c for c in df_train_eng.columns if c != 'is_default']
    print(f"  [+] Total Engineered Features   : {len(feature_cols)} features")
    print(f"  [+] Added Interaction Features   : ['debt_to_income_ratio', 'loan_to_income_ratio', 'annual_interest_burden', 'line_utilization_intensity']")
    print(f"  [+] Log Skewness Correction      : ['log_revolving_balance', 'log_annual_income']")
    print(f"  [+] Outlier IQR Bounds Applied   : {engineer.iqr_bounds}")

    X_train_raw_np = df_train_eng[feature_cols].values
    y_train = df_train_eng['is_default'].values
    X_test_raw_np = df_test_eng[feature_cols].values
    y_test = df_test_eng['is_default'].values

    # -------------------------------------------------------------------------
    # STEP 3: Feature Scaling Showdown & Leakage Prevention
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Feature Scaling Showdown: Standard vs. MinMax vs. Robust...")
    sample_col_idx = feature_cols.index('revolving_balance')
    raw_sample = X_train_raw_np[:, sample_col_idx]

    std_scaler = StandardScalerCustom().fit(X_train_raw_np)
    minmax_scaler = MinMaxScalerCustom().fit(X_train_raw_np)
    robust_scaler = RobustScalerCustom().fit(X_train_raw_np)

    std_sample = std_scaler.transform(X_train_raw_np)[:, sample_col_idx]
    minmax_sample = minmax_scaler.transform(X_train_raw_np)[:, sample_col_idx]
    robust_sample = robust_scaler.transform(X_train_raw_np)[:, sample_col_idx]

    visualizer.plot_feature_scaling_distributions(raw_sample, std_sample, minmax_sample, robust_sample, feat_name="Revolving Balance ($)")

    # Standardize data with robust scaler for model training (handles outliers best)
    X_train_scaled = robust_scaler.transform(X_train_raw_np)
    X_test_scaled = robust_scaler.transform(X_test_raw_np)

    # -------------------------------------------------------------------------
    # STEP 4: Stratified K-Fold Cross-Validation (k=5)
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Executing Stratified 5-Fold Cross-Validation on Training Data...")
    
    # Factory to create clean model instance for each fold
    def model_factory():
        return LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    cv_summary = CrossValidationEngine.cross_validate(model_factory, X_train_scaled, y_train, n_splits=5)

    print(f"  [+] 5-Fold Stratified CV Results:")
    print(f"    * Mean Accuracy    : {cv_summary['mean_accuracy']*100:.2f}% (+/- {cv_summary['std_accuracy']*100:.2f}%)")
    print(f"    * Mean Precision   : {cv_summary['mean_precision']*100:.2f}% (+/- {cv_summary['std_precision']*100:.2f}%)")
    print(f"    * Mean Recall      : {cv_summary['mean_recall']*100:.2f}% (+/- {cv_summary['std_recall']*100:.2f}%)")
    print(f"    * Mean F1-Score    : {cv_summary['mean_f1']*100:.2f}% (+/- {cv_summary['std_f1']*100:.2f}%)")
    print(f"    * Mean ROC-AUC     : {cv_summary['mean_roc_auc']:.4f} (+/- {cv_summary['std_roc_auc']:.4f})")

    visualizer.plot_cross_validation_fold_variance(cv_summary)

    # -------------------------------------------------------------------------
    # STEP 5: Model Evaluation on Hold-Out Test Set
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Final Model Evaluation on Unseen Holdout Test Partition...")
    final_model = model_factory().fit(X_train_scaled, y_train)

    y_test_pred = final_model.predict(X_test_scaled)
    y_test_proba = final_model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    cm = ModelMetrics.confusion_matrix(y_test, y_test_pred)
    acc = ModelMetrics.accuracy_score(y_test, y_test_pred)
    prec = ModelMetrics.precision_score(y_test, y_test_pred)
    rec = ModelMetrics.recall_score(y_test, y_test_pred)
    spec = ModelMetrics.specificity_score(y_test, y_test_pred)
    f1 = ModelMetrics.f1_score(y_test, y_test_pred)
    f2 = ModelMetrics.f_beta_score(y_test, y_test_pred, beta=2.0)
    roc_auc = ModelMetrics.roc_auc_score(y_test, y_test_proba)

    print("  --- Out-of-Sample Performance Summary ---")
    print(f"  * Confusion Matrix       : TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    print(f"  * Accuracy               : {acc*100:.2f}%")
    print(f"  * Precision              : {prec*100:.2f}% (Defaulter Alert Precision)")
    print(f"  * Recall (Sensitivity)   : {rec*100:.2f}% (Defaulter Detection Coverage)")
    print(f"  * Specificity            : {spec*100:.2f}% (Good Borrower Approval Accuracy)")
    print(f"  * F1-Score               : {f1*100:.2f}% (Harmonic Mean)")
    print(f"  * F2-Score (Recall Focus): {f2*100:.2f}%")
    print(f"  * ROC-AUC Score          : {roc_auc:.4f}")

    # Economic Cost & Threshold Optimization
    cost_data = ThresholdOptimizer.compute_cost_curve(
        y_test, y_test_proba,
        profit_good_loan=1200.0,
        loss_default_loan=6500.0
    )
    print(f"\n  --- Business Economic Threshold Optimization ---")
    print(f"  * Baseline Cutoff (tau = 0.50) Net Profit   : ${cost_data['profits'][45]:,.2f}")
    print(f"  * Optimal Cutoff (tau = {cost_data['optimal_threshold']:.2f}) Net Profit    : ${cost_data['max_profit']:,.2f}")
    print(f"  * Financial Gain via Threshold Tuning      : +${cost_data['max_profit'] - cost_data['profits'][45]:,.2f}")

    visualizer.plot_confusion_matrix_and_cost_analysis(cm, cost_data)

    # ROC & Precision-Recall Curves
    fpr_arr, tpr_arr, _ = ModelMetrics.roc_curve(y_test, y_test_proba)
    p_arr, r_arr, pr_ths = ModelMetrics.precision_recall_curve(y_test, y_test_proba)
    visualizer.plot_roc_and_precision_recall_curves(fpr_arr, tpr_arr, roc_auc, p_arr, r_arr, pr_ths)

    # -------------------------------------------------------------------------
    # STEP 6: Interactive Credit Underwriting Advisor
    # -------------------------------------------------------------------------
    run_applicant_risk_advisor(final_model, engineer, robust_scaler, feature_cols, cost_data['optimal_threshold'])

    # -------------------------------------------------------------------------
    # STEP 7: Executive Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("  [STEP 7] MACHINE LEARNING PIPELINE SUMMARY (WEEK 9)")
    print("=" * 85)
    print(f"  * Dataset Analyzed               : 10,000 Credit Applications")
    print(f"  * Missing Value Columns Imputed  : {len(missing_summary)} features (Median/Mode + Indicators)")
    print(f"  * Feature Engineering Scaler     : RobustScaler (resilient to extreme debt outliers)")
    print(f"  * 5-Fold Cross-Validation ROC-AUC: {cv_summary['mean_roc_auc']:.4f} (+/- {cv_summary['std_roc_auc']:.4f})")
    print(f"  * Out-of-Sample ROC-AUC Score    : {roc_auc:.4f}")
    print(f"  * Financial Threshold Optimizer  : Shifted cutoff to {cost_data['optimal_threshold']:.2f}, saving ${cost_data['max_profit'] - cost_data['profits'][45]:,.2f}")
    print("=" * 85)
    print(f"  [+] All 5 Diagnostic Visualizations Exported to: {reports_dir}")
    print("=" * 85 + "\n")

if __name__ == '__main__':
    main()
