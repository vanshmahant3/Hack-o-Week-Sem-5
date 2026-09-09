import os
import sys
import numpy as np
import pandas as pd

# Add local path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_dataset import prepare_dataset
from missing_data_handler import MissingDataImputer
from feature_engineering import FeatureEngineer
from feature_scaling import RobustScalerCustom, StandardScalerCustom, MinMaxScalerCustom
from cross_validation import StratifiedKFoldCustom
from model_evaluator import ModelMetrics, ThresholdOptimizer
from sklearn.linear_model import LogisticRegression

class InteractiveCreditPipeline:
    def __init__(self):
        print("  [+] Calibrating Credit Underwriting & Risk Pipeline...")
        self.csv_path = prepare_dataset()
        self.df = pd.read_csv(self.csv_path)

        # Stratified 80/20 train/test split
        skf = StratifiedKFoldCustom(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = skf.split(self.df.values, self.df['is_default'].values)[0]
        
        self.train_df = self.df.iloc[train_idx].copy().reset_index(drop=True)
        self.test_df = self.df.iloc[test_idx].copy().reset_index(drop=True)

        # Fit preprocessing strictly on train
        self.imputer = MissingDataImputer(strategy_num='median', strategy_cat='mode', add_indicators=True)
        self.imputer.fit(self.train_df)
        train_imp = self.imputer.transform(self.train_df)
        test_imp = self.imputer.transform(self.test_df)

        self.engineer = FeatureEngineer()
        self.engineer.fit(train_imp)
        train_eng = self.engineer.transform(train_imp)
        test_eng = self.engineer.transform(test_imp)

        self.feature_names = [c for c in train_eng.columns if c != 'is_default']
        X_train_raw = train_eng[self.feature_names].values
        self.y_train = train_eng['is_default'].values
        X_test_raw = test_eng[self.feature_names].values
        self.y_test = test_eng['is_default'].values

        self.scaler = RobustScalerCustom().fit(X_train_raw)
        self.X_train = self.scaler.transform(X_train_raw)
        self.X_test = self.scaler.transform(X_test_raw)

        self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.model.fit(self.X_train, self.y_train)

        self.y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        self.cost_data = ThresholdOptimizer.compute_cost_curve(self.y_test, self.y_test_proba)
        print("  [+] Underwriting Engine Calibrated and Online!\n")

    def score_applicant(self, name: str, profile: dict):
        df_single = pd.DataFrame([profile])
        df_imp = self.imputer.transform(df_single)
        df_eng = self.engineer.transform(df_imp)
        X_app = self.scaler.transform(df_eng[self.feature_names].values)
        
        prob_default = float(self.model.predict_proba(X_app)[0, 1]) * 100.0
        opt_th = self.cost_data['optimal_threshold'] * 100.0

        if prob_default < 15.0:
            verdict = "INSTANT APPROVAL [PRIME BORROWER]"
            action = f"Approve requested loan of ${profile.get('loan_amount', 0):,.0f} at prime interest rate."
        elif prob_default < opt_th:
            verdict = "CONDITIONAL APPROVAL [MODERATE RISK]"
            action = "Approve with collateral, verified W-2 income proof, and risk surcharge (+1.75%)."
        else:
            verdict = "APPLICATION DECLINED [HIGH DEFAULT RISK]"
            action = f"Mitigates direct credit charge-off loss of up to ${profile.get('loan_amount', 0):,.0f}."

        print("\n" + "=" * 78)
        print(f"  CREDIT UNDERWRITING DECISION REPORT: {name}")
        print("=" * 78)
        print(f"  * Income: ${profile.get('annual_income', 0):,.0f} | FICO Score: {profile.get('credit_score', 0):.0f} | Loan: ${profile.get('loan_amount', 0):,.0f}")
        print(f"  * Interest Rate: {profile.get('interest_rate', 0):.1f}% | Utilization: {profile.get('revolving_utilization', 0)*100:.1f}%")
        print(f"  * Model Default Probability: {prob_default:5.1f}%")
        print(f"  * Underwriting Verdict      : {verdict}")
        print(f"  * Recommended Action        : {action}")
        print("=" * 78 + "\n")

    def simulate_threshold(self, threshold: float):
        threshold = np.clip(threshold, 0.05, 0.95)
        y_pred = (self.y_test_proba >= threshold).astype(int)

        cm = ModelMetrics.confusion_matrix(self.y_test, y_pred)
        acc = ModelMetrics.accuracy_score(self.y_test, y_pred)
        prec = ModelMetrics.precision_score(self.y_test, y_pred)
        rec = ModelMetrics.recall_score(self.y_test, y_pred)
        f1 = ModelMetrics.f1_score(self.y_test, y_pred)

        # Economic calculation
        approved = (y_pred == 0)
        approval_rate = float(np.mean(approved)) * 100.0
        good_approved = np.sum((approved) & (self.y_test == 0))
        bad_approved = np.sum((approved) & (self.y_test == 1))
        profit = (good_approved * 1200.0) - (bad_approved * 6500.0)

        print("\n" + "=" * 78)
        print(f"  DECISION THRESHOLD SIMULATOR: Cutoff tau = {threshold:.2f}")
        print("=" * 78)
        print(f"  * Portfolio Approval Rate : {approval_rate:.1f}% ({np.sum(approved):,} of {len(self.y_test):,} applicants)")
        print(f"  * Good Loans Approved     : {good_approved:,} (Yielding +${good_approved*1200:,.0f} interest revenue)")
        print(f"  * Bad Loans Approved (FN) : {bad_approved:,} (Incurring -${bad_approved*6500:,.0f} charge-off loss)")
        print(f"  * Net Portfolio Profit    : ${profit:,.2f}")
        print("-" * 78)
        print(f"  * Defaulter Recall (Catch Rate) : {rec*100:.1f}% of all actual defaults intercepted")
        print(f"  * Defaulter Precision (Accuracy): {prec*100:.1f}% of risk alerts were true defaults")
        print(f"  * F1-Score                      : {f1*100:.1f}%")
        print("=" * 78 + "\n")

def interactive_menu():
    print("=" * 78)
    print("      CREDIT UNDERWRITING & ADVANCED MODEL EVALUATION CLI")
    print("      Feature Engineering, Missing Data & Metrics Suite (Week 9)")
    print("=" * 78)

    pipeline = InteractiveCreditPipeline()

    while True:
        print("\nCHOOSE AN OPTION FOR INPUT:")
        print("  [1] Score Prime Applicant (780 FICO, $125k Income, $15k Loan)")
        print("  [2] Score Subprime Applicant (640 FICO, $48k Income, $18k Loan)")
        print("  [3] Score Distressed Applicant (530 FICO, $26k Income, $25k Loan)")
        print("  [4] ENTER CUSTOM APPLICANT PROFILE (Interactive Prompt)")
        print("  [5] SIMULATE BUSINESS DECISION THRESHOLD (Test tau from 0.05 to 0.95)")
        print("  [6] VIEW OPTIMAL ECONOMIC CUTOFF REPORT")
        print("  [0] Exit")

        try:
            choice = input("\nEnter choice [0-6]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Thank you!")
            break

        if choice == '1':
            pipeline.score_applicant("Prime Salaried Homeowner", {
                "annual_income": 125000.0, "credit_score": 780.0, "employment_years": 8.5,
                "loan_amount": 15000.0, "interest_rate": 7.5, "revolving_balance": 4200.0,
                "revolving_utilization": 0.12, "total_credit_lines": 14, "delinquencies_2yrs": 0,
                "home_ownership": "MORTGAGE", "loan_purpose": "home_improvement", "application_type": "INDIVIDUAL"
            })
        elif choice == '2':
            pipeline.score_applicant("Subprime Debt Consolidator", {
                "annual_income": 48000.0, "credit_score": 640.0, "employment_years": 2.0,
                "loan_amount": 18000.0, "interest_rate": 16.8, "revolving_balance": 19500.0,
                "revolving_utilization": 0.72, "total_credit_lines": 11, "delinquencies_2yrs": 1,
                "home_ownership": "RENT", "loan_purpose": "debt_consolidation", "application_type": "INDIVIDUAL"
            })
        elif choice == '3':
            pipeline.score_applicant("High-Risk Distressed Borrower", {
                "annual_income": 26000.0, "credit_score": 530.0, "employment_years": 0.5,
                "loan_amount": 25000.0, "interest_rate": 24.5, "revolving_balance": 34000.0,
                "revolving_utilization": 0.94, "total_credit_lines": 22, "delinquencies_2yrs": 3,
                "home_ownership": "RENT", "loan_purpose": "small_business", "application_type": "INDIVIDUAL"
            })
        elif choice == '4':
            print("\n--- Enter Custom Applicant Attributes ---")
            try:
                name = input("  Applicant Name / ID: ").strip() or "Custom Applicant"
                income = float(input("  Annual Income ($) [or press Enter for NaN]: ") or np.nan)
                score = float(input("  Credit Score (300-850) [or press Enter for NaN]: ") or np.nan)
                emp = float(input("  Employment Duration (Years) [or press Enter for NaN]: ") or np.nan)
                loan = float(input("  Loan Amount Requested ($): ").strip())
                rate = float(input("  Interest Rate (%) [e.g. 14.5]: ").strip())
                bal = float(input("  Revolving Credit Balance ($): ").strip())
                util = float(input("  Credit Card Utilization Ratio (0.0 to 1.2) [e.g. 0.45]: ").strip())
                lines = int(input("  Total Open Credit Lines: ").strip())
                delinq = int(input("  Delinquencies in Last 2 Years (0, 1, 2+): ").strip())
                home = input("  Home Ownership (MORTGAGE / RENT / OWN): ").strip().upper() or "RENT"
                purpose = input("  Loan Purpose (debt_consolidation, credit_card, etc.): ").strip() or "debt_consolidation"

                profile = {
                    "annual_income": income, "credit_score": score, "employment_years": emp,
                    "loan_amount": loan, "interest_rate": rate, "revolving_balance": bal,
                    "revolving_utilization": util, "total_credit_lines": lines,
                    "delinquencies_2yrs": delinq, "home_ownership": home,
                    "loan_purpose": purpose, "application_type": "INDIVIDUAL"
                }
                pipeline.score_applicant(name, profile)
            except Exception as e:
                print(f"\n  [!] Invalid input: {e}. Please try again.")
        elif choice == '5':
            try:
                th_input = float(input("  Enter Decision Cutoff Threshold (0.05 to 0.95, default 0.50): ").strip())
                pipeline.simulate_threshold(th_input)
            except Exception as e:
                print(f"\n  [!] Invalid threshold: {e}.")
        elif choice == '6':
            opt_th = pipeline.cost_data['optimal_threshold']
            print(f"\n  [+] Analyzing Optimal Economic Cutoff (tau = {opt_th:.2f})...")
            pipeline.simulate_threshold(opt_th)
        elif choice == '0':
            print("\nExiting. Thank you!")
            break
        else:
            print("\n  [!] Invalid selection. Please enter a number from 0 to 6.")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        p = InteractiveCreditPipeline()
        p.score_applicant("Demo Prime Borrower", {
            "annual_income": 110000.0, "credit_score": 760.0, "employment_years": 6.0,
            "loan_amount": 12000.0, "interest_rate": 8.0, "revolving_balance": 3500.0,
            "revolving_utilization": 0.15, "total_credit_lines": 12, "delinquencies_2yrs": 0,
            "home_ownership": "MORTGAGE", "loan_purpose": "home_improvement", "application_type": "INDIVIDUAL"
        })
        p.simulate_threshold(p.cost_data['optimal_threshold'])
    else:
        interactive_menu()
