import os
import sys
import numpy as np
import pandas as pd

# Add local path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_dataset import prepare_dataset
from sklearn_pipelines import ScikitLearnWorkflowPipelines
from kmeans_clustering import KMeansClusteringPipeline
from cluster_profiler import ClusterProfiler

class CustomerSegmentationAdvisor:
    def __init__(self):
        print("  [+] Initializing Customer Segmentation & Scikit-Learn Workflow Engine...")
        self.csv_path = prepare_dataset()
        self.df_raw = pd.read_csv(self.csv_path)

        self.numerical_cols = [
            'annual_income_k', 'spending_score', 'monthly_orders',
            'avg_order_value', 'web_browsing_hours', 'return_rate', 'tenure_months'
        ]
        self.categorical_cols = ['membership_tier']

        # Fit Scikit-Learn Preprocessing Pipeline
        self.preprocessor = ScikitLearnWorkflowPipelines.build_preprocessing_pipeline(
            numerical_cols=self.numerical_cols,
            categorical_cols=self.categorical_cols,
            scaler_type='robust'
        )
        self.X_scaled = self.preprocessor.fit_transform(self.df_raw)

        # Fit KMeans Model
        self.kmeans = KMeansClusteringPipeline(random_state=42)
        self.labels, self.centers = self.kmeans.fit_predict(self.X_scaled, k=5)

        # Profile clusters
        self.profile_df = ClusterProfiler.profile_clusters(self.df_raw, self.labels)
        self.cluster_map = {row['Cluster']: row['Persona_Name'] for _, row in self.profile_df.iterrows()}
        print("  [+] Customer Segmentation Engine Online & Ready!\n")

    def score_customer(self, name: str, profile: dict):
        df_single = pd.DataFrame([profile])
        X_single = self.preprocessor.transform(df_single)
        cluster_id = int(self.kmeans.predict(X_single)[0])
        persona_name = self.cluster_map.get(cluster_id, "Core Mainstream Customer")
        action = ClusterProfiler.get_marketing_action(persona_name)

        print("\n" + "=" * 80)
        print(f"  CUSTOMER PERSONA & SEGMENTATION REPORT: {name}")
        print("=" * 80)
        print(f"  * Profile Demographics     : Income: ${profile.get('annual_income_k', 0):,.1f}k | Spend Score: {profile.get('spending_score', 0):.0f}/100")
        print(f"                               Orders: {profile.get('monthly_orders', 0):.1f}/mo | AOV: ${profile.get('avg_order_value', 0):.2f} | Web: {profile.get('web_browsing_hours', 0):.1f} hrs")
        print(f"                               Membership: {profile.get('membership_tier', 'Standard')} | Tenure: {profile.get('tenure_months', 0):.0f} mos")
        print("-" * 80)
        print(f"  * Assigned Cluster Cohort  : Cluster #{cluster_id}")
        print(f"  * Behavioral Persona       : {persona_name}")
        print(f"  * Strategic Marketing CRM  : {action}")
        print("=" * 80 + "\n")

PRESET_CUSTOMERS = [
    {
        "name": "Sarah Jenkins (Tech Director)",
        "profile": {"annual_income_k": 135.0, "spending_score": 88.0, "monthly_orders": 18.0, "avg_order_value": 320.0, "web_browsing_hours": 22.0, "return_rate": 0.08, "tenure_months": 45.0, "membership_tier": "Platinum"}
    },
    {
        "name": "Marcus Vance (Senior Accountant)",
        "profile": {"annual_income_k": 118.0, "spending_score": 22.0, "monthly_orders": 3.0, "avg_order_value": 85.0, "web_browsing_hours": 6.0, "return_rate": 0.04, "tenure_months": 38.0, "membership_tier": "Gold"}
    },
    {
        "name": "Chloe Adams (Fashion Influencer)",
        "profile": {"annual_income_k": 40.0, "spending_score": 85.0, "monthly_orders": 14.0, "avg_order_value": 140.0, "web_browsing_hours": 19.0, "return_rate": 0.16, "tenure_months": 15.0, "membership_tier": "Silver"}
    },
    {
        "name": "Liam Patel (University Student)",
        "profile": {"annual_income_k": 28.0, "spending_score": 19.0, "monthly_orders": 2.0, "avg_order_value": 35.0, "web_browsing_hours": 4.0, "return_rate": 0.07, "tenure_months": 12.0, "membership_tier": "Standard"}
    }
]

def interactive_menu():
    print("=" * 80)
    print("      CUSTOMER BEHAVIORAL SEGMENTATION & PERSONA ADVISOR CLI")
    print("      Scikit-Learn Workflow & Unsupervised Clustering Suite (Week 10)")
    print("=" * 80)

    advisor = CustomerSegmentationAdvisor()

    while True:
        print("\nCHOOSE AN OPTION FOR INPUT:")
        print("  [1] Classify Sarah Jenkins (High Income $135k, High Spend 88/100, Platinum)")
        print("  [2] Classify Marcus Vance (High Income $118k, Low Spend 22/100, Gold)")
        print("  [3] Classify Chloe Adams (Low Income $40k, High Spend 85/100, Silver)")
        print("  [4] Classify Liam Patel (Low Income $28k, Low Spend 19/100, Standard)")
        print("  [5] ENTER CUSTOM CUSTOMER ATTRIBUTES (Interactive Prompt)")
        print("  [6] EVALUATE ALL PRESET PERSONAS TOGETHER")
        print("  [0] Exit")

        try:
            choice = input("\nEnter choice [0-6]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Thank you!")
            break

        if choice in ['1', '2', '3', '4']:
            idx = int(choice) - 1
            cust = PRESET_CUSTOMERS[idx]
            advisor.score_customer(cust["name"], cust["profile"])
        elif choice == '5':
            print("\n--- Enter Custom Customer Behavioral Attributes ---")
            try:
                name = input("  Customer Name / ID: ").strip() or "Custom Customer"
                inc = float(input("  Annual Income ($k, e.g. 75.0): ").strip())
                score = float(input("  Spending Score (1 to 100, e.g. 55.0): ").strip())
                orders = float(input("  Monthly Order Count (e.g. 7.5): ").strip())
                aov = float(input("  Average Order Value ($ e.g. 120.0): ").strip())
                hours = float(input("  Web Browsing Hours per Month (e.g. 11.5): ").strip())
                ret = float(input("  Return Rate (0.0 to 0.5, e.g. 0.08): ").strip())
                tenure = float(input("  Tenure in Months (e.g. 24.0): ").strip())
                tier = input("  Membership Tier (Standard / Silver / Gold / Platinum): ").strip().capitalize() or "Standard"

                profile = {
                    "annual_income_k": inc, "spending_score": score, "monthly_orders": orders,
                    "avg_order_value": aov, "web_browsing_hours": hours, "return_rate": ret,
                    "tenure_months": tenure, "membership_tier": tier
                }
                advisor.score_customer(name, profile)
            except Exception as e:
                print(f"\n  [!] Invalid input: {e}. Please try again.")
        elif choice == '6':
            print("\n--- Evaluating All Preset Personas ---")
            for cust in PRESET_CUSTOMERS:
                advisor.score_customer(cust["name"], cust["profile"])
        elif choice == '0':
            print("\nExiting. Thank you!")
            break
        else:
            print("\n  [!] Invalid selection. Please enter a number from 0 to 6.")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        adv = CustomerSegmentationAdvisor()
        for c in PRESET_CUSTOMERS[:2]:
            adv.score_customer(c["name"], c["profile"])
    else:
        interactive_menu()
