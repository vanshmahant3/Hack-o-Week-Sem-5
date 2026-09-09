import numpy as np
import pandas as pd
from typing import Dict, List, Any

class ClusterProfiler:
    """Interprets unsupervised cluster partitions, creates behavioral customer personas,
    and maps actionable marketing and CRM strategies.
    """
    @staticmethod
    def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Aggregates demographic and behavioral means across customer clusters."""
        df_copy = df.copy()
        df_copy['Cluster'] = labels

        feature_cols = [
            'annual_income_k', 'spending_score', 'monthly_orders',
            'avg_order_value', 'web_browsing_hours', 'return_rate', 'tenure_months'
        ]

        # Group by cluster and compute means
        grouped = df_copy.groupby('Cluster')[feature_cols].mean().round(1)
        grouped['Customer_Count'] = df_copy.groupby('Cluster').size()
        grouped['Percentage'] = ((grouped['Customer_Count'] / len(df_copy)) * 100.0).round(1)

        # Assign intuitive persona names based on income and spend patterns
        persona_names = []
        for cluster_id, row in grouped.iterrows():
            if cluster_id == -1:
                persona_names.append("Anomalous Outliers / High-Risk Noise")
            elif row['annual_income_k'] > 95 and row['spending_score'] > 65:
                persona_names.append("VIP Whales (High Income, High Spend)")
            elif row['annual_income_k'] > 95 and row['spending_score'] <= 65:
                persona_names.append("Careful Savers (High Income, Low Spend)")
            elif row['annual_income_k'] <= 60 and row['spending_score'] > 65:
                persona_names.append("Impulsive Trendsetters (Low Income, High Spend)")
            elif row['annual_income_k'] <= 60 and row['spending_score'] <= 65:
                persona_names.append("Budget Seekers (Low Income, Low Spend)")
            else:
                persona_names.append("Core Mainstream (Moderate Income & Spend)")

        grouped['Persona_Name'] = persona_names
        return grouped.reset_index()

    @staticmethod
    def get_marketing_action(persona_name: str) -> str:
        """Returns tailored business action recommendations for each persona."""
        actions = {
            "VIP Whales (High Income, High Spend)": "Concierge loyalty program, luxury early-access releases, zero-friction delivery.",
            "Careful Savers (High Income, Low Spend)": "Value proposition messaging, long-term durability proof, premium trial bundles.",
            "Impulsive Trendsetters (Low Income, High Spend)": "Flash sales, dynamic mobile push promotions, buy-now-pay-later (BNPL) options.",
            "Budget Seekers (Low Income, Low Spend)": "Clearance discount vouchers, bundle savings, free shipping thresholds.",
            "Core Mainstream (Moderate Income & Spend)": "Evergreen catalog recommendations, seasonal sales, subscription membership upgrades.",
            "Anomalous Outliers / High-Risk Noise": "Security / fraud risk inspection, automated bot throttling, manual KYC check."
        }
        for key, act in actions.items():
            if key in persona_name:
                return act
        return "Standard personalized email marketing."
