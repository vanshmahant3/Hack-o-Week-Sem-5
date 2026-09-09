import os
import numpy as np
import pandas as pd

def prepare_dataset():
    """Generates 5,000 E-Commerce & Retail Customer Behavioral Records
    structured into 5 distinct behavioral market cohorts plus realistic noise/outliers.
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'customer_segmentation.csv')

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 50000:
        print(f"  [+] Dataset already exists: {csv_path} ({os.path.getsize(csv_path) / 1024:.1f} KB)")
        return csv_path

    print("  [+] Generating Customer Behavioral & Market Segmentation Dataset (5,000 profiles)...")
    np.random.seed(42)

    n_cohort = 960  # 960 * 5 = 4,800 + 200 noise = 5,000

    # 1. VIP Whales (High Income, High Spending)
    c1_income = np.random.normal(120.0, 14.0, n_cohort)
    c1_score = np.random.normal(84.0, 7.0, n_cohort)
    c1_freq = np.random.normal(16.0, 3.5, n_cohort)
    c1_aov = np.random.normal(290.0, 45.0, n_cohort)
    c1_hours = np.random.normal(21.0, 4.0, n_cohort)
    c1_return = np.random.beta(2.0, 18.0, n_cohort)
    c1_tenure = np.random.normal(42.0, 12.0, n_cohort)
    c1_membership = np.random.choice(['Platinum', 'Gold'], size=n_cohort, p=[0.75, 0.25])

    # 2. Careful Savers (High Income, Low Spending)
    c2_income = np.random.normal(115.0, 15.0, n_cohort)
    c2_score = np.random.normal(24.0, 8.0, n_cohort)
    c2_freq = np.random.normal(3.5, 1.2, n_cohort)
    c2_aov = np.random.normal(85.0, 22.0, n_cohort)
    c2_hours = np.random.normal(6.5, 2.2, n_cohort)
    c2_return = np.random.beta(1.5, 25.0, n_cohort)
    c2_tenure = np.random.normal(36.0, 14.0, n_cohort)
    c2_membership = np.random.choice(['Silver', 'Gold'], size=n_cohort, p=[0.65, 0.35])

    # 3. Impulsive Trendsetters (Low-to-Mid Income, High Spending)
    c3_income = np.random.normal(38.0, 8.0, n_cohort)
    c3_score = np.random.normal(82.0, 8.0, n_cohort)
    c3_freq = np.random.normal(12.5, 3.0, n_cohort)
    c3_aov = np.random.normal(135.0, 28.0, n_cohort)
    c3_hours = np.random.normal(18.0, 4.5, n_cohort)
    c3_return = np.random.beta(3.0, 15.0, n_cohort)
    c3_tenure = np.random.normal(16.0, 7.0, n_cohort)
    c3_membership = np.random.choice(['Standard', 'Silver'], size=n_cohort, p=[0.70, 0.30])

    # 4. Budget Conscious (Low Income, Low Spending)
    c4_income = np.random.normal(32.0, 7.5, n_cohort)
    c4_score = np.random.normal(21.0, 7.5, n_cohort)
    c4_freq = np.random.normal(2.2, 0.9, n_cohort)
    c4_aov = np.random.normal(38.0, 12.0, n_cohort)
    c4_hours = np.random.normal(4.5, 1.8, n_cohort)
    c4_return = np.random.beta(2.0, 20.0, n_cohort)
    c4_tenure = np.random.normal(18.0, 9.0, n_cohort)
    c4_membership = np.random.choice(['Standard'], size=n_cohort, p=[1.0])

    # 5. Core Mainstream (Moderate Income, Moderate Spending)
    c5_income = np.random.normal(68.0, 11.0, n_cohort)
    c5_score = np.random.normal(52.0, 8.0, n_cohort)
    c5_freq = np.random.normal(6.8, 1.8, n_cohort)
    c5_aov = np.random.normal(110.0, 24.0, n_cohort)
    c5_hours = np.random.normal(11.0, 3.0, n_cohort)
    c5_return = np.random.beta(2.5, 18.0, n_cohort)
    c5_tenure = np.random.normal(28.0, 10.0, n_cohort)
    c5_membership = np.random.choice(['Silver', 'Standard'], size=n_cohort, p=[0.55, 0.45])

    # 6. Realistic Injected Noise / Outliers (200 records)
    n_noise = 200
    noise_income = np.random.uniform(15.0, 165.0, n_noise)
    noise_score = np.random.uniform(1.0, 99.0, n_noise)
    noise_freq = np.random.uniform(0.5, 28.0, n_noise)
    noise_aov = np.random.uniform(10.0, 480.0, n_noise)
    noise_hours = np.random.uniform(0.5, 38.0, n_noise)
    noise_return = np.random.uniform(0.01, 0.65, n_noise)
    noise_tenure = np.random.uniform(1.0, 72.0, n_noise)
    noise_membership = np.random.choice(['Standard', 'Silver', 'Gold', 'Platinum'], size=n_noise)

    # Combine cohorts
    income = np.concatenate([c1_income, c2_income, c3_income, c4_income, c5_income, noise_income])
    score = np.concatenate([c1_score, c2_score, c3_score, c4_score, c5_score, noise_score])
    freq = np.concatenate([c1_freq, c2_freq, c3_freq, c4_freq, c5_freq, noise_freq])
    aov = np.concatenate([c1_aov, c2_aov, c3_aov, c4_aov, c5_aov, noise_aov])
    hours = np.concatenate([c1_hours, c2_hours, c3_hours, c4_hours, c5_hours, noise_hours])
    ret_rate = np.concatenate([c1_return, c2_return, c3_return, c4_return, c5_return, noise_return])
    tenure = np.concatenate([c1_tenure, c2_tenure, c3_tenure, c4_tenure, c5_tenure, noise_tenure])
    membership = np.concatenate([c1_membership, c2_membership, c3_membership, c4_membership, c5_membership, noise_membership])

    # Bound and round realistically
    df = pd.DataFrame({
        'annual_income_k': np.clip(income, 14.0, 175.0).round(1),
        'spending_score': np.clip(score, 1.0, 99.0).round(1),
        'monthly_orders': np.clip(freq, 1.0, 30.0).round(1),
        'avg_order_value': np.clip(aov, 12.0, 520.0).round(2),
        'web_browsing_hours': np.clip(hours, 0.5, 42.0).round(1),
        'return_rate': np.clip(ret_rate, 0.0, 0.70).round(3),
        'tenure_months': np.clip(tenure, 1.0, 75.0).round(1),
        'membership_tier': membership
    })

    # Shuffle records
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"  [+] Saved {len(df):,} customer segmentation records to: {csv_path}")
    return csv_path

if __name__ == '__main__':
    prepare_dataset()
