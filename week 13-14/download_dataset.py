import os
import numpy as np
import pandas as pd
from typing import Tuple

def generate_streaming_churn_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generates a realistic, intuitive Streaming Service Subscriber Churn dataset.

    Simulates 2,000 subscriber accounts with continuous behavioral attributes,
    discrete account configurations, and ground-truth non-linear churn interactions.

    Target:
    - churned: 1 if subscriber canceled subscription, 0 if active/retained.
    """
    np.random.seed(random_state)

    # 1. Continuous & Discrete Feature Synthesis
    account_age_months = np.random.exponential(scale=14.0, size=n_samples) + 1.0
    account_age_months = np.clip(np.round(account_age_months), 1, 48).astype(int)

    # Subscription plan distribution
    plan_choices = ["Basic", "Standard", "Premium"]
    plan_probs = [0.35, 0.45, 0.20]
    sub_plan = np.random.choice(plan_choices, size=n_samples, p=plan_probs)

    base_fees = {"Basic": 8.99, "Standard": 14.99, "Premium": 21.99}
    monthly_fee = np.array([base_fees[p] + np.random.uniform(-0.5, 0.5) for p in sub_plan])
    monthly_fee = np.round(monthly_fee, 2)

    # Connected devices depends loosely on plan
    device_means = {"Basic": 1.4, "Standard": 2.6, "Premium": 4.1}
    devices_connected = np.array([
        np.clip(int(np.round(np.random.normal(device_means[p], 0.8))), 1, 6)
        for p in sub_plan
    ])

    # Daily watch hours (inversely correlated with account abandonment)
    daily_watch_hours = np.random.gamma(shape=2.5, scale=1.2, size=n_samples)
    daily_watch_hours = np.clip(np.round(daily_watch_hours, 2), 0.2, 8.5)

    # Content downloads per month
    content_downloads_monthly = np.random.poisson(lam=daily_watch_hours * 2.2)
    content_downloads_monthly = np.clip(content_downloads_monthly, 0, 35)

    # Skip rate (proportion of started content abandoned within 5 mins)
    skip_rate = np.random.beta(a=2.0, b=5.0, size=n_samples)
    skip_rate = np.clip(np.round(skip_rate, 3), 0.05, 0.88)

    # Customer service tickets (0 to 6)
    ticket_probs = [0.50, 0.25, 0.12, 0.07, 0.04, 0.015, 0.005]
    customer_service_tickets = np.random.choice(len(ticket_probs), size=n_samples, p=ticket_probs)

    # Inactivity: Days since last active streaming session (0 to 45)
    last_login_days_ago = np.random.exponential(scale=5.0, size=n_samples)
    last_login_days_ago = np.clip(np.round(last_login_days_ago), 0, 45).astype(int)

    # Payment methods
    payment_choices = ["Credit Card", "PayPal", "Direct Debit", "Gift Card"]
    payment_probs = [0.48, 0.32, 0.14, 0.06]
    payment_method = np.random.choice(payment_choices, size=n_samples, p=payment_probs)

    # 2. Non-linear Ground-Truth Churn Logit
    # Interaction 1: Disengagement risk (low watch time + high absence + high skip rate)
    disengagement_score = (
        1.8 * (last_login_days_ago > 12).astype(float)
        + 1.4 * np.maximum(0, 2.5 - daily_watch_hours)
        + 2.0 * (skip_rate > 0.45).astype(float)
    )

    # Interaction 2: Friction / Pricing dissatisfaction (high tickets + expensive plan)
    friction_score = (
        2.2 * (customer_service_tickets >= 3).astype(float) * (monthly_fee / 15.0)
        + 0.8 * (payment_method == "Gift Card").astype(float)  # transient users
    )

    # Interaction 3: Retention anchors (tenure, multi-device lock-in, downloads)
    loyalty_anchor = (
        1.5 * np.log1p(account_age_months / 6.0)
        + 1.1 * (devices_connected >= 3).astype(float)
        + 0.8 * (content_downloads_monthly > 8).astype(float)
    )

    # Latent logit
    base_logit = -0.4  # baseline retention (~26% churn rate)
    logit = base_logit + disengagement_score + friction_score - loyalty_anchor

    # Convert to probability via sigmoid
    prob_churn = 1.0 / (1.0 + np.exp(-logit))

    # Binary outcome with calibrated threshold
    churned = (prob_churn > 0.50).astype(int)

    # Inject 4% irreducible label noise to simulate real-world stochasticity and test overfitting
    noise_mask = np.random.rand(n_samples) < 0.04
    churned[noise_mask] = 1 - churned[noise_mask]

    # Create DataFrame
    subscriber_ids = [f"SUB_{10001 + i}" for i in range(n_samples)]
    df = pd.DataFrame({
        "subscriber_id": subscriber_ids,
        "daily_watch_hours": daily_watch_hours,
        "account_age_months": account_age_months,
        "monthly_fee": monthly_fee,
        "sub_plan": sub_plan,
        "devices_connected": devices_connected,
        "content_downloads_monthly": content_downloads_monthly,
        "skip_rate": skip_rate,
        "customer_service_tickets": customer_service_tickets,
        "last_login_days_ago": last_login_days_ago,
        "payment_method": payment_method,
        "churned": churned
    })

    # Shuffle DataFrame
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

def prepare_dataset() -> str:
    """Acquires, caches, and returns the path to the Streaming Churn CSV."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "streaming_service_churn.csv")

    if not os.path.exists(csv_path):
        print("[*] Generating Streaming Service Subscriber Churn Dataset (2,000 records)...")
        df = generate_streaming_churn_dataset(n_samples=2000, random_state=42)
        df.to_csv(csv_path, index=False)
        print(f"[+] Dataset saved to: {csv_path}")
    else:
        print(f"[+] Dataset already cached: {csv_path}")
        df = pd.read_csv(csv_path)

    churn_rate = df["churned"].mean() * 100
    print(f"    - Records: {len(df):,}")
    print(f"    - Features: {df.shape[1] - 2} predictors (Continuous & Categorical)")
    print(f"    - Churn Rate: {churn_rate:.1f}% ({df['churned'].sum():,} churned vs {(1-df['churned']).sum():,} retained)")
    return csv_path

if __name__ == "__main__":
    prepare_dataset()
