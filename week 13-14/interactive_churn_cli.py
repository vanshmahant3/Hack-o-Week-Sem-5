import os
import sys
import argparse
import numpy as np
import pandas as pd

# Ensure local module visibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_dataset import prepare_dataset
from bias_variance_engine import BiasVarianceEngine
from ensemble_models import EnsembleTournament

def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Week 13-14: Interactive Streaming Subscriber Churn Predictor & Retention Engine"
    )
    parser.add_argument("--preset", choices=["loyal_binger", "disengaged_at_risk", "frustrated_churner"],
                        help="Select a benchmark subscriber persona archetype.")
    parser.add_argument("--watch", type=float, default=3.2, help="Daily watch hours (0.2 to 8.5)")
    parser.add_argument("--account_age", type=int, default=18, help="Account age in months (1 to 48)")
    parser.add_argument("--fee", type=float, default=14.99, help="Monthly subscription fee")
    parser.add_argument("--plan", choices=["Basic", "Standard", "Premium"], default="Standard", help="Subscription plan")
    parser.add_argument("--devices", type=int, default=3, help="Devices connected (1 to 6)")
    parser.add_argument("--downloads", type=int, default=6, help="Monthly offline downloads (0 to 35)")
    parser.add_argument("--skip", type=float, default=0.22, help="Skip / drop-off rate (0.05 to 0.88)")
    parser.add_argument("--tickets", type=int, default=0, help="Customer service tickets (0 to 6)")
    parser.add_argument("--login", type=int, default=2, help="Days since last active streaming session (0 to 45)")
    parser.add_argument("--payment", choices=["Credit Card", "PayPal", "Direct Debit", "Gift Card"],
                        default="Credit Card", help="Payment method")
    return parser

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = build_cli_parser()
    args = parser.parse_args()

    presets = {
        "loyal_binger": {
            "daily_watch_hours": 5.5,
            "account_age_months": 36,
            "monthly_fee": 21.99,
            "sub_plan": "Premium",
            "devices_connected": 5,
            "content_downloads_monthly": 18,
            "skip_rate": 0.12,
            "customer_service_tickets": 0,
            "last_login_days_ago": 0,
            "payment_method": "Credit Card"
        },
        "disengaged_at_risk": {
            "daily_watch_hours": 0.8,
            "account_age_months": 5,
            "monthly_fee": 14.99,
            "sub_plan": "Standard",
            "devices_connected": 2,
            "content_downloads_monthly": 1,
            "skip_rate": 0.68,
            "customer_service_tickets": 1,
            "last_login_days_ago": 21,
            "payment_method": "PayPal"
        },
        "frustrated_churner": {
            "daily_watch_hours": 2.2,
            "account_age_months": 4,
            "monthly_fee": 22.50,
            "sub_plan": "Premium",
            "devices_connected": 2,
            "content_downloads_monthly": 2,
            "skip_rate": 0.45,
            "customer_service_tickets": 5,
            "last_login_days_ago": 8,
            "payment_method": "Gift Card"
        }
    }

    if args.preset:
        profile = presets[args.preset]
        print(f"[*] Loaded Persona Preset: '{args.preset.upper()}'")
    else:
        profile = {
            "daily_watch_hours": args.watch,
            "account_age_months": args.account_age,
            "monthly_fee": args.fee,
            "sub_plan": args.plan,
            "devices_connected": args.devices,
            "content_downloads_monthly": args.downloads,
            "skip_rate": args.skip,
            "customer_service_tickets": args.tickets,
            "last_login_days_ago": args.login,
            "payment_method": args.payment
        }

    # Load and fit fast ensemble models
    csv_path = prepare_dataset()
    df = pd.read_csv(csv_path)

    X_processed, y, feature_names = BiasVarianceEngine.prepare_feature_pipeline(df)
    tournament = EnsembleTournament()
    tournament.train_and_evaluate(X_processed, y, X_processed[:200], y[:200])

    # Preprocess incoming sample
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    sample_df = pd.DataFrame([profile])
    # Build preprocessor on full df
    feature_cols = [
        "daily_watch_hours", "account_age_months", "monthly_fee",
        "sub_plan", "devices_connected", "content_downloads_monthly",
        "skip_rate", "customer_service_tickets", "last_login_days_ago",
        "payment_method"
    ]
    categorical_cols = ["sub_plan", "payment_method"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ]
    )
    preprocessor.fit(df[feature_cols])
    sample_processed = preprocessor.transform(sample_df[feature_cols])

    print("\n" + "=" * 80)
    print("  WEEK 13-14: SUBSCRIBER CHURN PREDICTION & ENSEMBLE CONSENSUS")
    print("=" * 80)
    print("Subscriber Input Attributes:")
    for k, v in profile.items():
        print(f"  {k:28s}: {v}")
    print("-" * 80)

    # Multi-model inference
    print(f"{'Ensemble Model Architecture':<32s} | {'Churn Prediction':<18s} | {'Probability':<12s}")
    print("-" * 80)

    probs = []
    for name, model in tournament.fitted_models.items():
        if "Pruned" in name:
            continue
        p = model.predict_proba(sample_processed)[0, 1]
        probs.append(p)
        status = "CHURN ALERT" if p >= 0.50 else "RETAINED SAFE"
        print(f"{name:<32s} | {status:<18s} | {p*100:5.1f}%")

    mean_prob = float(np.mean(probs))
    xgb_prob = tournament.fitted_models["XGBoost (100 Trees)"].predict_proba(sample_processed)[0, 1]
    lgb_prob = tournament.fitted_models["LightGBM (100 Trees)"].predict_proba(sample_processed)[0, 1]

    print("=" * 80)
    if mean_prob >= 0.65:
        risk_tier = "CRITICAL RISK (Immediate Churn Vulnerability)"
        action = "Automated VIP Concierge Call + 25% Retention Discount for 3 Months."
    elif mean_prob >= 0.35:
        risk_tier = "MODERATE RISK (Disengagement Developing)"
        action = "Send Personalized Push Notification with Trending Releases in Favorite Genre."
    else:
        risk_tier = "LOYAL / LOW RISK (High Engagement)"
        action = "Eligible for Annual Plan Discount Incentive & Early Access Content."

    print(f"Consensus Risk Assessment : {risk_tier}")
    print(f"Flagship XGBoost Probability: {xgb_prob*100:.1f}%")
    print(f"Flagship LightGBM Probability: {lgb_prob*100:.1f}%")
    print(f"Prescribed Retention Action: {action}")
    print("=" * 80)

if __name__ == "__main__":
    main()
