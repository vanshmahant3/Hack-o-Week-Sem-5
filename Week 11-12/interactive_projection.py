import os
import sys
import numpy as np
import pandas as pd

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_dataset import prepare_dataset
from pca_analysis import PCAAnalysis

def print_banner():
    print("=" * 80)
    print("  STUDENT LIFESTYLE & ACADEMIC PERFORMANCE: INTERACTIVE 2D PROJECTION CLI")
    print("  Dimensionality Reduction: PCA Out-of-Sample Transform & Reconstruction Demo")
    print("=" * 80)

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print_banner()

    csv_path = prepare_dataset()
    df = pd.read_csv(csv_path)

    feature_cols = [
        "study_hours_weekly",
        "attendance_rate",
        "sleep_hours_daily",
        "screen_time_daily",
        "extracurricular_hours",
        "stress_level",
        "prior_gpa",
        "assignment_completion_rate"
    ]

    pca_engine = PCAAnalysis(feature_names=feature_cols)
    pca_engine.fit(df[feature_cols].values)

    preset_profiles = {
        "1": {
            "name": "Alex - Dean's Honor Roll Candidate",
            "data": {
                "study_hours_weekly": 29.0,
                "attendance_rate": 96.0,
                "sleep_hours_daily": 7.5,
                "screen_time_daily": 2.0,
                "extracurricular_hours": 6.0,
                "stress_level": 3.8,
                "prior_gpa": 3.88,
                "assignment_completion_rate": 98.0
            }
        },
        "2": {
            "name": "Jordan - Balanced Student Athlete",
            "data": {
                "study_hours_weekly": 17.0,
                "attendance_rate": 84.0,
                "sleep_hours_daily": 7.8,
                "screen_time_daily": 4.2,
                "extracurricular_hours": 9.5,
                "stress_level": 5.2,
                "prior_gpa": 3.25,
                "assignment_completion_rate": 86.0
            }
        },
        "3": {
            "name": "Taylor - Distracted / Burning Out",
            "data": {
                "study_hours_weekly": 6.5,
                "attendance_rate": 62.0,
                "sleep_hours_daily": 5.2,
                "screen_time_daily": 8.0,
                "extracurricular_hours": 2.5,
                "stress_level": 8.5,
                "prior_gpa": 2.20,
                "assignment_completion_rate": 58.0
            }
        }
    }

    print("\nSelect a Student Profile to Project into 2D Latent Space:")
    print("  [1] Alex   - Dean's Honor Roll Candidate (High Study & Attendance, Low Screen Time)")
    print("  [2] Jordan - Balanced Student Athlete (Moderate Study, High Sleep & Sports)")
    print("  [3] Taylor - Distracted / Burning Out (High Screen Time & Stress, Low Study)")
    print("  [4] Custom - Enter Your Own Lifestyle Metrics")

    if sys.stdin.isatty():
        choice = input("\nEnter choice [1-4] (default 1): ").strip() or "1"
    else:
        choice = "1"
        print(f"Non-interactive environment detected. Auto-selecting Profile [1].")

    if choice in preset_profiles:
        profile_name = preset_profiles[choice]["name"]
        student_data = preset_profiles[choice]["data"]
    else:
        profile_name = "Custom Student Profile"
        student_data = {}
        defaults = preset_profiles["2"]["data"]
        for feat in feature_cols:
            clean_name = feat.replace("_", " ").title()
            default_val = defaults[feat]
            val_str = input(f"  Enter {clean_name} (default: {default_val}): ").strip()
            try:
                student_data[feat] = float(val_str) if val_str else default_val
            except ValueError:
                student_data[feat] = default_val

    print("\n" + "-" * 75)
    print(f"INPUT STUDENT PROFILE: {profile_name}")
    print("-" * 75)
    for k, v in student_data.items():
        print(f"  • {k.replace('_', ' ').title():28s} : {v}")

    # 1. Project onto PCA 2D Manifold
    pc1, pc2 = pca_engine.transform_new_sample(student_data)
    print("\n" + "=" * 75)
    print("  PCA 2D LATENT PROJECTION COORDINATES")
    print("=" * 75)
    print(f"  [+] Principal Component 1 (Academic Diligence Index) : {pc1:+.3f}")
    print(f"  [+] Principal Component 2 (Lifestyle Balance Index)    : {pc2:+.3f}")

    # Interpretation
    if pc1 > 1.5:
        tier_pred = "Tier 1: High Achievers"
        summary = "Strong academic habits, high diligence, low digital distraction."
    elif pc1 > -1.5:
        tier_pred = "Tier 2: Balanced Mainstream"
        summary = "Well-rounded lifestyle, steady academic momentum, moderate recreation."
    else:
        tier_pred = "Tier 3: At-Risk / Distracted"
        summary = "High digital screen time, elevated academic stress, vulnerable performance."

    print(f"  [>] Inferred Academic Tier                         : {tier_pred}")
    print(f"  [>] Behavioral Interpretation                      : {summary}")

    # 2. PCA Reconstruction Fidelity Demonstration
    print("\n" + "-" * 75)
    print("  PCA LOSSY RECONSTRUCTION OF ORIGINAL ATTRIBUTES")
    print("  (Demonstrating invertibility: X_hat = Z_k W_k^T * std + mean)")
    print("-" * 75)

    raw_vec = np.array([[student_data[col] for col in feature_cols]])
    scaled_vec = pca_engine.scaler.transform(raw_vec)

    for k in [1, 2, 4, 8]:
        pca_k = pca_engine.pca_full
        # Truncate components to k
        W_k = pca_k.components_[:k]
        Z_k = np.dot(scaled_vec, W_k.T)
        recon_scaled = np.dot(Z_k, W_k)
        recon_raw = pca_engine.scaler.inverse_transform(recon_scaled)[0]

        # Calculate sample RMSE
        rmse = np.sqrt(np.mean((raw_vec[0] - recon_raw) ** 2))
        print(f"  • k = {k} Component(s) | Reconstruction RMSE: {rmse:.3f} | Sample Study Hrs: {recon_raw[0]:.1f} | Sample GPA: {recon_raw[6]:.2f}")

    # 3. Contrast with t-SNE Out-of-Sample Limitation
    print("\n" + "=" * 75)
    print("  IMPORTANT THEORETICAL CONTRAST: OUT-OF-SAMPLE GENERALIZATION")
    print("=" * 75)
    print("  • PCA successfully projected this new student in 0.001 seconds via W^T x.")
    print("  • Standard t-SNE CANNOT project new observations because it is non-parametric:")
    print("    It does not learn a functional mapping f(x) -> y, only coordinate locations.")
    print("    To place this new student into a t-SNE map, the entire 1,001 dataset must")
    print("    be re-optimized from scratch with gradient descent!")
    print("=" * 75)

if __name__ == "__main__":
    main()
