import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure local module visibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_dataset import prepare_dataset
from bias_variance_engine import BiasVarianceEngine
from ensemble_models import EnsembleTournament
from regularization_study import RegularizationStudy
from visualizer import EnsembleVisualizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def main():
    print("=" * 85)
    print("  WEEK 13-14: ENSEMBLE METHODS & GENERALIZATION FOUNDATIONS")
    print("  Bagging, Boosting (XGBoost & LightGBM), Bias-Variance Tradeoff & Regularization")
    print("  Domain: Streaming Service Subscriber Retention & Churn Prevention")
    print("=" * 85)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, "reports")
    visualizer = EnsembleVisualizer(output_dir=reports_dir)

    # -------------------------------------------------------------------------
    # STEP 0: Ingestion & Feature Pipeline
    # -------------------------------------------------------------------------
    print("\n[STEP 0] Ingesting Streaming Subscriber Dataset (2,000 records)...")
    csv_path = prepare_dataset()
    df = pd.read_csv(csv_path)

    X_processed, y, feature_names = BiasVarianceEngine.prepare_feature_pipeline(df)
    print(f"[+] Matrix Prepared: {X_processed.shape[0]} samples x {X_processed.shape[1]} encoded features")

    # Train / Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"    - Training Set: {len(X_train)} samples ({y_train.sum()} churned)")
    print(f"    - Holdout Test Set: {len(X_test)} samples ({y_test.sum()} churned)")

    # -------------------------------------------------------------------------
    # STEP 1: Bias-Variance Decomposition & Underfitting/Overfitting Sweep
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Running Bootstrap Bias-Variance Decomposition (B=50 resamples)...")
    bv_engine = BiasVarianceEngine(random_state=42)

    df_bv = bv_engine.decompose_bias_variance(
        X_train, y_train, X_test, y_test,
        depths=[1, 2, 3, 4, 6, 8, 12, 16, 20],
        n_bootstraps=50
    )
    print("\n=== Bias-Variance Error Decomposition Table ===")
    print(df_bv.to_string(index=False))

    # Single Tree vs Bagging vs Random Forest
    print("\n[*] Evaluating Bagging Variance Reduction Principle...")
    df_bagging_bv = bv_engine.compare_single_vs_bagging_bias_variance(
        X_train, y_train, X_test, y_test, n_bootstraps=50
    )
    print("\n=== Single Tree vs. Bagging vs. Random Forest Decomposition ===")
    print(df_bagging_bv.to_string(index=False))

    fig1_path = visualizer.plot_bias_variance_decomposition(df_bv, df_bagging_bv)
    print(f"[✓] Saved: {fig1_path}")

    # Learning Curves
    print("\n[*] Computing 5-Fold Learning Curves across Sample Sizes...")
    curve_data = bv_engine.compute_learning_curves(X_processed, y)
    fig2_path = visualizer.plot_learning_curves(curve_data)
    print(f"[✓] Saved: {fig2_path}")

    # -------------------------------------------------------------------------
    # STEP 2: Ensemble Tournament: Bagging vs Boosting (XGBoost & LightGBM)
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Training & Evaluating Flagship Ensemble Architectures...")
    tournament = EnsembleTournament(random_state=42)
    df_leaderboard = tournament.train_and_evaluate(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 95)
    print("  ENSEMBLE TOURNAMENT BENCHMARK LEADERBOARD (Holdout Test Evaluation)")
    print("=" * 95)
    print(df_leaderboard.to_string(index=False))

    # Study Random Forest OOB convergence
    print("\n[*] Analyzing Random Forest Out-of-Bag (OOB) Score Convergence...")
    df_oob = tournament.study_random_forest_oob_convergence(X_train, y_train, X_test, y_test)
    print(f"    - Final OOB Accuracy (180 trees): {df_oob.iloc[-1]['oob_accuracy']:.2f}% | Test Accuracy: {df_oob.iloc[-1]['test_accuracy']:.2f}%")

    # Feature Importances
    df_imp = tournament.get_feature_importances(feature_names)
    print("\n=== Top 6 Feature Importances across Flagship Ensembles ===")
    print(df_imp.head(6).to_string(index=False))

    fig3_path = visualizer.plot_ensemble_comparison(tournament.eval_results, df_leaderboard)
    print(f"[✓] Saved: {fig3_path}")

    fig4_path = visualizer.plot_xgboost_vs_lightgbm_performance(df_leaderboard, df_imp)
    print(f"[✓] Saved: {fig4_path}")

    # -------------------------------------------------------------------------
    # STEP 3: Regularization Study (L1 Alpha, L2 Lambda, Early Stopping)
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Investigating Regularization (L1 vs. L2) & Early Stopping...")
    reg_study = RegularizationStudy(random_state=42)

    # L1 Sweep
    df_alpha = reg_study.sweep_l1_alpha(X_train, y_train, X_test, y_test)
    print("\n=== L1 Regularization (reg_alpha) Sparsity & Generalization Sweep ===")
    print(df_alpha.to_string(index=False))

    # L2 Sweep
    df_lambda = reg_study.sweep_l2_lambda(X_train, y_train, X_test, y_test)
    print("\n=== L2 Regularization (reg_lambda) Leaf Smoothing Sweep ===")
    print(df_lambda.to_string(index=False))

    # Early Stopping
    print("\n[*] Simulating Early Stopping on 250 Boosting Iterations...")
    # Further split train into train/validation for early stopping demonstration
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
    )
    early_stop_data = reg_study.demonstrate_early_stopping(X_tr_sub, y_tr_sub, X_val, y_val)
    print(f"    - Optimal Stopping Iteration: Round {early_stop_data['optimal_round']} (Best Val Loss: {early_stop_data['best_val_loss']:.4f})")
    print(f"    - Loss at Round 250 without stopping: {early_stop_data['final_val_loss_at_250']:.4f}")
    print(f"    - Overfitting Loss Avoided: {early_stop_data['overfitting_loss_penalty']:.4f}")

    fig5_path = visualizer.plot_regularization_and_early_stopping(df_alpha, df_lambda, early_stop_data)
    print(f"[✓] Saved: {fig5_path}")

    # -------------------------------------------------------------------------
    # STEP 4: Summary & Key Takeaways
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("  WEEK 13-14 PIPELINE EXECUTION COMPLETE")
    print("=" * 85)
    print("Key Takeaways for Students:")
    print("1. Bias-Variance Tradeoff: Simple models underfit (High Bias); deep unconstrained models overfit (High Variance).")
    print("2. Bagging & Random Forest: Reduce variance via parallel bootstrap averaging without inflating bias.")
    print("3. Boosting (XGBoost / LightGBM): Sequentially reduce bias; use shrinkage + column subsampling + L1/L2 to control variance.")
    print("4. LightGBM vs XGBoost: LightGBM leaf-wise tree growth and histogram binning deliver significant speedups with state-of-the-art AUC.")
    print("5. Regularization & Early Stopping: L1 induces feature sparsity; L2 dampens leaf weights; Early Stopping halts boosting before over-memorization.")
    print("=" * 85)

if __name__ == "__main__":
    main()
