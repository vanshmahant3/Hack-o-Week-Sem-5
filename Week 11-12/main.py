import os
import sys
import time
import numpy as np
import pandas as pd

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_dataset import prepare_dataset
from pca_analysis import PCAAnalysis
from tsne_analysis import TSNEAnalysis
from pca_vs_tsne_comparator import PCAvsTSNEComparator
from visualizer import DimensionalityReductionVisualizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def main():
    print("=" * 85)
    print("  WEEK 11-12: DIMENSIONALITY REDUCTION SUITE")
    print("  Principal Component Analysis (PCA) & t-Distributed Stochastic Neighbor Embedding (t-SNE)")
    print("  Application: Student Lifestyle & Academic Performance Manifold Exploration")
    print("=" * 85)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, "reports")
    visualizer = DimensionalityReductionVisualizer(output_dir=reports_dir)

    # -------------------------------------------------------------------------
    # STEP 0: Ingestion
    # -------------------------------------------------------------------------
    print("\n[STEP 0] Ingesting Student Lifestyle & Academic Dataset (1,000 records)...")
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
    target_col = "academic_tier"

    X = df[feature_cols].values
    y = df[target_col].values

    print(f"  [+] Samples (N)               : {X.shape[0]}")
    print(f"  [+] Features (D)              : {X.shape[1]} continuous predictors")
    print(f"  [+] Cohort Class Distribution : {df[target_col].value_counts().to_dict()}")

    # -------------------------------------------------------------------------
    # STEP 1: Principal Component Analysis (PCA)
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Executing Principal Component Analysis (PCA)...")
    start_pca = time.time()
    pca_engine = PCAAnalysis(feature_names=feature_cols)
    pca_engine.fit(X)
    pca_duration = time.time() - start_pca

    df_var = pca_engine.get_explained_variance_summary()
    print("\n--- PCA Explained Variance Summary Table ---")
    print(df_var.to_string(index=False))

    thresholds = pca_engine.get_threshold_components([0.70, 0.80, 0.90, 0.95])
    print("\n--- Component Threshold Milestones ---")
    for th, k in thresholds.items():
        print(f"  [>] {th:15s}: Requires {k} components (out of 8)")

    df_loadings = pca_engine.get_loadings_dataframe()
    print("\n--- Factor Loadings Matrix (PC1 & PC2) ---")
    print(df_loadings[["PC1", "PC2"]].to_string())

    print("\nEvaluating PCA Inverse Reconstruction Loss (RMSE vs k)...")
    df_errors = pca_engine.evaluate_reconstruction_errors(X)
    print(df_errors.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 2: t-Distributed Stochastic Neighbor Embedding (t-SNE)
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Executing t-SNE Non-Linear Manifold Optimization...")
    start_tsne = time.time()
    tsne_engine = TSNEAnalysis(random_state=42)
    perplexities = [5, 15, 30, 50]
    perp_embeddings = tsne_engine.fit_perplexity_sweep(pca_engine.X_scaled, perplexities=perplexities)
    tsne_duration = time.time() - start_tsne

    df_perp = tsne_engine.get_perplexity_summary()
    print("\n--- t-SNE Perplexity Exploration Summary ---")
    print(df_perp.to_string(index=False))

    print("\nComparing t-SNE Initialization Schemes (Perplexity = 30)...")
    init_comparison = tsne_engine.compare_initialization_schemes(pca_engine.X_scaled, perplexity=30)
    for init_k, res in init_comparison.items():
        print(f"  [>] Initialization: {init_k.upper():6s} | Final KL Divergence: {res['kl_divergence']:.4f} | Compute Time: {res['duration_sec']:.3f}s")

    # -------------------------------------------------------------------------
    # STEP 3: Rigorous Benchmark Comparison
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Running PCA vs. t-SNE Quantitative & Qualitative Tournament...")
    X_tsne_opt = tsne_engine.optimal_embedding
    df_benchmark = PCAvsTSNEComparator.evaluate_2d_embeddings(
        X_high=pca_engine.X_scaled,
        X_pca_2d=pca_engine.X_pca_2d,
        X_tsne_2d=X_tsne_opt,
        labels=y,
        pca_time=pca_duration,
        tsne_time=tsne_duration
    )
    print("\n=== DIMENSIONALITY REDUCTION SHOWDOWN LEADERBOARD ===")
    print(df_benchmark.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 4: Export Diagnostic Visualizations
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Generating Publication-Grade Visual Reports (300 DPI)...")

    p1 = visualizer.plot_pca_scree_and_cumulative_variance(df_var)
    print(f"  [+] Saved Plot 1: {os.path.basename(p1)}")

    p2 = visualizer.plot_pca_biplot_and_loadings(pca_engine.X_pca_2d, y, df_loadings)
    print(f"  [+] Saved Plot 2: {os.path.basename(p2)}")

    p3 = visualizer.plot_tsne_perplexity_exploration(perp_embeddings, y)
    print(f"  [+] Saved Plot 3: {os.path.basename(p3)}")

    p4 = visualizer.plot_pca_vs_tsne_2d_showdown(pca_engine.X_pca_2d, X_tsne_opt, y)
    print(f"  [+] Saved Plot 4: {os.path.basename(p4)}")

    p5 = visualizer.plot_reconstruction_error_and_benchmark(df_errors, df_benchmark)
    print(f"  [+] Saved Plot 5: {os.path.basename(p5)}")

    print("\n" + "=" * 85)
    print("  WEEK 11–12 EXECUTION COMPLETE: ALL MODELS BENCHMARKED & PLOTS EXPORTED")
    print(f"  Visualizations available in: {reports_dir}")
    print("=" * 85)

if __name__ == "__main__":
    main()
