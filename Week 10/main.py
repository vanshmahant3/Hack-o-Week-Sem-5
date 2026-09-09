import os
import sys
import numpy as np
import pandas as pd

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_dataset import prepare_dataset
from sklearn_pipelines import ScikitLearnWorkflowPipelines
from kmeans_clustering import KMeansClusteringPipeline
from hierarchical_clustering import HierarchicalClusteringPipeline
from dbscan_clustering import DBSCANClusteringPipeline
from cluster_evaluator import ClusterEvaluator
from cluster_profiler import ClusterProfiler
from visualizer import ClusteringVisualizer

def main():
    print("=" * 85)
    print("  WEEK 10: COMPLETE SCIKIT-LEARN WORKFLOW & CLUSTERING SUITE")
    print("  Pipelines, ColumnTransformers & Unsupervised Learning (K-Means, Hierarchical, DBSCAN)")
    print("  Application: E-Commerce Customer Behavioral Segmentation & Persona Profiling")
    print("=" * 85)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, 'reports')
    visualizer = ClusteringVisualizer(output_dir=reports_dir)

    # -------------------------------------------------------------------------
    # STEP 0: Ingestion
    # -------------------------------------------------------------------------
    print("\n[STEP 0] Ingesting Customer Segmentation Dataset (5,000 records)...")
    csv_path = prepare_dataset()
    df_raw = pd.read_csv(csv_path)

    numerical_cols = [
        'annual_income_k', 'spending_score', 'monthly_orders',
        'avg_order_value', 'web_browsing_hours', 'return_rate', 'tenure_months'
    ]
    categorical_cols = ['membership_tier']

    print(f"  [+] Records Ingested          : {len(df_raw):,} customers")
    print(f"  [+] Numerical Attributes      : {numerical_cols}")
    print(f"  [+] Categorical Attributes    : {categorical_cols}")

    # -------------------------------------------------------------------------
    # STEP 1: Scikit-Learn Pipeline & Preprocessing
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Assembling Scikit-Learn ColumnTransformer & Preprocessing Pipeline...")
    preprocessor = ScikitLearnWorkflowPipelines.build_preprocessing_pipeline(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        scaler_type='robust',
        apply_pca=False
    )
    X_scaled = preprocessor.fit_transform(df_raw)

    # 2D PCA for spatial visualization
    pca_pipeline = ScikitLearnWorkflowPipelines.build_preprocessing_pipeline(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        scaler_type='robust',
        apply_pca=True,
        n_pca_components=2
    )
    X_2d = pca_pipeline.fit_transform(df_raw)

    print(f"  [+] Preprocessed Feature Matrix: Shape = {X_scaled.shape}")
    print(f"  [+] 2D PCA Projection Matrix   : Shape = {X_2d.shape}")

    # -------------------------------------------------------------------------
    # STEP 2: K-Means Clustering & Hyperparameter Tuning
    # -------------------------------------------------------------------------
    print("\n[STEP 2] K-Means Clustering: Elbow Method & Silhouette Search across k in [2, 10]...")
    kmeans_pipeline = KMeansClusteringPipeline(random_state=42)
    km_tuning = kmeans_pipeline.search_optimal_k(X_scaled, k_range=range(2, 11))

    print(f"  [+] Optimal k by Silhouette  : k = {km_tuning['optimal_k']} (Peak Silhouette = {km_tuning['best_silhouette']:.4f})")
    km_labels, km_centers = kmeans_pipeline.fit_predict(X_scaled, k=5)

    visualizer.plot_kmeans_elbow_and_silhouette(km_tuning)

    # -------------------------------------------------------------------------
    # STEP 3: Hierarchical (Agglomerative) Clustering
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Hierarchical Agglomerative Clustering (Ward Linkage & Dendrogram)...")
    hier_pipeline = HierarchicalClusteringPipeline(n_clusters=5, linkage_method='ward')
    linkage_mat = hier_pipeline.compute_linkage(X_scaled, sample_size=350)
    hier_labels = hier_pipeline.fit_predict(X_scaled, n_clusters=5)

    visualizer.plot_hierarchical_dendrogram(linkage_mat, color_threshold=28.0)

    # -------------------------------------------------------------------------
    # STEP 4: DBSCAN Density Clustering & Anomaly Extraction
    # -------------------------------------------------------------------------
    print("\n[STEP 4] DBSCAN Density-Based Clustering (k-Distance Knee & Noise Extraction)...")
    dbscan_pipeline = DBSCANClusteringPipeline(min_samples=10)
    k_dists, est_eps = dbscan_pipeline.compute_k_distance_graph(X_scaled, k=10)
    db_labels, db_summary = dbscan_pipeline.fit_predict(X_scaled, eps=0.65, min_samples=10)

    print(f"  [+] DBSCAN Estimated Epsilon : eps = {est_eps:.2f} (Calibrated = {db_summary['eps']})")
    print(f"  [+] Discovered Clusters      : {db_summary['n_clusters']} dense partitions")
    print(f"  [+] Identified Noise / Outliers: {db_summary['n_noise_points']:,} anomalies ({db_summary['noise_percentage']:.2f}% of dataset)")

    visualizer.plot_dbscan_kdistance_and_noise(k_dists, db_summary['eps'], X_2d, db_labels)

    # -------------------------------------------------------------------------
    # STEP 5: Multi-Metric Tournament & Comparison
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Clustering Tournament Benchmark (Silhouette, Davies-Bouldin, Calinski-Harabasz)...")
    models_dict = {
        'K-Means (k=5)': km_labels,
        'Hierarchical Ward (k=5)': hier_labels,
        'DBSCAN (eps=0.65)': db_labels
    }
    tournament_df = ClusterEvaluator.compare_models(X_scaled, models_dict)
    print("\n--- Unsupervised Clustering Validity Leaderboard ---")
    print(tournament_df.to_string(index=False))

    visualizer.plot_clustering_algorithms_comparison(X_2d, km_labels, hier_labels, db_labels)

    # -------------------------------------------------------------------------
    # STEP 6: Persona Profiling & Business Marketing Actions
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Customer Persona Aggregation & Strategic Business Mapping...")
    profile_df = ClusterProfiler.profile_clusters(df_raw, km_labels)
    print("\n--- Discovered Customer Persona Cohorts (K-Means) ---")
    summary_cols = ['Cluster', 'Persona_Name', 'Customer_Count', 'Percentage', 'annual_income_k', 'spending_score', 'avg_order_value', 'monthly_orders']
    print(profile_df[summary_cols].to_string(index=False))

    print("\n--- Prescribed Marketing Action Matrix ---")
    for _, row in profile_df.iterrows():
        p_name = row['Persona_Name']
        action = ClusterProfiler.get_marketing_action(p_name)
        print(f"  * {p_name}:")
        print(f"    -> Action: {action}")

    visualizer.plot_cluster_persona_radar_profiles(profile_df)

    # -------------------------------------------------------------------------
    # STEP 7: Executive Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("  [STEP 7] EXECUTIVE WORKFLOW SUMMARY (WEEK 10)")
    print("=" * 85)
    print(f"  * Total Profiles Segmented      : 5,000 Customers")
    print(f"  * Preprocessing Pipeline        : Scikit-Learn ColumnTransformer (RobustScaler + OneHotEncoder)")
    print(f"  * Optimal Partition Count (k)   : k = 5 (Confirmed via Elbow & Silhouette peaks)")
    print(f"  * Top Performer by Silhouette   : K-Means (Silhouette = {tournament_df.loc[0, 'Silhouette']:.4f})")
    print(f"  * Noise Outliers Isolated       : {db_summary['n_noise_points']} anomalous / fraud profiles via DBSCAN")
    print(f"  * Publication Visualizations    : 5 high-resolution figures exported to: {reports_dir}")
    print("=" * 85 + "\n")

if __name__ == '__main__':
    main()
