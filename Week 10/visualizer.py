import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from typing import Dict, Any, List

class ClusteringVisualizer:
    """Generates publication-grade diagnostic visualizations for Scikit-Learn Clustering Workflows."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['axes.edgecolor'] = '#cbd5e1'
        plt.rcParams['axes.linewidth'] = 0.8

    def plot_kmeans_elbow_and_silhouette(self, tuning_results: Dict[str, Any]):
        """Plot 1: K-Means Elbow Curve (Inertia) & Silhouette Scores across k."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
        ks = tuning_results['k_values']
        inertias = tuning_results['inertias']
        silhouettes = tuning_results['silhouettes']
        opt_k = tuning_results['optimal_k']

        # Subplot 1: Elbow Curve
        ax1.plot(ks, inertias, marker='o', color='#2563eb', linewidth=2.5, markersize=7)
        ax1.axvline(x=opt_k, color='#ef4444', linestyle='--', linewidth=1.8, label=f'Optimal k = {opt_k}')
        ax1.set_title("Elbow Method: Within-Cluster Sum of Squares (WCSS)", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Number of Clusters (k)", fontweight='bold')
        ax1.set_ylabel("Inertia (WCSS)", fontweight='bold')
        ax1.set_xticks(ks)
        ax1.legend(frameon=True, loc='upper right')

        # Subplot 2: Silhouette Scores
        ax2.plot(ks, silhouettes, marker='s', color='#10b981', linewidth=2.5, markersize=7)
        ax2.axvline(x=opt_k, color='#ef4444', linestyle='--', linewidth=1.8, label=f'Peak Silhouette (k = {opt_k})')
        best_sil = tuning_results['best_silhouette']
        ax2.scatter([opt_k], [best_sil], color='#ef4444', s=90, zorder=5)
        ax2.set_title("Silhouette Analysis Across Cluster Quantities", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Number of Clusters (k)", fontweight='bold')
        ax2.set_ylabel("Mean Silhouette Score", fontweight='bold')
        ax2.set_xticks(ks)
        ax2.legend(frameon=True, loc='lower right')

        plt.suptitle("K-Means Hyperparameter Search: WCSS Inertia & Silhouette Convergence", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "01_kmeans_elbow_and_silhouette.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_hierarchical_dendrogram(self, linkage_matrix: np.ndarray, color_threshold: float = 38.0):
        """Plot 2: Hierarchical Clustering Tree Dendrogram (Ward Linkage)."""
        plt.figure(figsize=(15, 6))
        dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            p=30,
            leaf_rotation=90,
            leaf_font_size=10,
            show_contracted=True,
            color_threshold=color_threshold
        )
        plt.axhline(y=color_threshold, color='#ef4444', linestyle='--', linewidth=2.0, label=f'Cut Threshold (Distance = {color_threshold})')
        plt.title("Hierarchical Agglomerative Dendrogram (Ward Linkage)", fontsize=13, fontweight='bold')
        plt.xlabel("Cluster Cohort Sub-Tree (Truncated to Top 30 Branches)", fontweight='bold')
        plt.ylabel("Euclidean Fusion Distance", fontweight='bold')
        plt.legend(frameon=True, loc='upper right')
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "02_hierarchical_dendrogram.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_dbscan_kdistance_and_noise(self, k_distances: np.ndarray, eps_val: float,
                                       X_2d: np.ndarray, dbscan_labels: np.ndarray):
        """Plot 3: DBSCAN k-Distance Knee Plot & Noise Outlier Projection."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

        # Subplot 1: k-distance knee graph
        x_indices = np.arange(len(k_distances))
        ax1.plot(x_indices, k_distances, color='#3b82f6', linewidth=2.5)
        ax1.axhline(y=eps_val, color='#ef4444', linestyle='--', linewidth=1.8, label=f'Estimated Epsilon (eps = {eps_val:.2f})')
        ax1.set_title("k-Distance Graph for Optimal Epsilon (eps) Selection", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Points Sorted by k-Nearest Neighbor Distance", fontweight='bold')
        ax1.set_ylabel("Distance to kth Nearest Neighbor", fontweight='bold')
        ax1.legend(frameon=True, loc='upper left')

        # Subplot 2: DBSCAN 2D projection with noise highlighted
        noise_mask = dbscan_labels == -1
        core_mask = ~noise_mask

        scatter_core = ax2.scatter(X_2d[core_mask, 0], X_2d[core_mask, 1], c=dbscan_labels[core_mask],
                                   cmap='tab10', s=25, alpha=0.7, label='Core & Border Points')
        scatter_noise = ax2.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1], color='#ef4444',
                                    marker='x', s=45, label=f'Noise / Anomalies ({np.sum(noise_mask)} points)')
        ax2.set_title(f"DBSCAN Density Partition ({len(set(dbscan_labels)) - 1} Clusters + Outliers)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Principal Component 1 (PCA)", fontweight='bold')
        ax2.set_ylabel("Principal Component 2 (PCA)", fontweight='bold')
        ax2.legend(frameon=True, loc='upper right')

        plt.suptitle("DBSCAN Density Diagnostics: Epsilon Determination & Anomaly Extraction", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "03_dbscan_kdistance_and_noise.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_clustering_algorithms_comparison(self, X_2d: np.ndarray, km_labels: np.ndarray,
                                            hc_labels: np.ndarray, db_labels: np.ndarray):
        """Plot 4: 3-Panel Side-by-Side 2D PCA Comparison of K-Means, Hierarchical, and DBSCAN."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

        # 1. K-Means
        axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=km_labels, cmap='tab10', s=20, alpha=0.65)
        axes[0].set_title(f"1. K-Means Clustering (k = {len(set(km_labels))})\nSpherical Voronoi Centroids", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("Principal Component 1")
        axes[0].set_ylabel("Principal Component 2")

        # 2. Hierarchical (Agglomerative)
        axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=hc_labels, cmap='tab10', s=20, alpha=0.65)
        axes[1].set_title(f"2. Hierarchical Clustering (k = {len(set(hc_labels))})\nWard Linkage Agglomeration", fontsize=11, fontweight='bold')
        axes[1].set_xlabel("Principal Component 1")
        axes[1].set_ylabel("Principal Component 2")

        # 3. DBSCAN
        noise_mask = db_labels == -1
        core_mask = ~noise_mask
        axes[2].scatter(X_2d[core_mask, 0], X_2d[core_mask, 1], c=db_labels[core_mask], cmap='tab10', s=20, alpha=0.65)
        axes[2].scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1], color='#64748b', marker='.', s=15, alpha=0.5, label='Noise')
        axes[2].set_title(f"3. DBSCAN Density Clustering\nArbitrary Shapes & Noise Extraction", fontsize=11, fontweight='bold')
        axes[2].set_xlabel("Principal Component 1")
        axes[2].set_ylabel("Principal Component 2")

        plt.suptitle("Clustering Algorithms Tournament: 2D Spatial Partition Comparison", fontsize=13, fontweight='bold', y=1.04)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "04_clustering_algorithms_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_cluster_persona_radar_profiles(self, profile_df: pd.DataFrame):
        """Plot 5: Multi-Feature Radar / Spider Profile Charts for Customer Personas."""
        # Filter valid clusters (exclude noise if present)
        df_valid = profile_df[profile_df['Cluster'] != -1].copy()
        
        features = ['annual_income_k', 'spending_score', 'monthly_orders', 'avg_order_value', 'web_browsing_hours']
        labels_pretty = ['Income ($k)', 'Spend Score', 'Orders/Mo', 'AOV ($)', 'Web Hours']
        num_vars = len(features)

        # Min-Max normalize mean features across clusters for visual radar comparison
        normalized_df = df_valid[features].copy()
        for col in features:
            c_min = normalized_df[col].min()
            c_max = normalized_df[col].max()
            normalized_df[col] = (normalized_df[col] - c_min) / (c_max - c_min + 1e-9)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = ['#2563eb', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4']

        for idx, (_, row) in enumerate(normalized_df.iterrows()):
            values = row.values.flatten().tolist()
            values += values[:1]
            p_name = df_valid.iloc[idx]['Persona_Name'].split('(')[0].strip()
            ax.plot(angles, values, color=colors[idx % len(colors)], linewidth=2.2, label=p_name)
            ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.12)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_pretty, fontweight='bold', fontsize=10)
        ax.set_title("Customer Persona Radar Profiles: Behavioral Attribute Footprint", fontsize=12, fontweight='bold', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), frameon=True)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "05_cluster_persona_radar_profiles.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")
