import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

class Visualizer:
    """Renders and exports publication-grade Linear Algebra & Calculus diagnostic charts."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['axes.edgecolor'] = '#cbd5e1'
        plt.rcParams['axes.linewidth'] = 0.8

    def plot_booking_vector_similarity(self, sim_matrix: np.ndarray, sample_labels: List[str]):
        """Plot 1: Heatmap of Pairwise Vector Dot Products & Cosine Similarities."""
        plt.figure(figsize=(8.5, 7))
        sns.heatmap(
            sim_matrix,
            annot=True,
            fmt=".2f",
            cmap="Purples",
            xticklabels=sample_labels,
            yticklabels=sample_labels,
            cbar_kws={'label': 'Cosine Similarity: cos(θ) = (u · v) / (||u|| ||v||)'}
        )
        plt.title("Hotel Booking Feature Vector Cosine Similarity Matrix", fontsize=12, fontweight='bold', pad=12)
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "01_booking_vector_similarity.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_covariance_and_eigen_scree(self, cov_matrix: np.ndarray, feature_names: List[str], explained_var_ratio: np.ndarray):
        """Plot 2: Feature Covariance Matrix & Eigenvalue Explained Variance Scree Plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

        # Subplot 1: Covariance Matrix Heatmap
        sns.heatmap(
            cov_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            xticklabels=feature_names,
            yticklabels=feature_names,
            ax=ax1,
            cbar_kws={'label': 'Sample Covariance: Σ = 1/(N-1) X^T X'}
        )
        ax1.set_title("Standardized Feature Covariance Matrix (Σ)", fontsize=11, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

        # Subplot 2: Eigenvalue Scree & Cumulative Variance
        cum_var = np.cumsum(explained_var_ratio) * 100
        ind_var = explained_var_ratio * 100
        pcs = [f"PC{i+1}" for i in range(len(explained_var_ratio))]

        ax2.bar(pcs, ind_var, color='#702A8C', alpha=0.7, label='Individual Variance % (λ_i / Σλ)')
        ax2.plot(pcs, cum_var, color='#25D366', marker='o', linewidth=2.5, label='Cumulative Variance %')
        
        ax2.set_title("Eigenvalue Spectral Decomposition & Scree Plot", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Principal Eigenvectors (v_i)", fontweight='bold')
        ax2.set_ylabel("Variance Explained (%)", fontweight='bold')
        ax2.set_ylim(0, 105)
        ax2.axhline(y=80, color='#dc2626', linestyle='--', alpha=0.6, label='80% Variance Threshold')
        ax2.legend(loc='center right', frameon=True)
        ax2.tick_params(axis='x', rotation=30)

        plt.suptitle("Linear Algebra: Covariance Structure & Eigendecomposition (Σ v = λ v)", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "02_covariance_eigen_scree.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_pca_2d_projection(self, Z: np.ndarray, y: np.ndarray, sample_size: int = 1500):
        """Plot 3: 2D PCA Space Projection colored by cancellation status."""
        plt.figure(figsize=(9, 6.5))
        
        # Subsample for clear visualization
        idx = np.random.choice(len(Z), min(sample_size, len(Z)), replace=False)
        Z_sub = Z[idx]
        y_sub = y[idx]

        scatter = plt.scatter(
            Z_sub[y_sub == 0, 0], Z_sub[y_sub == 0, 1],
            color='#0284c7', alpha=0.6, s=25, label='Checked-In (Completed)'
        )
        scatter_cancel = plt.scatter(
            Z_sub[y_sub == 1, 0], Z_sub[y_sub == 1, 1],
            color='#dc2626', alpha=0.6, s=25, label='Canceled Reservation'
        )

        plt.title("2D PCA Projection: High-Dimensional Booking Space -> Top 2 Eigen-Axes", fontsize=12, fontweight='bold', pad=12)
        plt.xlabel("Principal Component 1 (v_1: Planning Horizon & Lead Time Axis)", fontweight='bold')
        plt.ylabel("Principal Component 2 (v_2: Price & Commitment Volatility Axis)", fontweight='bold')
        plt.axhline(0, color='grey', linestyle=':', alpha=0.5)
        plt.axvline(0, color='grey', linestyle=':', alpha=0.5)
        plt.legend(frameon=True, loc='upper right')
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "03_pca_cancellation_projection.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_gradient_flow_and_verification(self, analytical_grads: List[float], numerical_grads: List[float], grad_history: List[float]):
        """Plot 4: Numerical vs Analytical Gradient Verification & Backprop Gradient Flow."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot 1: Analytical vs Numerical Finite Difference (Multi-point scatter)
        num_arr = np.array(numerical_grads)
        ana_arr = np.array(analytical_grads)

        ax1.scatter(num_arr, ana_arr, color='#702A8C', s=65, alpha=0.85, edgecolors='#1e1b4b', zorder=5, label='Network Parameters (w, b)')
        
        # Determine dynamic plot limits with padding
        all_vals = np.concatenate([num_arr, ana_arr])
        min_v = np.min(all_vals) - 0.05
        max_v = np.max(all_vals) + 0.05
        ax1.plot([min_v, max_v], [min_v, max_v], color='#16a34a', linestyle='--', linewidth=2.5, zorder=3, label=r'Perfect Chain Rule Agreement: $y = x$')
        
        ax1.set_xlim(min_v, max_v)
        ax1.set_ylim(min_v, max_v)
        ax1.set_title(r"Multivariate Chain Rule vs. Finite Difference Gradients", fontsize=11, fontweight='bold')
        ax1.set_xlabel(r"Central Finite Difference: $\frac{f(w+\epsilon) - f(w-\epsilon)}{2\epsilon}$", fontsize=10, fontweight='bold')
        ax1.set_ylabel(r"Autograd Reverse-Mode: $\frac{\partial \mathcal{L}}{\partial w}$", fontsize=10, fontweight='bold')
        ax1.legend(frameon=True, loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Subplot 2: Gradient Norm Trajectory during Backprop
        epochs = range(1, len(grad_history) + 1)
        ax2.plot(epochs, grad_history, color='#0284c7', linewidth=2.4, marker='o', markersize=5, label=r'Total Gradient L2 Norm: $||\nabla_w \mathcal{L}||_2$')
        ax2.set_title(r"SGD Optimization: Gradient Norm Dynamics over Training Epochs", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Training Epoch", fontweight='bold')
        ax2.set_ylabel("Total Gradient Magnitude", fontweight='bold')
        ax2.legend(frameon=True, loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle("Calculus & Autograd Engine: Exact Reverse-Mode Automatic Differentiation", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "04_gradient_flow_autograd_verification.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_neural_net_loss_decision_boundary(self, history: Dict, mlp_pca, X_pca: np.ndarray, y: np.ndarray):
        """Plot 5: Training Loss Curve and Non-Linear Classification Decision Boundary."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot 1: Loss & Accuracy Curves
        epochs = range(1, len(history['loss']) + 1)
        ax1.plot(epochs, history['loss'], color='#dc2626', linewidth=2.5, label='Binary Cross-Entropy Loss')
        ax1.set_xlabel("Epoch", fontweight='bold')
        ax1.set_ylabel("Loss Magnitude", color='#dc2626', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#dc2626')

        ax1_twin = ax1.twinx()
        ax1_twin.plot(epochs, [acc * 100 for acc in history['accuracy']], color='#16a34a', linewidth=2.5, linestyle='-.', label='Accuracy (%)')
        ax1_twin.set_ylabel("Accuracy (%)", color='#16a34a', fontweight='bold')
        ax1_twin.tick_params(axis='y', labelcolor='#16a34a')
        ax1.set_title("Neural Network Backpropagation Convergence (Loss & Accuracy)", fontsize=11, fontweight='bold')

        # Subplot 2: 2D Classification Decision Boundary on PCA Space
        x_min, x_max = X_pca[:, 0].min() - 0.8, X_pca[:, 0].max() + 0.8
        y_min, y_max = X_pca[:, 1].min() - 0.8, X_pca[:, 1].max() + 0.8
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))

        # Evaluate MLP over mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        probs = []
        for pt in mesh_points:
            pred = mlp_pca(pt.tolist())
            probs.append(pred.data)
        Z_mesh = np.array(probs).reshape(xx.shape)

        # Plot contour
        ax2.contourf(xx, yy, Z_mesh, levels=20, cmap='RdBu_r', alpha=0.6)
        contour_line = ax2.contour(xx, yy, Z_mesh, levels=[0.5], colors='black', linewidths=2.0)
        ax2.clabel(contour_line, inline=True, fontsize=9, fmt="Decision Boundary (p=0.5)")

        # Scatter samples
        sub_idx = np.random.choice(len(X_pca), min(500, len(X_pca)), replace=False)
        ax2.scatter(X_pca[sub_idx, 0][y[sub_idx] == 0], X_pca[sub_idx, 1][y[sub_idx] == 0], color='#0284c7', s=20, alpha=0.8, label='Checked-In (0)')
        ax2.scatter(X_pca[sub_idx, 0][y[sub_idx] == 1], X_pca[sub_idx, 1][y[sub_idx] == 1], color='#dc2626', s=20, alpha=0.8, label='Canceled (1)')
        
        ax2.set_title("Trained Neural Network Decision Boundary on 2D PCA Space", fontsize=11, fontweight='bold')
        ax2.set_xlabel("PC 1", fontweight='bold')
        ax2.set_ylabel("PC 2", fontweight='bold')
        ax2.legend(loc='upper right', frameon=True)

        plt.suptitle("Calculus in Action: Backpropagation & Non-Linear Decision Surface", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "05_neural_net_loss_decision_boundary.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")
