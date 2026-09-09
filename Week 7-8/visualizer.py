import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

class Visualizer:
    """Renders and exports publication-grade Regression and Classification diagnostic plots."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['axes.edgecolor'] = '#cbd5e1'
        plt.rcParams['axes.linewidth'] = 0.8

    def plot_linear_vs_polynomial_depreciation(self, x_test: np.ndarray, y_test: np.ndarray,
                                              preds_linear: np.ndarray, preds_poly2: np.ndarray, preds_poly4: np.ndarray,
                                              degrees: List[int], train_mse: List[float], test_mse: List[float]):
        """Plot 1: Non-linear Car Age Depreciation and Bias-Variance Tradeoff Curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot 1: Depreciation Curves
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        
        ax1.scatter(x_test, y_test, color='#94a3b8', alpha=0.35, s=20, label='Actual Car Resale Price')
        ax1.plot(x_sorted, preds_linear[sort_idx], color='#ef4444', linewidth=2.5, linestyle='--', label='Linear Regression (Underfitting, Degree 1)')
        ax1.plot(x_sorted, preds_poly2[sort_idx], color='#10b981', linewidth=3.0, label='Polynomial Degree 2 (Optimal Fit)')
        ax1.plot(x_sorted, preds_poly4[sort_idx], color='#8b5cf6', linewidth=2.0, linestyle=':', label='Polynomial Degree 4 (Overfitting / High Variance)')
        
        ax1.set_title("Used Car Price Depreciation: Linear vs. Polynomial Fits", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Car Age (Years)", fontweight='bold')
        ax1.set_ylabel("Resale Price (Lakh INR)", fontweight='bold')
        ax1.legend(frameon=True, loc='upper right')

        # Subplot 2: Bias-Variance Tradeoff Curve
        ax2.plot(degrees, train_mse, color='#0284c7', marker='o', linewidth=2.5, label='Training MSE (Decreases with Complexity)')
        ax2.plot(degrees, test_mse, color='#f59e0b', marker='s', linewidth=2.5, linestyle='-.', label='Testing MSE (U-Shape / Overfitting)')
        ax2.axvline(x=2, color='#10b981', linestyle=':', linewidth=2, label='Optimal Degree (Min Test MSE)')
        
        ax2.set_title("Bias-Variance Tradeoff: Model Complexity vs. Error", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Polynomial Degree", fontweight='bold')
        ax2.set_ylabel("Mean Squared Error (MSE)", fontweight='bold')
        ax2.legend(frameon=True, loc='upper center')

        plt.suptitle("Regression Analysis: Capturing Non-Linear Market Dynamics & Overfitting", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "01_linear_vs_polynomial_depreciation.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_ridge_vs_lasso_regularization(self, alphas: np.ndarray, ridge_coefs: np.ndarray, lasso_coefs: np.ndarray, feature_names: List[str]):
        """Plot 2: Ridge L2 Weight Shrinkage vs. Lasso L1 Feature Sparsity Paths."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot 1: Ridge L2 Shrinkage
        for i in range(ridge_coefs.shape[1]):
            ax1.plot(alphas, ridge_coefs[:, i], label=feature_names[i], linewidth=2.0)
        ax1.set_xscale('log')
        ax1.set_title("Ridge Regression (L2): Continuous Coefficient Shrinkage", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Regularization Strength (alpha)", fontweight='bold')
        ax1.set_ylabel("Standardized Feature Coefficients", fontweight='bold')
        ax1.legend(frameon=True, loc='upper right', fontsize=8)

        # Subplot 2: Lasso L1 Sparsity
        for i in range(lasso_coefs.shape[1]):
            ax2.plot(alphas, lasso_coefs[:, i], label=feature_names[i], linewidth=2.0)
        ax2.set_xscale('log')
        ax2.axhline(0, color='grey', linestyle='--', alpha=0.5)
        ax2.set_title("Lasso Regression (L1): Exact Feature Sparsity & Selection", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Regularization Strength (alpha)", fontweight='bold')
        ax2.set_ylabel("Standardized Feature Coefficients", fontweight='bold')
        ax2.legend(frameon=True, loc='upper right', fontsize=8)

        plt.suptitle("Regularization Showdown: Ridge (L2 Penalty) vs. Lasso (L1 Penalty)", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "02_ridge_vs_lasso_regularization.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_logistic_regression_roc_confusion(self, cm: np.ndarray, fpr: np.ndarray, tpr: np.ndarray, auc_score: float):
        """Plot 3: Logistic Regression Confusion Matrix & ROC-AUC Curve."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Subplot 1: Confusion Matrix
        labels = ['Fair / Overpriced', 'Great Deal']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax1, cbar=False)
        ax1.set_title("Logistic Regression Confusion Matrix", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Predicted Deal Category", fontweight='bold')
        ax1.set_ylabel("Actual Deal Category", fontweight='bold')

        # Subplot 2: ROC Curve
        ax2.plot(fpr, tpr, color='#2563eb', linewidth=2.8, label=f'Logistic Regression (AUC = {auc_score:.3f})')
        ax2.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', linewidth=1.8, label='Random Chance (AUC = 0.500)')
        ax2.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=11, fontweight='bold')
        ax2.set_xlabel("False Positive Rate (FPR)", fontweight='bold')
        ax2.set_ylabel("True Positive Rate (TPR)", fontweight='bold')
        ax2.legend(frameon=True, loc='lower right')

        plt.suptitle("Classification Performance: Logistic Regression Probability Calibration", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "03_logistic_regression_roc_confusion.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_knn_decision_boundary_k_tuning(self, X_2d: np.ndarray, y_2d: np.ndarray, knn_model, k_scores: Dict[int, float], best_k: int):
        """Plot 4: 2D KNN Decision Boundary and k-Hyperparameter Tuning Curve."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot 1: 2D Decision Surface on (Car Age vs Max Power)
        x_min, x_max = X_2d[:, 0].min() - 0.8, X_2d[:, 0].max() + 0.8
        y_min, y_max = X_2d[:, 1].min() - 0.8, X_2d[:, 1].max() + 0.8
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 70), np.linspace(y_min, y_max, 70))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z_mesh = knn_model.predict(mesh_points).reshape(xx.shape)

        ax1.contourf(xx, yy, Z_mesh, levels=1, cmap='coolwarm', alpha=0.35)
        scatter_0 = ax1.scatter(X_2d[y_2d == 0, 0], X_2d[y_2d == 0, 1], color='#3b82f6', s=20, alpha=0.7, label='Fair Price (0)')
        scatter_1 = ax1.scatter(X_2d[y_2d == 1, 0], X_2d[y_2d == 1, 1], color='#ef4444', s=20, alpha=0.7, label='Great Deal (1)')
        
        ax1.set_title(f"KNN Decision Surface (Optimal k = {best_k})", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Standardized Car Age", fontweight='bold')
        ax1.set_ylabel("Standardized Engine Power (BHP)", fontweight='bold')
        ax1.legend(frameon=True, loc='upper right')

        # Subplot 2: Tuning curve across k values
        ks = list(k_scores.keys())
        accs = [v * 100 for v in k_scores.values()]
        ax2.plot(ks, accs, color='#8b5cf6', marker='o', linewidth=2.5, label='Validation Accuracy (%)')
        ax2.axvline(x=best_k, color='#10b981', linestyle='--', linewidth=2, label=f'Best k = {best_k} ({k_scores[best_k]*100:.1f}%)')
        
        ax2.set_title("KNN Hyperparameter Tuning: Neighborhood Size (k) vs. Accuracy", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Number of Neighbors (k)", fontweight='bold')
        ax2.set_ylabel("Classification Accuracy (%)", fontweight='bold')
        ax2.legend(frameon=True, loc='lower right')

        plt.suptitle("Non-Parametric Classification: K-Nearest Neighbors (KNN) Dynamics", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "04_knn_decision_boundary_k_tuning.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_model_leaderboard_comparison(self, reg_df, clf_df):
        """Plot 5: Comprehensive Benchmark Comparison across all 6 Models."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

        # Subplot 1: Regression Comparison (R2 and RMSE)
        models_reg = reg_df['Model'].tolist()
        r2_vals = reg_df['R2_Score'].tolist()
        rmse_vals = reg_df['RMSE_Lakh'].tolist()

        x = np.arange(len(models_reg))
        width = 0.35
        ax1.bar(x - width/2, r2_vals, width, label='R² Score (Higher is Better)', color='#10b981')
        ax1.bar(x + width/2, rmse_vals, width, label='RMSE in Lakh ₹ (Lower is Better)', color='#ef4444')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models_reg, rotation=20, ha='right')
        ax1.set_title("Regression Benchmark Leaderboard (Linear, Poly, Ridge, Lasso)", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Metric Score", fontweight='bold')
        ax1.legend(frameon=True)

        # Subplot 2: Classification Comparison (Accuracy and F1)
        models_clf = clf_df['Model'].tolist()
        acc_vals = [a * 100 for a in clf_df['Accuracy'].tolist()]
        f1_vals = [f * 100 for f in clf_df['F1_Score'].tolist()]

        x2 = np.arange(len(models_clf))
        ax2.bar(x2 - width/2, acc_vals, width, label='Accuracy %', color='#3b82f6')
        ax2.bar(x2 + width/2, f1_vals, width, label='F1-Score %', color='#8b5cf6')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(models_clf, rotation=15, ha='right')
        ax2.set_title("Classification Benchmark Leaderboard (Logistic Regression vs. KNN)", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Percentage (%)", fontweight='bold')
        ax2.set_ylim(0, 105)
        ax2.legend(frameon=True, loc='lower right')

        plt.suptitle("Week 7-8 Executive Model Tournament: Regression & Classification Leaderboard", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "05_model_leaderboard_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")
