import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Configure publication aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.autolayout'] = True

class EnsembleVisualizer:
    """Generates 5 publication-grade 300 DPI figures for Week 13-14."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_bias_variance_decomposition(self, df_bv: pd.DataFrame, df_bag: pd.DataFrame) -> str:
        """Figure 1: Bias-Variance Decomposition across Model Complexity & Bagging Variance Reduction."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

        depths = df_bv["max_depth"]
        bias_sq = df_bv["bias_squared"]
        variance = df_bv["variance"]
        total_err = df_bv["total_mse"]

        # Left Panel: Bias-Variance Curve
        ax1.plot(depths, bias_sq, 'o-', color='#e74c3c', linewidth=2.5, markersize=7, label=r'Bias$^2$ (Underfitting Risk)')
        ax1.plot(depths, variance, 's-', color='#3498db', linewidth=2.5, markersize=7, label=r'Variance (Overfitting Risk)')
        ax1.plot(depths, total_err, '^-', color='#2ecc71', linewidth=3.0, markersize=8, label=r'Total Expected Error (MSE)')

        # Annotate Sweet Spot
        min_idx = np.argmin(total_err)
        opt_depth = depths.iloc[min_idx]
        opt_err = total_err.iloc[min_idx]
        ax1.axvline(opt_depth, color='#27ae60', linestyle='--', linewidth=1.8, alpha=0.8, label=f'Optimal Depth (k={opt_depth})')
        ax1.scatter([opt_depth], [opt_err], color='#27ae60', s=140, zorder=5, edgecolor='black')

        # Shaded Regime Zones
        ax1.axvspan(1, 2.5, alpha=0.12, color='red', label='High Bias Zone (Underfitting)')
        ax1.axvspan(2.5, 6.5, alpha=0.12, color='green', label='Balanced Generalization Zone')
        ax1.axvspan(6.5, 20.5, alpha=0.12, color='blue', label='High Variance Zone (Overfitting)')

        ax1.set_title(r"Bias-Variance Trade-off ($\text{MSE} = \text{Bias}^2 + \text{Variance} + \sigma^2$)", weight='bold')
        ax1.set_xlabel("Model Complexity: Decision Tree Max Depth", weight='bold')
        ax1.set_ylabel("Expected Test Error Components", weight='bold')
        ax1.set_xticks(depths)
        ax1.legend(loc='upper right', frameon=True, fontsize=8.5)
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Right Panel: Bagging Variance Reduction Bar Chart
        models = df_bag["Model Architecture"]
        bias_vals = df_bag["Bias^2"]
        var_vals = df_bag["Variance"]
        x_indices = np.arange(len(models))
        bar_width = 0.35

        ax2.bar(x_indices - bar_width/2, bias_vals, width=bar_width, color='#e74c3c', label=r'Bias$^2$', alpha=0.85)
        ax2.bar(x_indices + bar_width/2, var_vals, width=bar_width, color='#3498db', label='Variance', alpha=0.85)

        for i in range(len(models)):
            var_red = df_bag.loc[i, "Variance Reduction vs Single Tree (%)"]
            if var_red > 0:
                ax2.annotate(f"-{var_red}% Var", (x_indices[i] + bar_width/2, var_vals[i]),
                             textcoords="offset points", xytext=(0, 7), ha='center',
                             weight='bold', color='#2980b9', fontsize=9)

        ax2.set_xticks(x_indices)
        ax2.set_xticklabels([m.replace(" (", "\n(") for m in models], fontsize=9)
        ax2.set_title("Bagging Variance Reduction Principle", weight='bold')
        ax2.set_ylabel("Error Contribution", weight='bold')
        ax2.legend(loc='upper right', frameon=True)
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.suptitle("WEEK 13-14: BIAS-VARIANCE DECOMPOSITION & GENERALIZATION DYNAMICS", weight='bold', fontsize=13)
        file_path = os.path.join(self.output_dir, "01_bias_variance_tradeoff_decomposition.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        return file_path

    def plot_learning_curves(self, curve_data: Dict[str, Any]) -> str:
        """Figure 2: 3-Panel Comparative Learning Curves (Underfitting vs Balanced vs Overfitting)."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300, sharey=True)

        panels = [
            ("Underfitting (Depth 1 Stump)", axes[0], "#e74c3c", "High Bias: Poor Train & Val Convergence"),
            ("Balanced (Random Forest)", axes[1], "#2ecc71", "Optimal: High Val AUC, Narrow Healthy Gap"),
            ("Overfitting (Unpruned Tree)", axes[2], "#3498db", "High Variance: 100% Train, Wide Generalization Gap")
        ]

        for label, ax, theme_col, subtitle in panels:
            data = curve_data[label]
            sizes = data["train_sizes"]
            train_m, train_s = data["train_mean"], data["train_std"]
            val_m, val_s = data["val_mean"], data["val_std"]

            ax.plot(sizes, train_m, 'o-', color='#e67e22', linewidth=2.2, label='Training AUC')
            ax.fill_between(sizes, train_m - train_s, train_m + train_s, alpha=0.15, color='#e67e22')

            ax.plot(sizes, val_m, 's-', color='#2980b9', linewidth=2.2, label='5-Fold Validation AUC')
            ax.fill_between(sizes, val_m - val_s, val_m + val_s, alpha=0.15, color='#2980b9')

            # Highlight gap
            gap = train_m[-1] - val_m[-1]
            ax.annotate(f"Gap: Δ={gap:.3f}", (sizes[-1], (train_m[-1] + val_m[-1])/2),
                        textcoords="offset points", xytext=(-55, 0), ha='center',
                        weight='bold', color='#7f8c8d', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.7))

            ax.set_title(f"{label}\n({subtitle})", fontsize=11, weight='bold')
            ax.set_xlabel("Training Sample Size ($N$)", weight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower right', frameon=True)

        axes[0].set_ylabel("ROC-AUC Score", weight='bold')
        plt.suptitle("LEARNING CURVE DIAGNOSTICS: IDENTIFYING OVERFITTING & UNDERFITTING", weight='bold', fontsize=13)
        file_path = os.path.join(self.output_dir, "02_overfitting_underfitting_learning_curves.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        return file_path

    def plot_ensemble_comparison(self, eval_results: Dict[str, Dict[str, Any]], df_leaderboard: pd.DataFrame) -> str:
        """Figure 3: Ensemble Paradigm Comparison (ROC Curves, PR Curves, Metric Radar)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

        palette = {
            "Decision Tree (Unpruned)": "#95a5a6",
            "Decision Tree (Pruned Depth=4)": "#7f8c8d",
            "Bagging (100 Trees)": "#3498db",
            "Random Forest (100 Trees)": "#2980b9",
            "AdaBoost (100 Stumps)": "#f39c12",
            "XGBoost (100 Trees)": "#2ecc71",
            "LightGBM (100 Trees)": "#9b59b6"
        }

        # Left Panel: ROC Curves
        for name, res in eval_results.items():
            col = palette.get(name, "#333333")
            auc = res["roc_auc"]
            ax1.plot(res["fpr"], res["tpr"], label=f"{name} (AUC={auc:.3f})", color=col, linewidth=2.0)

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC=0.500)')
        ax1.set_title("Receiver Operating Characteristic (ROC Curves)", weight='bold')
        ax1.set_xlabel("False Positive Rate (1 - Specificity)", weight='bold')
        ax1.set_ylabel("True Positive Rate (Recall / Sensitivity)", weight='bold')
        ax1.legend(loc='lower right', frameon=True, fontsize=8.5)
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Right Panel: Precision-Recall Curves
        for name, res in eval_results.items():
            col = palette.get(name, "#333333")
            pr_auc = res["pr_auc"]
            ax2.plot(res["recall_curve"], res["precision_curve"], label=f"{name} (PR-AUC={pr_auc:.3f})", color=col, linewidth=2.0)

        ax2.set_title("Precision-Recall (PR Curves) for Subscriber Churn", weight='bold')
        ax2.set_xlabel("Recall (Coverage of Churned Subscribers)", weight='bold')
        ax2.set_ylabel("Precision (Accuracy of Churn Alerts)", weight='bold')
        ax2.legend(loc='lower left', frameon=True, fontsize=8.5)
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.suptitle("WEEK 13-14: ENSEMBLE METHODS SHOWDOWN (BAGGING vs. BOOSTING)", weight='bold', fontsize=13)
        file_path = os.path.join(self.output_dir, "03_ensemble_paradigm_comparison.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        return file_path

    def plot_xgboost_vs_lightgbm_performance(self, df_leaderboard: pd.DataFrame, df_imp: pd.DataFrame) -> str:
        """Figure 4: XGBoost vs. LightGBM Head-to-Head & Multi-Model Feature Importance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

        # Left Panel: Training Latency vs ROC-AUC Efficiency
        df_plot = df_leaderboard.copy()
        colors = ['#2ecc71' if 'XGBoost' in m else ('#9b59b6' if 'LightGBM' in m else '#3498db') for m in df_plot["Algorithm"]]

        bars = ax1.barh(df_plot["Algorithm"], df_plot["Train Time (ms)"], color=colors, alpha=0.85, height=0.55)
        for bar, time_ms, auc in zip(bars, df_plot["Train Time (ms)"], df_plot["ROC-AUC"]):
            ax1.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                     f"{time_ms:.1f} ms  |  AUC: {auc:.4f}",
                     va='center', weight='bold', fontsize=9)

        ax1.set_xlim(0, max(df_plot["Train Time (ms)"]) * 1.35)
        ax1.set_title("Computational Efficiency & Wall-Clock Training Latency", weight='bold')
        ax1.set_xlabel("Training Duration (Milliseconds, lower is faster)", weight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Right Panel: Top 8 Feature Importances Comparison
        top_features = df_imp.head(8).iloc[::-1]  # ascending for horizontal plot
        y_pos = np.arange(len(top_features))
        bar_h = 0.25

        ax2.barh(y_pos + bar_h, top_features["XGBoost (Gain)"], height=bar_h, color='#2ecc71', label='XGBoost (Gain)', alpha=0.85)
        ax2.barh(y_pos, top_features["LightGBM (Split Count)"], height=bar_h, color='#9b59b6', label='LightGBM (Splits)', alpha=0.85)
        ax2.barh(y_pos - bar_h, top_features["Random Forest (MDI)"], height=bar_h, color='#3498db', label='Random Forest (Gini)', alpha=0.85)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_features["Feature"], fontsize=9.5)
        ax2.set_title("Feature Importance Attribution (Top 8 Predictors)", weight='bold')
        ax2.set_xlabel("Normalized Relative Importance", weight='bold')
        ax2.legend(loc='lower right', frameon=True)
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.suptitle("XGBOOST vs. LIGHTGBM ARCHITECTURAL & ATTRIBUTION ANALYSIS", weight='bold', fontsize=13)
        file_path = os.path.join(self.output_dir, "04_xgboost_vs_lightgbm_performance.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        return file_path

    def plot_regularization_and_early_stopping(self, df_alpha: pd.DataFrame, df_lambda: pd.DataFrame,
                                               early_stop_data: Dict[str, Any]) -> str:
        """Figure 5: L1 vs L2 Regularization Impact and Early Stopping Inflection."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

        # Panel 1: L1 Regularization (reg_alpha) & Feature Sparsity
        ax1.plot(df_alpha["L1 Alpha"], df_alpha["Test Loss"], 'o-', color='#e74c3c', linewidth=2.2, label='Test Log-Loss')
        ax1.plot(df_alpha["L1 Alpha"], df_alpha["Train Loss"], 's--', color='#95a5a6', linewidth=1.8, label='Train Log-Loss')
        ax1_twin = ax1.twinx()
        ax1_twin.step(df_alpha["L1 Alpha"], df_alpha["Zero Importance Features"], color='#8e44ad', linewidth=2.0, where='mid', label='Zero-Weight Features')
        ax1_twin.set_ylabel("Pruned (Zero-Weight) Features", color='#8e44ad', weight='bold')
        ax1_twin.grid(False)

        ax1.set_xscale('log')
        ax1.set_title("L1 Regularization (Alpha)\nSparsity & Feature Selection", weight='bold')
        ax1.set_xlabel(r"L1 Penalty $\alpha$ (Log Scale)", weight='bold')
        ax1.set_ylabel("Log-Loss (Cross-Entropy)", weight='bold')
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Panel 2: L2 Regularization (reg_lambda) & Variance Smoothing
        ax2.plot(df_lambda["L2 Lambda"], df_lambda["Test Loss"], 'o-', color='#2980b9', linewidth=2.2, label='Test Log-Loss')
        ax2.plot(df_lambda["L2 Lambda"], df_lambda["Train Loss"], 's--', color='#95a5a6', linewidth=1.8, label='Train Log-Loss')
        ax2.set_xscale('log')
        ax2.set_title("L2 Regularization (Lambda)\nLeaf Weight Smoothing", weight='bold')
        ax2.set_xlabel(r"L2 Penalty $\lambda$ (Log Scale)", weight='bold')
        ax2.set_ylabel("Log-Loss", weight='bold')
        ax2.legend(loc='upper left', frameon=True)
        ax2.grid(True, linestyle=':', alpha=0.6)

        # Panel 3: Early Stopping Validation Dynamics
        rounds = early_stop_data["rounds"]
        train_l = early_stop_data["train_loss"]
        val_l = early_stop_data["val_loss"]
        opt_round = early_stop_data["optimal_round"]
        best_loss = early_stop_data["best_val_loss"]

        ax3.plot(rounds, train_l, color='#e67e22', linewidth=2.0, label='Train Loss (Continues Dropping)')
        ax3.plot(rounds, val_l, color='#2980b9', linewidth=2.2, label='Validation Loss')

        ax3.axvline(opt_round, color='#27ae60', linestyle='--', linewidth=2.0, label=f'Optimal Stop (Round {opt_round})')
        ax3.scatter([opt_round], [best_loss], color='#27ae60', s=120, zorder=5, edgecolor='black')

        # Shade overfitting penalty zone
        ax3.axvspan(opt_round, len(rounds), color='red', alpha=0.10, label='Overfitting Zone')

        ax3.set_title(f"Early Stopping Inflection\n(Best Val Loss: {best_loss:.4f} at Round {opt_round})", weight='bold')
        ax3.set_xlabel("Boosting Iterations (Trees Built)", weight='bold')
        ax3.set_ylabel("Log-Loss", weight='bold')
        ax3.legend(loc='upper right', frameon=True, fontsize=8.5)
        ax3.grid(True, linestyle=':', alpha=0.6)

        plt.suptitle("PREVENTING OVERFITTING: REGULARIZATION (L1/L2) & EARLY STOPPING", weight='bold', fontsize=13)
        file_path = os.path.join(self.output_dir, "05_regularization_l1_l2_impact.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        return file_path
