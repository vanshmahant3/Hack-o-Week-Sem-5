import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, List

class DiagnosticVisualizer:
    """Generates publication-grade diagnostic plots for Missing Data, Scaling, CV, and Metrics."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['axes.edgecolor'] = '#cbd5e1'
        plt.rcParams['axes.linewidth'] = 0.8

    def plot_missing_data_diagnostics(self, missing_df: pd.DataFrame, raw_df: pd.DataFrame, imputed_df: pd.DataFrame):
        """Plot 1: Missing Data Prevalence and Pre- vs. Post-Imputation Distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

        # Subplot 1: Missing Percentages
        cols = missing_df['Column'].tolist()
        pcts = missing_df['Missing_Percent'].tolist()
        bars = ax1.barh(cols, pcts, color='#38bdf8', edgecolor='#0284c7', height=0.55)
        ax1.axvline(x=5.0, color='#f59e0b', linestyle='--', linewidth=1.5, label='5% Missingness Threshold')
        for bar in bars:
            w = bar.get_width()
            ax1.text(w + 0.2, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=9, fontweight='bold')
        ax1.set_xlim(0, max(pcts) + 2.5)
        ax1.set_title("Missing Data Prevalence by Feature", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Missing Percentage (%)", fontweight='bold')
        ax1.legend(frameon=True, loc='lower right')

        # Subplot 2: Distribution Preservation (Credit Score Pre vs Post Imputation)
        raw_vals = raw_df['credit_score'].dropna()
        imp_vals = imputed_df['credit_score']
        sns.kdeplot(raw_vals, ax=ax2, label='Pre-Imputation (Observed Only)', color='#0284c7', linewidth=2.5)
        sns.kdeplot(imp_vals, ax=ax2, label='Post-Imputation (Median Imputed)', color='#10b981', linewidth=2.0, linestyle='--')
        ax2.set_title("Distribution Preservation: Credit Score", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Credit Score (FICO)", fontweight='bold')
        ax2.set_ylabel("Density", fontweight='bold')
        ax2.legend(frameon=True, loc='upper left')

        plt.suptitle("Missing Data Diagnostics & Imputation Integrity", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "01_missing_data_diagnostics.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_feature_scaling_distributions(self, raw_feat: np.ndarray, std_feat: np.ndarray, minmax_feat: np.ndarray, robust_feat: np.ndarray, feat_name: str = "Revolving Balance"):
        """Plot 2: Scaling Showdown - Raw vs. StandardScaler vs. MinMaxScaler vs. RobustScaler."""
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

        # 1. Raw Skewed Feature
        sns.histplot(raw_feat, bins=35, kde=True, ax=axes[0], color='#64748b')
        axes[0].set_title(f"1. Raw Skewed\n({feat_name})", fontsize=10, fontweight='bold')
        axes[0].set_xlabel("Raw Value ($)")

        # 2. StandardScaler
        sns.histplot(std_feat, bins=35, kde=True, ax=axes[1], color='#3b82f6')
        axes[1].set_title("2. StandardScaler\n(Zero Mean, Unit Std)", fontsize=10, fontweight='bold')
        axes[1].set_xlabel("Z-Score (std devs)")

        # 3. MinMaxScaler
        sns.histplot(minmax_feat, bins=35, kde=True, ax=axes[2], color='#10b981')
        axes[2].set_title("3. MinMaxScaler\n(Bounded in [0, 1])", fontsize=10, fontweight='bold')
        axes[2].set_xlabel("Normalized Value")

        # 4. RobustScaler
        sns.histplot(robust_feat, bins=35, kde=True, ax=axes[3], color='#8b5cf6')
        axes[3].set_title("4. RobustScaler\n(Median / IQR Resilient)", fontsize=10, fontweight='bold')
        axes[3].set_xlabel("Robust Normalized Units")

        plt.suptitle("Feature Scaling Showdown: Distribution Morphology Under Alternative Scalers", fontsize=13, fontweight='bold', y=1.04)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "02_feature_scaling_distributions.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_cross_validation_fold_variance(self, cv_summary: Dict[str, Any]):
        """Plot 3: Stratified K-Fold Cross-Validation Metrics & Variance Stability."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

        fold_metrics = cv_summary['fold_metrics']
        n_splits = cv_summary['n_splits']
        folds = [f"Fold {i+1}" for i in range(n_splits)]

        # Subplot 1: Trajectory across folds
        ax1.plot(folds, [v * 100 for v in fold_metrics['accuracy']], marker='o', linewidth=2, color='#3b82f6', label='Accuracy (%)')
        ax1.plot(folds, [v * 100 for v in fold_metrics['roc_auc']], marker='s', linewidth=2, color='#10b981', label='ROC-AUC (%)')
        ax1.plot(folds, [v * 100 for v in fold_metrics['f1']], marker='^', linewidth=2, color='#8b5cf6', label='F1-Score (%)')
        ax1.set_title(f"Stratified {n_splits}-Fold Performance Trajectory", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Metric Score (%)", fontweight='bold')
        ax1.legend(frameon=True, loc='lower right')

        # Subplot 2: Mean +/- Std Error Bar Comparison
        metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        means = [cv_summary[f"mean_{k}"] * 100 for k in metrics_keys]
        stds = [cv_summary[f"std_{k}"] * 100 for k in metrics_keys]

        x_pos = np.arange(len(labels))
        ax2.bar(x_pos, means, yerr=stds, capsize=6, color=['#3b82f6', '#06b6d4', '#f59e0b', '#8b5cf6', '#10b981'], alpha=0.85)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax2.text(i, m / 2, f"{m:.1f}%\n(±{s:.2f}%)", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, fontweight='bold')
        ax2.set_title("Cross-Validation Stability: Mean ± 1 Standard Deviation", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Percentage (%)", fontweight='bold')
        ax2.set_ylim(0, 105)

        plt.suptitle("Validation Integrity: Stratified Cross-Validation Stability Analysis", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "03_cross_validation_fold_variance.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_confusion_matrix_and_cost_analysis(self, cm: np.ndarray, cost_data: Dict[str, Any]):
        """Plot 4: Confusion Matrix Heatmap and Economic Decision Cost Curve."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

        # Subplot 1: Annotated Confusion Matrix Heatmap
        labels = ['Non-Defaulter (Good)', 'Defaulter (Bad)']
        total = np.sum(cm)
        annot_matrix = np.empty_like(cm, dtype=object)
        for r in range(2):
            for c in range(2):
                annot_matrix[r, c] = f"{cm[r, c]:,}\n({(cm[r, c]/total)*100:.1f}%)"

        sns.heatmap(cm, annot=annot_matrix, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax1, cbar=False)
        ax1.set_title("Model Confusion Matrix (Default Prediction)", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Predicted Class", fontweight='bold')
        ax1.set_ylabel("Actual Ground Truth", fontweight='bold')

        # Subplot 2: Business Cost & Net Profit Curve
        ths = cost_data['thresholds']
        profits_m = cost_data['profits'] / 1e6  # in Millions
        opt_th = cost_data['optimal_threshold']
        max_profit_m = cost_data['max_profit'] / 1e6

        ax2.plot(ths, profits_m, color='#10b981', linewidth=2.8, label='Net Portfolio Profit ($M)')
        ax2.axvline(x=opt_th, color='#ef4444', linestyle='--', linewidth=2.0, label=f'Optimal Threshold (tau = {opt_th:.2f})')
        ax2.axvline(x=0.50, color='#94a3b8', linestyle=':', linewidth=1.5, label='Default Threshold (tau = 0.50)')
        ax2.scatter([opt_th], [max_profit_m], color='#ef4444', s=70, zorder=5)
        ax2.text(opt_th + 0.02, max_profit_m, f"Max Profit:\n${max_profit_m:.2f}M", fontweight='bold', color='#0f172a', fontsize=9)

        ax2.set_title("Financial Decision Curve: Threshold vs. Net Profit", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Classification Decision Cutoff Threshold (tau)", fontweight='bold')
        ax2.set_ylabel("Net Portfolio Profit ($ Millions)", fontweight='bold')
        ax2.legend(frameon=True, loc='lower left')

        plt.suptitle("Classification Economics: Error Distribution & Business Threshold Optimization", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "04_confusion_matrix_and_cost_analysis.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")

    def plot_roc_and_precision_recall_curves(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float,
                                             precisions: np.ndarray, recalls: np.ndarray, pr_thresholds: np.ndarray):
        """Plot 5: Receiver Operating Characteristic (ROC) and Precision-Recall (PR) Curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

        # Subplot 1: ROC Curve
        ax1.plot(fpr, tpr, color='#2563eb', linewidth=2.8, label=f'Model ROC Curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', linewidth=1.8, label='Random Chance (AUC = 0.500)')
        ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight='bold')
        ax1.set_xlabel("False Positive Rate (FPR)", fontweight='bold')
        ax1.set_ylabel("True Positive Rate (TPR / Recall)", fontweight='bold')
        ax1.legend(frameon=True, loc='lower right')

        # Subplot 2: Precision-Recall Curve
        ax2.plot(recalls, precisions, color='#8b5cf6', linewidth=2.8, label='Precision-Recall Trajectory')
        
        # Optimal F1 threshold marker
        f1_vals = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
        best_f1_idx = np.argmax(f1_vals)
        best_rec = recalls[best_f1_idx]
        best_prec = precisions[best_f1_idx]
        best_th = pr_thresholds[best_f1_idx]
        best_f1 = f1_vals[best_f1_idx]

        ax2.scatter([best_rec], [best_prec], color='#ef4444', s=80, zorder=5, label=f'Max F1 ({best_f1:.3f}) at tau={best_th:.2f}')
        ax2.set_title("Precision-Recall Curve (Imbalanced Risk Assessment)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Recall (Coverage of Defaulters)", fontweight='bold')
        ax2.set_ylabel("Precision (Accuracy of Risk Alerts)", fontweight='bold')
        ax2.legend(frameon=True, loc='lower left')

        plt.suptitle("Discriminative Power: ROC-AUC & Precision-Recall Performance", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "05_roc_and_precision_recall_curves.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [+] Saved: {out_path}")
