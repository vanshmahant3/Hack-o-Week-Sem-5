"""
Publication-Grade Visualizer for Week 15-16:
Neural Networks from Scratch & Model Deployment Basics.
Generates 5 High-Resolution (300 DPI) Diagnostic Figures in week 15-16/reports/.

Author: Dhanish Ladwani
Curriculum: Machine Learning & Pattern Recognition — Week 15-16
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Styling & typography
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0


class NeuralNetVisualizer:
    """
    Produces 5 publication-ready diagnostic figures illustrating Perceptrons,
    Activation Functions, Optimizer Dynamics, Gradient Checking, and Model Deployment.
    """
    
    def __init__(self, output_dir: str = "week 15-16/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_01_perceptron_vs_mlp_xor(
        self,
        and_model: Any,
        or_model: Any,
        xor_perceptron: Any,
        xor_mlp: Any,
        filename: str = "01_perceptron_vs_mlp_xor_decision_boundaries.png"
    ) -> str:
        """
        Figure 1: 4-panel comparison of logic gates and decision boundaries,
        proving why Single Perceptron fails on XOR and how 2-layer MLP resolves it.
        """
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), dpi=300)
        
        # Standard logic truth tables
        X_logic = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_and = np.array([0, 0, 0, 1])
        y_or = np.array([0, 1, 1, 1])
        y_xor = np.array([0, 1, 1, 0])
        
        x_mesh = np.linspace(-0.3, 1.3, 200)
        y_mesh = np.linspace(-0.3, 1.3, 200)
        XX, YY = np.meshgrid(x_mesh, y_mesh)
        grid_points = np.c_[XX.ravel(), YY.ravel()]
        
        panels = [
            ("AND Gate (Perceptron)", and_model, y_and, axes[0], "Linearly Separable (100% Acc)"),
            ("OR Gate (Perceptron)", or_model, y_or, axes[1], "Linearly Separable (100% Acc)"),
            ("XOR Gate (Single Perceptron)", xor_perceptron, y_xor, axes[2], "FAILED: Linear Inseparable"),
            ("XOR Gate (2-Layer MLP)", xor_mlp, y_xor, axes[3], "SOLVED: Non-Linear Boundary (100% Acc)")
        ]
        
        for title, model, y_true, ax, subtitle in panels:
            if model == xor_mlp:
                probs = model.predict_proba(grid_points).reshape(XX.shape)
                contour = ax.contourf(XX, YY, probs, levels=20, cmap="coolwarm", alpha=0.35)
                ax.contour(XX, YY, probs, levels=[0.5], colors=['#2c3e50'], linewidths=2.5, linestyles='--')
            else:
                # Perceptron linear boundary
                preds_grid = model.predict(grid_points).reshape(XX.shape)
                ax.contourf(XX, YY, preds_grid, levels=1, colors=['#3498db', '#e74c3c'], alpha=0.20)
                # Decision boundary line
                line_y = model.get_decision_boundary(x_mesh)
                if line_y is not None:
                    valid_mask = (line_y >= -0.3) & (line_y <= 1.3)
                    ax.plot(x_mesh[valid_mask], line_y[valid_mask], color='#2c3e50', linewidth=2.5, linestyle='--')
                    
            # Scatter truth points
            for idx, (pt, label) in enumerate(zip(X_logic, y_true)):
                color = '#e74c3c' if label == 1 else '#3498db'
                marker = 'o' if label == 1 else 's'
                ax.scatter(pt[0], pt[1], c=color, s=160, marker=marker, edgecolors='black', linewidth=1.5, zorder=5)
                ax.text(pt[0] + 0.05, pt[1] + 0.05, f"({pt[0]},{pt[1]})={label}", weight='bold', fontsize=9.5)
                
            ax.set_title(f"{title}\n{subtitle}", weight='bold', fontsize=11)
            ax.set_xlabel("$x_1$", weight='bold', fontsize=10)
            ax.set_ylabel("$x_2$", weight='bold', fontsize=10)
            ax.set_xlim(-0.3, 1.3)
            ax.set_ylim(-0.3, 1.3)
            ax.set_aspect('equal')
            
        fig.suptitle("The Perceptron Non-Linearity Limitation & Multi-Layer Resolution (XOR Problem)", weight='bold', fontsize=13, y=1.03)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_02_activation_functions(self, filename: str = "02_activation_functions_and_gradients.png") -> str:
        """
        Figure 2: 4-panel mathematical landscape of Sigmoid, Tanh, ReLU, and LeakyReLU
        alongside their analytical derivatives, highlighting the Vanishing Gradient problem.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)
        z = np.linspace(-5.0, 5.0, 500)
        
        # 1. Sigmoid
        sig = 1.0 / (1.0 + np.exp(-z))
        d_sig = sig * (1.0 - sig)
        axes[0, 0].plot(z, sig, color='#2980b9', linewidth=2.5, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')
        axes[0, 0].plot(z, d_sig, color='#e74c3c', linewidth=2.0, linestyle='--', label=r"$\sigma'(z) = \sigma(z)(1 - \sigma(z))$")
        axes[0, 0].axhline(0.25, color='#e74c3c', linestyle=':', alpha=0.7, label='Max Derivative = 0.25')
        axes[0, 0].axvspan(-5.0, -3.0, alpha=0.12, color='red')
        axes[0, 0].axvspan(3.0, 5.0, alpha=0.12, color='red', label='Saturation / Vanishing Zone')
        axes[0, 0].set_title("Sigmoid Activation & Vanishing Gradient", weight='bold', fontsize=11)
        axes[0, 0].set_ylim(-0.05, 1.05)
        axes[0, 0].legend(loc='upper left', frameon=True, fontsize=9)
        axes[0, 0].grid(True, linestyle='--', alpha=0.6)
        
        # 2. Tanh
        tanh = np.tanh(z)
        d_tanh = 1.0 - np.square(tanh)
        axes[0, 1].plot(z, tanh, color='#27ae60', linewidth=2.5, label=r'$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$')
        axes[0, 1].plot(z, d_tanh, color='#e67e22', linewidth=2.0, linestyle='--', label=r"$\tanh'(z) = 1 - \tanh^2(z)$")
        axes[0, 1].axhline(1.0, color='#e67e22', linestyle=':', alpha=0.7, label='Max Derivative = 1.0')
        axes[0, 1].axvspan(-5.0, -2.5, alpha=0.12, color='red')
        axes[0, 1].axvspan(2.5, 5.0, alpha=0.12, color='red', label='Saturation / Vanishing Zone')
        axes[0, 1].set_title("Tanh Activation (Zero-Centered, Saturation Risk)", weight='bold', fontsize=11)
        axes[0, 1].set_ylim(-1.05, 1.05)
        axes[0, 1].legend(loc='upper left', frameon=True, fontsize=9)
        axes[0, 1].grid(True, linestyle='--', alpha=0.6)
        
        # 3. ReLU
        relu = np.maximum(0.0, z)
        d_relu = np.where(z > 0.0, 1.0, 0.0)
        axes[1, 0].plot(z, relu, color='#8e44ad', linewidth=2.5, label=r'$\text{ReLU}(z) = \max(0, z)$')
        axes[1, 0].plot(z, d_relu, color='#d35400', linewidth=2.0, linestyle='--', label=r"$\text{ReLU}'(z) = \mathbb{I}_{z > 0}$")
        axes[1, 0].axvspan(-5.0, 0.0, alpha=0.12, color='purple', label='Dying ReLU Zone (Grad = 0)')
        axes[1, 0].set_title("ReLU (Non-Saturating, Constant Gradient for z > 0)", weight='bold', fontsize=11)
        axes[1, 0].set_ylim(-0.2, 5.2)
        axes[1, 0].legend(loc='upper left', frameon=True, fontsize=9)
        axes[1, 0].grid(True, linestyle='--', alpha=0.6)
        
        # 4. LeakyReLU
        lrelu = np.where(z > 0.0, z, 0.1 * z)
        d_lrelu = np.where(z > 0.0, 1.0, 0.1)
        axes[1, 1].plot(z, lrelu, color='#16a085', linewidth=2.5, label=r'$\text{LeakyReLU}(z) = \max(\alpha z, z), \alpha=0.1$')
        axes[1, 1].plot(z, d_lrelu, color='#c0392b', linewidth=2.0, linestyle='--', label=r"$\text{LeakyReLU}'(z) = 1 \text{ if } z>0 \text{ else } \alpha$")
        axes[1, 1].axhline(0.1, color='#c0392b', linestyle=':', alpha=0.7, label=r'Leaky Slope $\alpha = 0.1$')
        axes[1, 1].set_title("Leaky ReLU (Prevents Dead Neurons with Non-Zero Leak)", weight='bold', fontsize=11)
        axes[1, 1].set_ylim(-0.8, 5.2)
        axes[1, 1].legend(loc='upper left', frameon=True, fontsize=9)
        axes[1, 1].grid(True, linestyle='--', alpha=0.6)
        
        for ax in axes.flat:
            ax.set_xlabel("Input Pre-activation ($z$)", weight='bold', fontsize=10)
            ax.set_ylabel("Activation / Gradient Value", weight='bold', fontsize=10)
            
        fig.suptitle("Mathematical Taxonomy of Activation Functions & Gradient Flow Dynamics", weight='bold', fontsize=13, y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_03_optimizer_convergence(
        self,
        histories: Dict[str, Dict[str, List[float]]],
        filename: str = "03_optimizer_convergence_showdown.png"
    ) -> str:
        """
        Figure 3: Optimizer Convergence Showdown comparing SGD, Momentum, RMSprop, and Adam
        across training loss and validation accuracy on the clinical dataset.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
        
        palette = {
            "SGD (lr=0.02)": "#e74c3c",
            "SGD + Momentum (beta=0.9)": "#e67e22",
            "RMSprop (gamma=0.9)": "#3498db",
            "Adam (beta1=0.9, beta2=0.999)": "#2ecc71"
        }
        
        for name, hist in histories.items():
            epochs = np.arange(1, len(hist["train_loss"]) + 1)
            color = palette.get(name, "#34495e")
            ax1.plot(epochs, hist["train_loss"], label=name, color=color, linewidth=2.2)
            if "val_acc" in hist and len(hist["val_acc"]) > 0:
                ax2.plot(epochs, [acc * 100.0 for acc in hist["val_acc"]], label=name, color=color, linewidth=2.2)
                
        # Loss panel
        ax1.set_title("Training Loss Trajectory (Binary Cross-Entropy)", weight='bold', fontsize=11)
        ax1.set_xlabel("Training Epochs", weight='bold', fontsize=10)
        ax1.set_ylabel("Binary Cross-Entropy Loss", weight='bold', fontsize=10)
        ax1.legend(frameon=True, fontsize=9.5)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Accuracy panel
        ax2.set_title("Held-Out Validation Accuracy Progression", weight='bold', fontsize=11)
        ax2.set_xlabel("Training Epochs", weight='bold', fontsize=10)
        ax2.set_ylabel("Validation Accuracy (%)", weight='bold', fontsize=10)
        ax2.legend(frameon=True, loc='lower right', fontsize=9.5)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        fig.suptitle("Optimization Algorithm Convergence Benchmark on Cardiovascular Risk Data", weight='bold', fontsize=13, y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_04_training_and_gradient_check(
        self,
        mlp_model: Any,
        grad_errors: Dict[str, float],
        filename: str = "04_neural_net_training_and_gradient_check.png"
    ) -> str:
        """
        Figure 4: Neural Net Training History (Loss & Accuracy) alongside
        finite-difference gradient checking relative error bar chart verifying backprop.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
        
        # 1. Training history
        epochs = np.arange(1, len(mlp_model.history["train_loss"]) + 1)
        ax1.plot(epochs, mlp_model.history["train_loss"], color='#2980b9', linewidth=2.2, label='Train Loss')
        ax1.plot(epochs, mlp_model.history["val_loss"], color='#e74c3c', linewidth=2.2, linestyle='--', label='Validation Loss')
        ax1.set_title("Scratch Neural Network Training & Validation Loss", weight='bold', fontsize=11)
        ax1.set_xlabel("Epochs", weight='bold', fontsize=10)
        ax1.set_ylabel("Binary Cross-Entropy", weight='bold', fontsize=10)
        ax1.legend(frameon=True, fontsize=9.5)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 2. Gradient check relative error
        layers = list(grad_errors.keys())
        errors = [grad_errors[k] for k in layers]
        log_errors = [np.log10(max(1e-15, e)) for e in errors]
        
        colors = ['#27ae60' if e < 1e-7 else '#e74c3c' for e in errors]
        bars = ax2.barh(layers, log_errors, color=colors, height=0.55, edgecolor='black', alpha=0.85)
        ax2.axvline(-7.0, color='#e74c3c', linestyle='--', linewidth=2.0, label='Pass Threshold: log10(1e-7) = -7.0')
        
        ax2.set_title("Numerical Gradient Check Relative Errors ($||\\nabla_{ana} - \\nabla_{num}|| / ||\\cdot||$)", weight='bold', fontsize=11)
        ax2.set_xlabel("$\\log_{10}(\\text{Relative Error})$ [Lower is More Accurate]", weight='bold', fontsize=10)
        ax2.legend(loc='lower left', frameon=True, fontsize=9.5)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        for bar, err in zip(bars, errors):
            w = bar.get_width()
            ax2.text(w - 0.6, bar.get_y() + bar.get_height()/2.0, f"{err:.2e}", va='center', ha='right', color='white', weight='bold', fontsize=8.5)
            
        fig.suptitle("NumPy Multi-Layer Perceptron: Generalization Dynamics & Mathematical Verification", weight='bold', fontsize=13, y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_05_deployment_benchmark(
        self,
        df_bench: pd.DataFrame,
        latencies_data: Dict[str, Any],
        filename: str = "05_model_deployment_benchmark.png"
    ) -> str:
        """
        Figure 5: 3-panel Model Deployment Benchmark: Serialization Latency,
        Artifact Size, and Real-Time Inference Latency Distribution.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
        
        labels = [p.replace("Python ", "").replace(" (Protocol 5)", "\n(Pkl)").replace("Joblib ", "Joblib\n") for p in df_bench["Serialization Protocol"]]
        colors = ['#3498db', '#2ecc71', '#e67e22']
        
        # 1. Save & Load Latency
        x = np.arange(len(labels))
        w = 0.35
        ax1.bar(x - w/2, df_bench["Save Time (ms)"], width=w, label="Save Time (ms)", color='#2980b9', alpha=0.85)
        ax1.bar(x + w/2, df_bench["Load Time (ms)"], width=w, label="Load Time (ms)", color='#27ae60', alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_title("Serialization / Deserialization Latency", weight='bold', fontsize=11)
        ax1.set_ylabel("Latency (Milliseconds)", weight='bold', fontsize=10)
        ax1.legend(frameon=True, fontsize=9)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 2. File Size
        ax2.bar(labels, df_bench["File Size (KB)"], color=colors, alpha=0.85, edgecolor='black', width=0.5)
        for i, v in enumerate(df_bench["File Size (KB)"]):
            ax2.text(i, v + 0.1, f"{v:.2f} KB", ha='center', weight='bold', fontsize=9.5)
        ax2.set_title("Serialized Model Artifact Size on Disk", weight='bold', fontsize=11)
        ax2.set_ylabel("File Size (Kilobytes)", weight='bold', fontsize=10)
        ax2.set_ylim(0, max(df_bench["File Size (KB)"]) * 1.25)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # 3. Real-Time Inference Latency Distribution
        raw_lat = np.array(latencies_data["raw_latencies_us"]) / 1000.0  # convert to ms
        sns.histplot(raw_lat, kde=True, ax=ax3, color='#8e44ad', bins=30, alpha=0.6)
        p50 = latencies_data["median_p50_us"] / 1000.0
        p95 = latencies_data["p95_us"] / 1000.0
        p99 = latencies_data["p99_us"] / 1000.0
        
        ax3.axvline(p50, color='#27ae60', linestyle='--', linewidth=2.0, label=f"P50 (Median): {p50:.3f} ms")
        ax3.axvline(p95, color='#e67e22', linestyle='--', linewidth=2.0, label=f"P95: {p95:.3f} ms")
        ax3.axvline(p99, color='#e74c3c', linestyle='--', linewidth=2.0, label=f"P99: {p99:.3f} ms")
        
        ax3.set_title(f"Production Scoring Latency ({latencies_data['n_requests']} Requests)", weight='bold', fontsize=11)
        ax3.set_xlabel("Inference Latency (Milliseconds)", weight='bold', fontsize=10)
        ax3.set_ylabel("Request Frequency", weight='bold', fontsize=10)
        ax3.legend(frameon=True, fontsize=8.5)
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        fig.suptitle("Model Deployment Infrastructure & Operational Performance Evaluation", weight='bold', fontsize=13, y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path
