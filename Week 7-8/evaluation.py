import numpy as np
import pandas as pd
from typing import Tuple, Dict

class ModelEvaluator:
    """Evaluates Regression and Classification models with statistical and performance metrics."""

    # -------------------------------------------------------------------------
    # Regression Metrics
    # -------------------------------------------------------------------------
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(ModelEvaluator.mean_squared_error(y_true, y_pred)))

    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))

    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        return {
            'Model': model_name,
            'R2_Score': ModelEvaluator.r2_score(y_true, y_pred),
            'RMSE_Lakh': ModelEvaluator.root_mean_squared_error(y_true, y_pred),
            'MAE_Lakh': ModelEvaluator.mean_absolute_error(y_true, y_pred),
            'MSE': ModelEvaluator.mean_squared_error(y_true, y_pred)
        }

    # -------------------------------------------------------------------------
    # Classification Metrics
    # -------------------------------------------------------------------------
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Computes 2x2 confusion matrix: [[TN, FP], [FN, TP]]."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return np.array([[tn, fp], [fn, tp]])

    @staticmethod
    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    @staticmethod
    def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    @staticmethod
    def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = ModelEvaluator.precision_score(y_true, y_pred)
        r = ModelEvaluator.recall_score(y_true, y_pred)
        return float(2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @staticmethod
    def roc_curve_and_auc(y_true: np.ndarray, y_proba: np.ndarray, n_thresholds: int = 100) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calculates True Positive Rate (TPR) and False Positive Rate (FPR) over thresholds,
        and computes Area Under Curve (AUC) via trapezoidal numerical integration.
        """
        thresholds = np.linspace(0.0, 1.0, n_thresholds)
        tpr_list = []
        fpr_list = []

        total_pos = np.sum(y_true == 1)
        total_neg = np.sum(y_true == 0)

        for th in thresholds:
            y_pred = (y_proba >= th).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))

            tpr = tp / total_pos if total_pos > 0 else 0.0
            fpr = fp / total_neg if total_neg > 0 else 0.0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # Reverse to sort FPR ascendingly
        fpr_arr = np.array(fpr_list)[::-1]
        tpr_arr = np.array(tpr_list)[::-1]

        # Numerical integration for AUC: trapezoidal rule (NumPy 2.0+ uses trapezoid)
        trap_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        if trap_fn is not None:
            auc = float(trap_fn(tpr_arr, fpr_arr))
        else:
            auc = float(np.sum((fpr_arr[1:] - fpr_arr[:-1]) * (tpr_arr[1:] + tpr_arr[:-1]) / 2.0))
        return fpr_arr, tpr_arr, abs(auc)

    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str) -> Dict[str, float]:
        fpr, tpr, auc = ModelEvaluator.roc_curve_and_auc(y_true, y_proba)
        return {
            'Model': model_name,
            'Accuracy': ModelEvaluator.accuracy_score(y_true, y_pred),
            'Precision': ModelEvaluator.precision_score(y_true, y_pred),
            'Recall': ModelEvaluator.recall_score(y_true, y_pred),
            'F1_Score': ModelEvaluator.f1_score(y_true, y_pred),
            'ROC_AUC': auc
        }
