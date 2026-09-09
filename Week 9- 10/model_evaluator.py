import numpy as np
from typing import Tuple, Dict, List, Any

class ModelMetrics:
    """Comprehensive mathematical evaluation metric suite for binary classification."""

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Computes 2x2 confusion matrix: [[TN, FP], [FN, TP]]."""
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
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
    def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = ModelMetrics.precision_score(y_true, y_pred)
        r = ModelMetrics.recall_score(y_true, y_pred)
        return float(2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @staticmethod
    def f_beta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 2.0) -> float:
        """F-beta score weighting recall beta-times as important as precision."""
        p = ModelMetrics.precision_score(y_true, y_pred)
        r = ModelMetrics.recall_score(y_true, y_pred)
        denom = (beta ** 2) * p + r
        if denom == 0:
            return 0.0
        return float((1.0 + beta ** 2) * (p * r) / denom)

    @staticmethod
    def roc_curve(y_true: np.ndarray, y_proba: np.ndarray, n_thresholds: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates False Positive Rate (FPR) and True Positive Rate (TPR) across thresholds."""
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

        fpr_arr = np.array(fpr_list)[::-1]
        tpr_arr = np.array(tpr_list)[::-1]
        th_arr = thresholds[::-1]
        return fpr_arr, tpr_arr, th_arr

    @staticmethod
    def roc_auc_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Trapezoidal integration under the ROC curve with NumPy 1.x & 2.x compatibility."""
        fpr_arr, tpr_arr, _ = ModelMetrics.roc_curve(y_true, y_proba)
        trap_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        if trap_fn is not None:
            auc = float(trap_fn(tpr_arr, fpr_arr))
        else:
            auc = float(np.sum((fpr_arr[1:] - fpr_arr[:-1]) * (tpr_arr[1:] + tpr_arr[:-1]) / 2.0))
        return abs(auc)

    @staticmethod
    def precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, n_thresholds: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates Precision and Recall across decision thresholds."""
        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        p_list = []
        r_list = []

        for th in thresholds:
            y_pred = (y_proba >= th).astype(int)
            p = ModelMetrics.precision_score(y_true, y_pred)
            r = ModelMetrics.recall_score(y_true, y_pred)
            p_list.append(p)
            r_list.append(r)

        return np.array(p_list), np.array(r_list), thresholds


class ThresholdOptimizer:
    """Evaluates business economic impact across classification decision thresholds."""

    @staticmethod
    def compute_cost_curve(y_true: np.ndarray, y_proba: np.ndarray,
                           profit_good_loan: float = 1200.0,
                           loss_default_loan: float = 6500.0) -> Dict[str, Any]:
        """Simulates lending portfolio net revenue across decision thresholds:
        - Defaulter identified (TP): Loss prevented.
        - Defaulter missed (FN): Incurs loss_default_loan ($6,500).
        - Good applicant approved (TN): Earns profit_good_loan ($1,200).
        - Good applicant rejected (FP): Opportunity loss of $1,200.
        """
        thresholds = np.linspace(0.05, 0.95, 91)
        profits = []
        approval_rates = []
        f1_scores = []

        for th in thresholds:
            # High risk if predicted probability >= th (so 1 = Reject loan, 0 = Approve loan)
            # Default prediction: y_pred = 1 (Defaulter)
            y_pred = (y_proba >= th).astype(int)
            
            # Loans approved are those predicted as NON-defaulters (y_pred == 0)
            approved = (y_pred == 0)
            approval_rate = float(np.mean(approved))
            
            # Approved good loans earn interest profit
            tp_good = np.sum((approved) & (y_true == 0))
            # Approved bad loans result in charge-off loss
            fp_bad = np.sum((approved) & (y_true == 1))
            
            net_pnl = (tp_good * profit_good_loan) - (fp_bad * loss_default_loan)
            f1 = ModelMetrics.f1_score(y_true, y_pred)

            profits.append(net_pnl)
            approval_rates.append(approval_rate * 100.0)
            f1_scores.append(f1)

        profits = np.array(profits)
        best_idx = int(np.argmax(profits))
        optimal_th = float(thresholds[best_idx])
        max_profit = float(profits[best_idx])

        return {
            'thresholds': thresholds,
            'profits': profits,
            'approval_rates': np.array(approval_rates),
            'f1_scores': np.array(f1_scores),
            'optimal_threshold': optimal_th,
            'max_profit': max_profit
        }
