import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb

class RegularizationStudy:
    """Rigorous Investigation of L1 (Alpha) and L2 (Lambda) Regularization and Early Stopping.

    Mathematical Foundations:
    XGBoost Objective Function with Explicit Leaf Regularization:
        L^(t) = sum_{i=1}^n [ g_i f_t(x_i) + 0.5 h_i f_t^2(x_i) ] + Omega(f_t)

    Where Tree Complexity Penalty is:
        Omega(f_t) = gamma * T + 0.5 * lambda * sum_{j=1}^T (w_j^2) + alpha * sum_{j=1}^T |w_j|

    Optimal Leaf Weight:
        w_j* = - (G_j) / (H_j + lambda)   [smoothed by L2 lambda]
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def sweep_l1_alpha(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       alphas: List[float] = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]) -> pd.DataFrame:
        """Sweeps L1 regularization (reg_alpha) in XGBoost to measure weight sparsity and test generalization."""
        records = []
        for alpha in alphas:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.08,
                reg_alpha=alpha,
                reg_lambda=0.0,   # isolate L1
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            train_probs = model.predict_proba(X_train)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]

            train_loss = log_loss(y_train, train_probs)
            test_loss = log_loss(y_test, test_probs)
            test_auc = roc_auc_score(y_test, test_probs)
            train_acc = accuracy_score(y_train, (train_probs > 0.5).astype(int))
            test_acc = accuracy_score(y_test, (test_probs > 0.5).astype(int))

            # Feature sparsity: count features with zero importance
            importances = model.feature_importances_
            zero_features = int(np.sum(importances == 0.0))

            records.append({
                "L1 Alpha": alpha,
                "Train Loss": round(train_loss, 4),
                "Test Loss": round(test_loss, 4),
                "Test ROC-AUC": round(test_auc, 4),
                "Overfitting Gap (Loss Δ)": round(test_loss - train_loss, 4),
                "Zero Importance Features": zero_features,
                "Test Accuracy (%)": round(test_acc * 100, 2)
            })

        return pd.DataFrame(records)

    def sweep_l2_lambda(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        lambdas: List[float] = [0.0, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]) -> pd.DataFrame:
        """Sweeps L2 regularization (reg_lambda) in XGBoost to measure leaf weight smoothing and variance reduction."""
        records = []
        for lam in lambdas:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.08,
                reg_alpha=0.0,    # isolate L2
                reg_lambda=lam,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            train_probs = model.predict_proba(X_train)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]

            train_loss = log_loss(y_train, train_probs)
            test_loss = log_loss(y_test, test_probs)
            test_auc = roc_auc_score(y_test, test_probs)
            test_acc = accuracy_score(y_test, (test_probs > 0.5).astype(int))

            records.append({
                "L2 Lambda": lam,
                "Train Loss": round(train_loss, 4),
                "Test Loss": round(test_loss, 4),
                "Test ROC-AUC": round(test_auc, 4),
                "Overfitting Gap (Loss Δ)": round(test_loss - train_loss, 4),
                "Test Accuracy (%)": round(test_acc * 100, 2)
            })

        return pd.DataFrame(records)

    def demonstrate_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray,
                                   n_rounds: int = 250,
                                   early_stopping_rounds: int = 20) -> Dict[str, Any]:
        """Tracks iteration-by-iteration train and validation log-loss, capturing early stopping inflection."""
        # Unregularized deep model with learning rate 0.1 to clearly demonstrate overfitting over 250 rounds
        model_no_stop = xgb.XGBClassifier(
            n_estimators=n_rounds,
            max_depth=6,
            learning_rate=0.1,
            reg_alpha=0.0,
            reg_lambda=0.0,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1
        )
        model_no_stop.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        evals_result_full = model_no_stop.evals_result()

        train_loss_history = evals_result_full["validation_0"]["logloss"]
        val_loss_history = evals_result_full["validation_1"]["logloss"]

        optimal_round = int(np.argmin(val_loss_history)) + 1
        best_val_loss = float(np.min(val_loss_history))

        # Model with early stopping enabled
        model_early_stop = xgb.XGBClassifier(
            n_estimators=n_rounds,
            max_depth=6,
            learning_rate=0.1,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1
        )
        model_early_stop.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        stopped_round = model_early_stop.best_iteration + 1

        return {
            "rounds": list(range(1, n_rounds + 1)),
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "optimal_round": optimal_round,
            "best_val_loss": round(best_val_loss, 4),
            "stopped_round": stopped_round,
            "final_val_loss_at_250": round(val_loss_history[-1], 4),
            "overfitting_loss_penalty": round(val_loss_history[-1] - best_val_loss, 4)
        }
