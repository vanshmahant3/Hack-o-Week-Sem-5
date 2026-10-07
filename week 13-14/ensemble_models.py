import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, log_loss
)
from sklearn.inspection import permutation_importance

import xgboost as xgb
import lightgbm as lgb

class EnsembleTournament:
    """Ensemble Methods Comparative Suite: Bagging vs. Boosting (XGBoost & LightGBM).

    Features:
    - Benchmarks Decision Tree, Bagging, Random Forest, AdaBoost, XGBoost, LightGBM.
    - Evaluates train/test accuracy, generalization gap, ROC-AUC, PR-AUC, Log-Loss.
    - Measures wall-clock training speed and inference throughput.
    - Extracts ROC curves, PR curves, and feature importance rankings.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.fitted_models: Dict[str, Any] = {}
        self.eval_results: Dict[str, Dict[str, Any]] = {}
        self._init_models()

    def _init_models(self):
        """Initializes benchmark model architectures."""
        self.models = {
            "Decision Tree (Unpruned)": DecisionTreeClassifier(
                max_depth=None,
                random_state=self.random_state
            ),
            "Decision Tree (Pruned Depth=4)": DecisionTreeClassifier(
                max_depth=4,
                random_state=self.random_state
            ),
            "Bagging (100 Trees)": BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=12),
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "Random Forest (100 Trees)": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                max_features="sqrt",
                oob_score=True,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "AdaBoost (100 Stumps)": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            "XGBoost (100 Trees)": xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,    # L1 regularization
                reg_lambda=1.0,   # L2 regularization
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1
            ),
            "LightGBM (100 Trees)": lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                num_leaves=31,
                learning_rate=0.08,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                reg_alpha=0.1,    # L1 regularization
                reg_lambda=1.0,   # L2 regularization
                verbosity=-1,
                random_state=self.random_state,
                n_jobs=-1
            )
        }

    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Fits all models and produces a comprehensive tournament leaderboard."""
        records = []

        for name, model in self.models.items():
            start_t = time.perf_counter()
            model.fit(X_train, y_train)
            train_duration = (time.perf_counter() - start_t) * 1000.0  # ms

            self.fitted_models[name] = model

            # In-sample (train) predictions
            y_train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)

            # Out-of-sample (test) predictions
            y_test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)

            # Probabilities for AUC & Log Loss
            if hasattr(model, "predict_proba"):
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_test_proba = model.decision_function(X_test)

            roc_auc = roc_auc_score(y_test, y_test_proba)
            pr_auc = average_precision_score(y_test, y_test_proba)
            loss = log_loss(y_test, y_test_proba)
            prec = precision_score(y_test, y_test_pred, zero_division=0)
            rec = recall_score(y_test, y_test_pred, zero_division=0)
            f1 = f1_score(y_test, y_test_pred, zero_division=0)

            # Generalization Gap (Overfitting metric: Train Acc - Test Acc)
            overfit_gap = train_acc - test_acc

            # Store metrics for curve visualizer
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)

            self.eval_results[name] = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "overfit_gap": overfit_gap,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "log_loss": loss,
                "train_time_ms": train_duration,
                "y_test_proba": y_test_proba,
                "fpr": fpr,
                "tpr": tpr,
                "precision_curve": precision_curve,
                "recall_curve": recall_curve
            }

            records.append({
                "Algorithm": name,
                "Train Acc (%)": round(train_acc * 100, 2),
                "Test Acc (%)": round(test_acc * 100, 2),
                "Overfit Gap (Δ%)": round(overfit_gap * 100, 2),
                "Precision (%)": round(prec * 100, 2),
                "Recall (%)": round(rec * 100, 2),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(roc_auc, 4),
                "PR-AUC": round(pr_auc, 4),
                "Log-Loss": round(loss, 4),
                "Train Time (ms)": round(train_duration, 1)
            })

        df_leaderboard = pd.DataFrame(records).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
        return df_leaderboard

    def get_feature_importances(self, feature_names: List[str]) -> pd.DataFrame:
        """Extracts and normalizes feature importance scores across flagship ensemble models."""
        importance_dict = {"Feature": feature_names}

        # 1. Random Forest (Gini Importance)
        if "Random Forest (100 Trees)" in self.fitted_models:
            rf = self.fitted_models["Random Forest (100 Trees)"]
            rf_imp = rf.feature_importances_
            importance_dict["Random Forest (MDI)"] = np.round(rf_imp / np.sum(rf_imp), 4)

        # 2. XGBoost (Gain Importance)
        if "XGBoost (100 Trees)" in self.fitted_models:
            xgb_m = self.fitted_models["XGBoost (100 Trees)"]
            xgb_imp = xgb_m.feature_importances_
            importance_dict["XGBoost (Gain)"] = np.round(xgb_imp / np.sum(xgb_imp), 4)

        # 3. LightGBM (Split Gain Importance)
        if "LightGBM (100 Trees)" in self.fitted_models:
            lgb_m = self.fitted_models["LightGBM (100 Trees)"]
            lgb_imp = lgb_m.feature_importances_
            importance_dict["LightGBM (Split Count)"] = np.round(lgb_imp / np.sum(lgb_imp), 4)

        df_imp = pd.DataFrame(importance_dict)
        df_imp["Ensemble Mean Importance"] = df_imp[[c for c in df_imp.columns if c != "Feature"]].mean(axis=1).round(4)
        return df_imp.sort_values(by="Ensemble Mean Importance", ascending=False).reset_index(drop=True)

    def study_random_forest_oob_convergence(self, X_train: np.ndarray, y_train: np.ndarray,
                                            X_test: np.ndarray, y_test: np.ndarray,
                                            estimator_counts: List[int] = [5, 10, 20, 40, 60, 80, 100, 140, 180]) -> pd.DataFrame:
        """Evaluates Out-of-Bag (OOB) error convergence vs holdout Test error as trees accumulate."""
        records = []
        for n in estimator_counts:
            rf = RandomForestClassifier(
                n_estimators=n,
                max_depth=10,
                max_features="sqrt",
                oob_score=True,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            oob_error = 1.0 - rf.oob_score_
            test_error = 1.0 - accuracy_score(y_test, rf.predict(X_test))

            records.append({
                "n_estimators": n,
                "oob_error": round(oob_error, 4),
                "test_error": round(test_error, 4),
                "oob_accuracy": round(rf.oob_score_ * 100, 2),
                "test_accuracy": round((1.0 - test_error) * 100, 2)
            })

        return pd.DataFrame(records)
