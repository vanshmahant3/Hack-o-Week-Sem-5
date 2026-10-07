import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class BiasVarianceEngine:
    """Rigorous Bias-Variance Decomposition and Overfitting/Underfitting Diagnostic Engine.

    Mathematical Foundations:
    Expected Test Error Decomposition:
        E[(y - f_hat(x))^2] = Bias^2(f_hat(x)) + Var(f_hat(x)) + sigma^2

    Where:
        Bias^2 = E_x[(E_D[f_hat(x)] - y)^2]
        Var    = E_x[E_D[(f_hat(x) - E_D[f_hat(x)])^2]]
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @staticmethod
    def prepare_feature_pipeline(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocesses streaming churn dataset into model-ready matrix."""
        feature_cols = [
            "daily_watch_hours", "account_age_months", "monthly_fee",
            "sub_plan", "devices_connected", "content_downloads_monthly",
            "skip_rate", "customer_service_tickets", "last_login_days_ago",
            "payment_method"
        ]
        X = df[feature_cols].copy()
        y = df["churned"].values

        categorical_cols = ["sub_plan", "payment_method"]
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
            ]
        )

        X_processed = preprocessor.fit_transform(X)
        cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
        final_feature_names = numeric_cols + cat_feature_names

        return X_processed, y, final_feature_names

    def decompose_bias_variance(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                depths: List[int] = [1, 2, 3, 4, 6, 8, 12, 16, 20],
                                n_bootstraps: int = 50) -> pd.DataFrame:
        """Performs bootstrap resampling to empirically isolate Bias^2, Variance, and Total Error.

        Evaluates Decision Trees across varying depths to visualize underfitting -> sweet spot -> overfitting.
        """
        np.random.seed(self.random_state)
        n_test = len(X_test)
        results = []

        for depth in depths:
            # Container for test set predictions across bootstrap iterations: shape (n_bootstraps, n_test)
            boot_preds = np.zeros((n_bootstraps, n_test))

            for b in range(n_bootstraps):
                # Bootstrap sampling with replacement
                boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
                X_boot = X_train[boot_idx]
                y_boot = y_train[boot_idx]

                tree = DecisionTreeClassifier(max_depth=depth, random_state=self.random_state + b)
                tree.fit(X_boot, y_boot)
                # Use predicted probability of positive class for continuous squared error decomposition
                boot_preds[b, :] = tree.predict_proba(X_test)[:, 1]

            # 1. Main prediction (mean prediction over all bootstrap models)
            mean_pred = np.mean(boot_preds, axis=0)

            # 2. Bias Squared: (E[f_hat] - y)^2
            bias_squared = float(np.mean((mean_pred - y_test) ** 2))

            # 3. Variance: E[(f_hat - E[f_hat])^2]
            variance = float(np.mean(np.var(boot_preds, axis=0)))

            # 4. Total Expected Error: Bias^2 + Variance
            total_error = bias_squared + variance

            # 5. Classification 0-1 error
            binary_preds = (boot_preds > 0.5).astype(int)
            overall_01_error = float(np.mean(binary_preds != y_test))

            regime = "Underfitting (High Bias)" if depth <= 2 else ("Sweet Spot (Optimal)" if depth in [4, 5, 6] else "Overfitting (High Var)")

            results.append({
                "max_depth": depth,
                "bias_squared": round(bias_squared, 4),
                "variance": round(variance, 4),
                "total_mse": round(total_error, 4),
                "error_01": round(overall_01_error, 4),
                "regime": regime
            })

        return pd.DataFrame(results)

    def compare_single_vs_bagging_bias_variance(self, X_train: np.ndarray, y_train: np.ndarray,
                                                X_test: np.ndarray, y_test: np.ndarray,
                                                n_bootstraps: int = 50) -> pd.DataFrame:
        """Compares Bias & Variance between a single unpruned tree, Bagging, and Random Forest.

        Empirically proves that Bagging reduces Variance while keeping Bias virtually unchanged.
        """
        np.random.seed(self.random_state)
        n_test = len(X_test)

        models = {
            "Single Decision Tree (Depth=15)": DecisionTreeClassifier(max_depth=15, random_state=self.random_state),
            "Bagging (50 Deep Trees)": BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=15),
                n_estimators=50, random_state=self.random_state
            ),
            "Random Forest (50 Trees)": RandomForestClassifier(
                max_depth=15, n_estimators=50, random_state=self.random_state
            )
        }

        records = []
        for name, model_template in models.items():
            boot_preds = np.zeros((n_bootstraps, n_test))

            for b in range(n_bootstraps):
                boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
                X_boot = X_train[boot_idx]
                y_boot = y_train[boot_idx]

                # Clone / fit
                from sklearn.base import clone
                m = clone(model_template)
                m.random_state = self.random_state + b
                m.fit(X_boot, y_boot)
                boot_preds[b, :] = m.predict_proba(X_test)[:, 1]

            mean_pred = np.mean(boot_preds, axis=0)
            bias_sq = float(np.mean((mean_pred - y_test) ** 2))
            var = float(np.mean(np.var(boot_preds, axis=0)))
            total_err = bias_sq + var
            acc = float(np.mean((mean_pred > 0.5).astype(int) == y_test))

            records.append({
                "Model Architecture": name,
                "Bias^2": round(bias_sq, 4),
                "Variance": round(var, 4),
                "Total MSE": round(total_err, 4),
                "Accuracy (%)": round(acc * 100, 2),
                "Variance Reduction vs Single Tree (%)": 0.0
            })

        df_res = pd.DataFrame(records)
        base_var = df_res.loc[0, "Variance"]
        df_res["Variance Reduction vs Single Tree (%)"] = [
            round((base_var - v) / base_var * 100, 1) if i > 0 else 0.0
            for i, v in enumerate(df_res["Variance"])
        ]
        return df_res

    def compute_learning_curves(self, X: np.ndarray, y: np.ndarray,
                                train_sizes: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0]) -> Dict[str, Any]:
        """Computes learning curves for Underfitting vs Overfitting vs Balanced models."""
        from sklearn.model_selection import learning_curve

        models = {
            "Underfitting (Depth 1 Stump)": DecisionTreeClassifier(max_depth=1, random_state=self.random_state),
            "Balanced (Random Forest)": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=self.random_state),
            "Overfitting (Unpruned Tree)": DecisionTreeClassifier(max_depth=20, min_samples_split=2, random_state=self.random_state)
        }

        curve_data = {}
        for label, model in models.items():
            sizes, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=5,
                scoring="roc_auc",
                random_state=self.random_state,
                n_jobs=-1
            )
            curve_data[label] = {
                "train_sizes": sizes,
                "train_mean": np.mean(train_scores, axis=1),
                "train_std": np.std(train_scores, axis=1),
                "val_mean": np.mean(val_scores, axis=1),
                "val_std": np.std(val_scores, axis=1)
            }

        return curve_data
