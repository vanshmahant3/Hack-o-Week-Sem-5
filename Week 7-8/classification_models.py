import numpy as np
from typing import Tuple, List, Dict

class LogisticRegressionCustom:
    """Logistic Regression Classifier with Sigmoid activation and Gradient Descent optimization."""
    def __init__(self, lr: float = 0.05, max_iter: int = 500, tol: float = 1e-5):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.intercept = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Clamped for numerical stability
        z_clamped = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z_clamped))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionCustom':
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0.0
        self.loss_history = []

        for it in range(self.max_iter):
            # Forward pass: z = X w + b
            z = np.dot(X, self.weights) + self.intercept
            y_pred = self._sigmoid(z)

            # Compute Binary Cross-Entropy Loss
            eps = 1e-12
            loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
            self.loss_history.append(loss)

            # Gradient calculation: dL/dw = 1/N * X^T (y_pred - y)
            error = y_pred - y
            dw = (1.0 / n_samples) * np.dot(X.T, error)
            db = (1.0 / n_samples) * np.sum(error)

            # Parameter update
            self.weights -= self.lr * dw
            self.intercept -= self.lr * db

            if np.max(np.abs(self.lr * dw)) < self.tol:
                break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.intercept
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class KNNClassifierCustom:
    """K-Nearest Neighbors (KNN) non-parametric classifier using Euclidean Minkowski distance."""
    def __init__(self, k: int = 5):
        self.k = int(k)
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifierCustom':
        self.X_train = X
        self.y_train = y
        return self

    def _predict_single(self, x: np.ndarray) -> Tuple[int, float]:
        # Vectorized Euclidean distances: sqrt(sum((x - x_i)^2))
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # Find indices of k smallest distances
        k_indices = np.argpartition(distances, self.k)[:self.k]
        k_labels = self.y_train[k_indices]
        
        # Majority voting
        prob = float(np.mean(k_labels == 1))
        pred_class = 1 if prob >= 0.5 else 0
        return pred_class, prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = [self._predict_single(x)[0] for x in X]
        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = [self._predict_single(x)[1] for x in X]
        return np.array(probs)

    @staticmethod
    def tune_k_parameter(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, k_candidates: List[int]) -> Tuple[Dict[int, float], int]:
        """Evaluates accuracy across k values to identify the optimal bias-variance neighborhood size."""
        scores = {}
        for k in k_candidates:
            knn = KNNClassifierCustom(k=k).fit(X_train, y_train)
            preds = knn.predict(X_val)
            acc = float(np.mean(preds == y_val))
            scores[k] = acc

        best_k = max(scores, key=scores.get)
        return scores, best_k
