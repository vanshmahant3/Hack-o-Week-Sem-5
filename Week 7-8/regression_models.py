import numpy as np
from typing import Tuple, List, Dict

class LinearRegressionCustom:
    """Ordinary Least Squares (OLS) Linear Regression via Normal Equation: w = (X^T X)^-1 X^T y."""
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionCustom':
        n_samples = X.shape[0]
        # Add bias / intercept column of ones
        X_b = np.c_[np.ones((n_samples, 1)), X]
        # Closed-form Normal Equation: w = (X^T X)^-1 X^T y
        try:
            w_all = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ y)
        except np.linalg.LinAlgError:
            # Pseudo-inverse fallback if matrix is singular
            w_all = np.linalg.pinv(X_b) @ y
        
        self.intercept = float(w_all[0])
        self.weights = w_all[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.intercept


class PolynomialRegressionCustom:
    """Polynomial Regression expanding features up to degree d with feature conditioning."""
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.model = LinearRegressionCustom()
        self.poly_mean = None
        self.poly_std = None

    def _transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        cols = [X]
        for d in range(2, self.degree + 1):
            cols.append(X ** d)
        raw_poly = np.hstack(cols)
        if fit:
            self.poly_mean = np.mean(raw_poly, axis=0)
            self.poly_std = np.std(raw_poly, axis=0)
            self.poly_std[self.poly_std == 0] = 1.0
        return (raw_poly - self.poly_mean) / self.poly_std

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialRegressionCustom':
        X_poly = self._transform(X, fit=True)
        self.model.fit(X_poly, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_poly = self._transform(X, fit=False)
        return self.model.predict(X_poly)


class RidgeRegressionCustom:
    """Ridge Regression (L2 Regularization) via Closed-Form Solution: w = (X^T X + alpha * I)^-1 X^T y."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.weights = None
        self.intercept = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegressionCustom':
        n_samples, n_features = X.shape
        X_b = np.c_[np.ones((n_samples, 1)), X]
        
        # Identity matrix, but do NOT regularize the bias term (index 0)
        I = np.eye(n_features + 1)
        I[0, 0] = 0.0

        # Ridge closed-form equation
        w_all = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ (X_b.T @ y)
        self.intercept = float(w_all[0])
        self.weights = w_all[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.intercept

    @staticmethod
    def compute_coefficient_path(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
        """Computes coefficient paths across a spectrum of alpha values for visualization."""
        coef_list = []
        for a in alphas:
            ridge = RidgeRegressionCustom(alpha=a).fit(X, y)
            coef_list.append(ridge.weights)
        return np.array(coef_list)


class LassoRegressionCustom:
    """Lasso Regression (L1 Regularization) implemented via Coordinate Descent with Soft-Thresholding."""
    def __init__(self, alpha: float = 0.1, max_iter: int = 1000, tol: float = 1e-4):
        self.alpha = float(alpha)
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.intercept = None

    @staticmethod
    def _soft_threshold(z: float, gamma: float) -> float:
        """Soft-thresholding operator: S(z, gamma) = sign(z) * max(0, |z| - gamma)."""
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoRegressionCustom':
        n_samples, n_features = X.shape
        
        # Center y and X
        self.intercept = float(np.mean(y))
        y_centered = y - self.intercept
        
        # Initialize weights to zero
        w = np.zeros(n_features)
        
        # Precompute column norms squared: sum(x_ij^2)
        col_sq_sums = np.sum(X ** 2, axis=0)
        col_sq_sums[col_sq_sums == 0] = 1.0

        for _ in range(self.max_iter):
            w_old = w.copy()
            for j in range(n_features):
                # Calculate partial residual: r_j = y_centered - sum_{k != j} (w_k * x_k)
                residual = y_centered - (X @ w) + (w[j] * X[:, j])
                rho_j = float(np.dot(X[:, j], residual))
                
                # Update w_j via soft-thresholding
                w[j] = self._soft_threshold(rho_j, self.alpha * n_samples) / col_sq_sums[j]

            # Check convergence
            if np.max(np.abs(w - w_old)) < self.tol:
                break

        self.weights = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.intercept

    @staticmethod
    def compute_coefficient_path(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
        """Computes Lasso coefficient paths showing feature elimination/sparsity as alpha increases."""
        coef_list = []
        for a in alphas:
            lasso = LassoRegressionCustom(alpha=a, max_iter=300).fit(X, y)
            coef_list.append(lasso.weights)
        return np.array(coef_list)
