import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

class LinearAlgebraEngine:
    """Core Linear Algebra Engine implementing vectors, dot products, covariance, eigenvalues, and PCA."""

    @staticmethod
    def l2_norm(v: np.ndarray) -> float:
        """Calculates Euclidean L2 norm: ||v|| = sqrt(sum(v_i^2))."""
        return float(np.sqrt(np.sum(np.square(v))))

    @staticmethod
    def dot_product(u: np.ndarray, v: np.ndarray) -> float:
        """Calculates dot product: u . v = sum(u_i * v_i)."""
        return float(np.sum(u * v))

    @staticmethod
    def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        """Calculates cosine similarity: cos(theta) = (u . v) / (||u|| * ||v||)."""
        norm_u = LinearAlgebraEngine.l2_norm(u)
        norm_v = LinearAlgebraEngine.l2_norm(v)
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return float(LinearAlgebraEngine.dot_product(u, v) / (norm_u * norm_v))

    @staticmethod
    def orthogonal_projection(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Projects vector u onto vector v: proj_v(u) = ((u . v) / ||v||^2) * v."""
        norm_v_sq = np.sum(np.square(v))
        if norm_v_sq == 0:
            return np.zeros_like(u)
        scalar_factor = LinearAlgebraEngine.dot_product(u, v) / norm_v_sq
        return scalar_factor * v

    @staticmethod
    def compute_pairwise_cosine_similarity(X: np.ndarray) -> np.ndarray:
        """Computes pairwise cosine similarity matrix between all sample row vectors."""
        # Row-wise L2 norms
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_normalized = X / norms
        # Dot product of normalized rows
        similarity_matrix = np.dot(X_normalized, X_normalized.T)
        return similarity_matrix

    @staticmethod
    def standardize_matrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mean-centers and standardizes matrix: X_std = (X - mu) / sigma."""
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma == 0] = 1.0
        X_std = (X - mu) / sigma
        return X_std, mu, sigma

    @staticmethod
    def compute_covariance_matrix(X_std: np.ndarray) -> np.ndarray:
        """Computes sample covariance matrix: Sigma = 1/(N - 1) * (X_std^T @ X_std)."""
        n_samples = X_std.shape[0]
        cov_matrix = np.dot(X_std.T, X_std) / (n_samples - 1)
        return cov_matrix

    @staticmethod
    def power_iteration(A: np.ndarray, num_iterations: int = 100) -> Tuple[float, np.ndarray]:
        """Power Iteration algorithm for computing the dominant eigenvalue and eigenvector."""
        # Initialize random non-zero vector
        np.random.seed(42)
        b_k = np.random.rand(A.shape[1])
        b_k = b_k / np.linalg.norm(b_k)

        for _ in range(num_iterations):
            # Matrix-vector multiplication: b_{k+1} = A * b_k
            b_k1 = np.dot(A, b_k)
            # Re-normalize
            norm = np.linalg.norm(b_k1)
            if norm == 0:
                break
            b_k = b_k1 / norm

        # Rayleigh quotient: lambda = (b^T * A * b) / (b^T * b)
        eigenvalue = float(np.dot(b_k.T, np.dot(A, b_k)))
        return eigenvalue, b_k

    @staticmethod
    def compute_eigendecomposition(cov_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solves the characteristic equation: Sigma * v_i = lambda_i * v_i.
        Returns sorted eigenvalues, sorted eigenvectors matrix, and explained variance ratios.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[idx]
        sorted_eigenvectors = eigenvectors[:, idx]

        total_variance = np.sum(sorted_eigenvalues)
        explained_variance_ratio = sorted_eigenvalues / total_variance

        return sorted_eigenvalues, sorted_eigenvectors, explained_variance_ratio

    @staticmethod
    def project_pca(X_std: np.ndarray, eigenvectors: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Projects standardized high-dimensional data onto the top k principal eigenvectors: Z = X_std @ V_k."""
        V_k = eigenvectors[:, :n_components]
        Z = np.dot(X_std, V_k)
        return Z

    @staticmethod
    def reconstruct_pca(Z: np.ndarray, eigenvectors: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reconstructs approximate data from lower-dimensional PCA projection: X_rec = Z @ V_k^T."""
        V_k = eigenvectors[:, :n_components]
        X_rec = np.dot(Z, V_k.T)
        return X_rec
