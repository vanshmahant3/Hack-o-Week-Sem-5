import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Any

class PCAAnalysis:
    """Comprehensive Principal Component Analysis (PCA) Engine.

    Covers mathematical foundations:
    - Feature standardization (Z-score)
    - Covariance matrix eigendecomposition and SVD
    - Eigenvalue spectrum and explained variance ratios
    - Component loadings (eigenvector weights)
    - Inverse reconstruction and lossy compression RMSE
    - Out-of-sample projection
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.pca_full = None
        self.pca_2d = None
        self.X_scaled = None
        self.X_pca_2d = None

    def fit(self, X: np.ndarray) -> "PCAAnalysis":
        """Fits StandardScaler and full-component PCA on the feature matrix."""
        self.X_scaled = self.scaler.fit_transform(X)
        n_features = X.shape[1]

        # Full-rank PCA for scree analysis & variance retention
        self.pca_full = PCA(n_components=n_features, random_state=42)
        self.pca_full.fit(self.X_scaled)

        # 2D PCA for visual manifold projection
        self.pca_2d = PCA(n_components=2, random_state=42)
        self.X_pca_2d = self.pca_2d.fit_transform(self.X_scaled)

        return self

    def get_explained_variance_summary(self) -> pd.DataFrame:
        """Returns a table of eigenvalues, variance ratios, and cumulative variance."""
        eigenvalues = self.pca_full.explained_variance_
        var_ratios = self.pca_full.explained_variance_ratio_
        cumulative_var = np.cumsum(var_ratios)

        df_var = pd.DataFrame({
            "Principal Component": [f"PC{i+1}" for i in range(len(eigenvalues))],
            "Eigenvalue (Lambda)": np.round(eigenvalues, 4),
            "Explained Variance Ratio": np.round(var_ratios, 4),
            "Percentage (%)": np.round(var_ratios * 100, 2),
            "Cumulative Variance (%)": np.round(cumulative_var * 100, 2)
        })
        return df_var

    def get_threshold_components(self, thresholds: List[float] = [0.70, 0.80, 0.90, 0.95]) -> Dict[str, int]:
        """Calculates minimum principal components required to meet variance thresholds."""
        cum_var = np.cumsum(self.pca_full.explained_variance_ratio_)
        result = {}
        for th in thresholds:
            k = int(np.argmax(cum_var >= th) + 1)
            result[f"{int(th*100)}% Variance"] = k
        return result

    def get_loadings_dataframe(self) -> pd.DataFrame:
        """Returns the factor loading matrix (correlations between features and PCs)."""
        loadings = self.pca_full.components_.T * np.sqrt(self.pca_full.explained_variance_)
        columns = [f"PC{i+1}" for i in range(self.pca_full.n_components_)]
        df_loadings = pd.DataFrame(loadings, index=self.feature_names, columns=columns)
        return df_loadings.round(4)

    def evaluate_reconstruction_errors(self, X: np.ndarray) -> pd.DataFrame:
        """Computes reconstruction RMSE as component count increases from 1 to D."""
        errors = []
        n_features = X.shape[1]

        for k in range(1, n_features + 1):
            pca_k = PCA(n_components=k, random_state=42)
            Z_k = pca_k.fit_transform(self.X_scaled)
            X_recon_scaled = pca_k.inverse_transform(Z_k)
            X_recon = self.scaler.inverse_transform(X_recon_scaled)

            rmse = float(np.sqrt(np.mean((X - X_recon) ** 2)))
            cum_var = float(np.sum(pca_k.explained_variance_ratio_))

            errors.append({
                "Components (k)": k,
                "Reconstruction RMSE": round(rmse, 4),
                "Variance Retained (%)": round(cum_var * 100, 2)
            })

        return pd.DataFrame(errors)

    def transform_new_sample(self, sample_dict: Dict[str, float]) -> Tuple[float, float]:
        """Projects an out-of-sample observation onto the 2D PCA plane."""
        raw_vals = np.array([[sample_dict[col] for col in self.feature_names]])
        scaled_vals = self.scaler.transform(raw_vals)
        coords = self.pca_2d.transform(scaled_vals)[0]
        return (round(float(coords[0]), 3), round(float(coords[1]), 3))
