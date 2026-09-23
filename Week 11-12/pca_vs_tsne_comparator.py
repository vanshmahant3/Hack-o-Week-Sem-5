import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.manifold import trustworthiness
from typing import Dict, List, Tuple, Any

class PCAvsTSNEComparator:
    """Direct Comparative Benchmark: PCA vs. t-SNE.

    Evaluates:
    - 2D Silhouette Score
    - 2D k-NN Classification Accuracy (Separability)
    - Trustworthiness (Local Neighborhood Preservation)
    - Global Distance Rank Correlation (Spearman rho)
    - Invertibility & Reconstruction capability
    - Out-of-Sample projection support
    - Computational wall-clock runtime & Big-O complexity
    """

    @staticmethod
    def evaluate_2d_embeddings(X_high: np.ndarray,
                               X_pca_2d: np.ndarray,
                               X_tsne_2d: np.ndarray,
                               labels: np.ndarray,
                               pca_time: float,
                               tsne_time: float) -> pd.DataFrame:
        """Computes comprehensive quantitative metrics comparing PCA and t-SNE 2D projections."""
        # 1. Silhouette Score on 2D coordinates
        sil_pca = float(silhouette_score(X_pca_2d, labels))
        sil_tsne = float(silhouette_score(X_tsne_2d, labels))

        # 2. 2D k-NN Classification Accuracy (k=5, 5-fold CV)
        knn = KNeighborsClassifier(n_neighbors=5)
        acc_pca = float(np.mean(cross_val_score(knn, X_pca_2d, labels, cv=5)))
        acc_tsne = float(np.mean(cross_val_score(knn, X_tsne_2d, labels, cv=5)))

        # 3. Trustworthiness (Local neighborhood preservation, k=5)
        trust_pca = float(trustworthiness(X_high, X_pca_2d, n_neighbors=5))
        trust_tsne = float(trustworthiness(X_high, X_tsne_2d, n_neighbors=5))

        # 4. Global Distance Correlation (Spearman rho on subsampled pairwise distances)
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_high), size=min(300, len(X_high)), replace=False)
        d_high = pdist(X_high[sample_idx])
        d_pca = pdist(X_pca_2d[sample_idx])
        d_tsne = pdist(X_tsne_2d[sample_idx])

        rho_pca, _ = spearmanr(d_high, d_pca)
        rho_tsne, _ = spearmanr(d_high, d_tsne)

        comparison_data = [
            {
                "Evaluation Criterion": "2D Silhouette Score (Cluster Cohesion)",
                "PCA (Global Linear)": round(sil_pca, 4),
                "t-SNE (Local Non-Linear)": round(sil_tsne, 4),
                "Superior Approach": "t-SNE" if sil_tsne > sil_pca else "PCA"
            },
            {
                "Evaluation Criterion": "2D 5-NN Classification Accuracy (%)",
                "PCA (Global Linear)": f"{acc_pca * 100:.2f}%",
                "t-SNE (Local Non-Linear)": f"{acc_tsne * 100:.2f}%",
                "Superior Approach": "t-SNE" if acc_tsne > acc_pca else "PCA"
            },
            {
                "Evaluation Criterion": "Trustworthiness (Local Neighborhoods)",
                "PCA (Global Linear)": round(trust_pca, 4),
                "t-SNE (Local Non-Linear)": round(trust_tsne, 4),
                "Superior Approach": "t-SNE" if trust_tsne > trust_pca else "PCA"
            },
            {
                "Evaluation Criterion": "Global Distance Rank Correlation (Spearman Rho)",
                "PCA (Global Linear)": round(float(rho_pca), 4),
                "t-SNE (Local Non-Linear)": round(float(rho_tsne), 4),
                "Superior Approach": "PCA" if rho_pca > rho_tsne else "t-SNE"
            },
            {
                "Evaluation Criterion": "Wall-Clock Compute Time (seconds)",
                "PCA (Global Linear)": f"{pca_time:.4f}s",
                "t-SNE (Local Non-Linear)": f"{tsne_time:.4f}s",
                "Superior Approach": "PCA (Much Faster)"
            },
            {
                "Evaluation Criterion": "Out-of-Sample Projection (New Data Points)",
                "PCA (Global Linear)": "Yes: Exact matrix multiplication (W^T x)",
                "t-SNE (Local Non-Linear)": "No: Requires re-running gradient descent",
                "Superior Approach": "PCA"
            },
            {
                "Evaluation Criterion": "Reconstruction / Invertibility",
                "PCA (Global Linear)": "Yes: Analytical inverse transform (Z W^T)",
                "t-SNE (Local Non-Linear)": "No: Non-parametric irreversible embedding",
                "Superior Approach": "PCA"
            }
        ]

        return pd.DataFrame(comparison_data)
