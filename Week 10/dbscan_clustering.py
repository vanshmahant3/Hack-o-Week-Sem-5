import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, Any

class DBSCANClusteringPipeline:
    """Manages Density-Based Spatial Clustering of Applications with Noise (DBSCAN),
    automated epsilon determination via k-distance graphs, and outlier noise isolation.
    """
    def __init__(self, eps: float = 0.45, min_samples: int = 10):
        self.eps = eps
        self.min_samples = min_samples
        self.fitted_model = None

    def compute_k_distance_graph(self, X: np.ndarray, k: int = None) -> Tuple[np.ndarray, float]:
        """Calculates sorted k-nearest neighbor distances to detect the optimal epsilon knee point."""
        k_val = k if k is not None else self.min_samples
        nbrs = NearestNeighbors(n_neighbors=k_val).fit(X)
        distances, _ = nbrs.kneighbors(X)
        # Sort distance to the kth neighbor
        k_distances = np.sort(distances[:, k_val - 1])

        # Estimate knee point using second-derivative / percentile heuristic
        knee_idx = int(len(k_distances) * 0.94)
        estimated_eps = float(k_distances[knee_idx])
        return k_distances, estimated_eps

    def fit_predict(self, X: np.ndarray, eps: float = None, min_samples: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fits DBSCAN model and returns cluster labels and summary diagnostic statistics."""
        eps_val = eps if eps is not None else self.eps
        min_samp = min_samples if min_samples is not None else self.min_samples

        self.fitted_model = DBSCAN(eps=eps_val, min_samples=min_samp)
        labels = self.fitted_model.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        noise_pct = (n_noise / len(labels)) * 100.0

        summary = {
            'eps': eps_val,
            'min_samples': min_samp,
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'noise_percentage': noise_pct
        }
        return labels, summary
