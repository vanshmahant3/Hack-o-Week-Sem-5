import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Any, List, Tuple

class KMeansClusteringPipeline:
    """Manages K-Means clustering, centroid optimization via k-means++,
    and hyperparameter selection using the Elbow Method and Silhouette Analysis.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_k = None
        self.fitted_model = None

    def search_optimal_k(self, X: np.ndarray, k_range: range = range(2, 11)) -> Dict[str, Any]:
        """Calculates Inertia (WCSS) and Silhouette Scores across k in k_range."""
        inertias = []
        silhouettes = []
        k_values = list(k_range)

        for k in k_values:
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=self.random_state)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            sil_score = silhouette_score(X, labels)
            silhouettes.append(sil_score)

        best_idx = int(np.argmax(silhouettes))
        self.best_k = k_values[best_idx]

        return {
            'k_values': k_values,
            'inertias': inertias,
            'silhouettes': silhouettes,
            'optimal_k': self.best_k,
            'best_silhouette': silhouettes[best_idx]
        }

    def fit_predict(self, X: np.ndarray, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fits KMeans model with selected or optimal k and returns labels and cluster centers."""
        n_clusters = k if k is not None else (self.best_k if self.best_k else 5)
        self.fitted_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=self.random_state)
        labels = self.fitted_model.fit_predict(X)
        centers = self.fitted_model.cluster_centers_
        return labels, centers

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assigns new customer feature vectors to their nearest centroid."""
        if self.fitted_model is None:
            raise ValueError("KMeans model is not fitted yet.")
        return self.fitted_model.predict(X)
