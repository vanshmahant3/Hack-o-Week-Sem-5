import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from typing import Tuple, Dict, Any

class HierarchicalClusteringPipeline:
    """Manages Agglomerative Hierarchical Clustering, linkage comparisons
    (Ward, Complete, Average), and hierarchical tree dendrogram constructions.
    """
    def __init__(self, n_clusters: int = 5, linkage_method: str = 'ward'):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.fitted_model = None
        self.linkage_matrix = None

    def compute_linkage(self, X: np.ndarray, sample_size: int = 400) -> np.ndarray:
        """Computes hierarchical linkage matrix on a representative sub-sample for clean dendrogram visualization."""
        if len(X) > sample_size:
            np.random.seed(42)
            sub_idx = np.random.choice(len(X), size=sample_size, replace=False)
            X_sub = X[sub_idx]
        else:
            X_sub = X

        self.linkage_matrix = linkage(X_sub, method=self.linkage_method)
        return self.linkage_matrix

    def fit_predict(self, X: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """Fits AgglomerativeClustering model and returns cluster labels."""
        clusters = n_clusters if n_clusters is not None else self.n_clusters
        self.fitted_model = AgglomerativeClustering(n_clusters=clusters, linkage=self.linkage_method)
        labels = self.fitted_model.fit_predict(X)
        return labels
