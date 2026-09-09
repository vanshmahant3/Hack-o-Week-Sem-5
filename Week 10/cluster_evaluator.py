import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, Any

class ClusterEvaluator:
    """Computes unsupervised clustering validity indices:
    Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Variance Ratio.
    """
    @staticmethod
    def evaluate_clustering(X: np.ndarray, labels: np.ndarray, model_name: str = "Clustering Model") -> Dict[str, Any]:
        """Calculates comprehensive validity metrics for a given clustering partition."""
        unique_labels = set(labels)
        # Filter noise points (-1) for DBSCAN evaluation if necessary
        valid_mask = labels != -1
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if n_clusters < 2 or np.sum(valid_mask) < n_clusters:
            return {
                'Model': model_name,
                'Clusters': n_clusters,
                'Silhouette': np.nan,
                'Davies_Bouldin': np.nan,
                'Calinski_Harabasz': np.nan,
                'Noise_Count': int(np.sum(labels == -1))
            }

        X_eval = X[valid_mask]
        labels_eval = labels[valid_mask]

        sil = float(silhouette_score(X_eval, labels_eval))
        dbi = float(davies_bouldin_score(X_eval, labels_eval))
        ch = float(calinski_harabasz_score(X_eval, labels_eval))
        noise_cnt = int(np.sum(labels == -1))

        return {
            'Model': model_name,
            'Clusters': n_clusters,
            'Silhouette': round(sil, 4),
            'Davies_Bouldin': round(dbi, 4),
            'Calinski_Harabasz': round(ch, 1),
            'Noise_Count': noise_cnt
        }

    @staticmethod
    def compare_models(X: np.ndarray, models_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Constructs an executive tournament leaderboard comparing multiple clustering partitions."""
        results = []
        for name, labels in models_dict.items():
            metrics = ClusterEvaluator.evaluate_clustering(X, labels, model_name=name)
            results.append(metrics)
        df_results = pd.DataFrame(results)
        return df_results
