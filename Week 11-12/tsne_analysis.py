import time
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Any

class TSNEAnalysis:
    """t-Distributed Stochastic Neighbor Embedding (t-SNE) Analysis Engine.

    Covers non-linear manifold learning concepts:
    - High-dimensional Gaussian affinities (p_ij)
    - Low-dimensional Student-t distribution (q_ij, 1 d.o.f.)
    - Resolution of the 'crowding problem' via heavy-tailed repulsion
    - Perplexity parameter sensitivity (effective number of local neighbors)
    - Kullback-Leibler (KL) divergence minimization via gradient descent
    - PCA vs Random initialization
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.perplexity_embeddings = {}
        self.kl_divergences = {}
        self.fit_times = {}
        self.optimal_embedding = None

    def fit_perplexity_sweep(self, X_scaled: np.ndarray,
                             perplexities: List[int] = [5, 15, 30, 50],
                             max_iter: int = 1000) -> Dict[int, np.ndarray]:
        """Runs t-SNE across a spectrum of perplexities to evaluate local vs global geometry."""
        for perp in perplexities:
            start_t = time.time()
            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                max_iter=max_iter,
                init="pca",
                learning_rate="auto",
                random_state=self.random_state
            )
            embedding = tsne.fit_transform(X_scaled)
            duration = time.time() - start_t

            self.perplexity_embeddings[perp] = embedding
            self.kl_divergences[perp] = round(float(tsne.kl_divergence_), 4)
            self.fit_times[perp] = round(duration, 3)

        # Default optimal embedding set to perplexity 30 (standard best practice)
        self.optimal_embedding = self.perplexity_embeddings.get(30, list(self.perplexity_embeddings.values())[0])
        return self.perplexity_embeddings

    def compare_initialization_schemes(self, X_scaled: np.ndarray,
                                       perplexity: int = 30) -> Dict[str, Any]:
        """Compares PCA initialization vs Random initialization in t-SNE."""
        results = {}
        for init_type in ["pca", "random"]:
            start_t = time.time()
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                init=init_type,
                learning_rate="auto",
                random_state=self.random_state
            )
            emb = tsne.fit_transform(X_scaled)
            duration = time.time() - start_t

            results[init_type] = {
                "embedding": emb,
                "kl_divergence": round(float(tsne.kl_divergence_), 4),
                "duration_sec": round(duration, 3)
            }
        return results

    def get_perplexity_summary(self) -> pd.DataFrame:
        """Returns a summary table of t-SNE execution times and KL divergences."""
        rows = []
        for perp, kl in self.kl_divergences.items():
            rows.append({
                "Perplexity": perp,
                "KL Divergence": kl,
                "Compute Time (s)": self.fit_times[perp],
                "Geometric Focus": "Micro-clusters & isolated sub-islands" if perp < 15
                                  else "Balanced local & global structure" if perp <= 35
                                  else "Smoothed, continuous global manifold"
            })
        return pd.DataFrame(rows)
