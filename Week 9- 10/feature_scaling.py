import numpy as np
from typing import Tuple, Dict

class StandardScalerCustom:
    """Standardizes features by removing the mean and scaling to unit variance: z = (x - mu) / sigma."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> 'StandardScalerCustom':
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class MinMaxScalerCustom:
    """Transforms features by scaling each feature to a given range, default [0, 1]."""
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, X: np.ndarray) -> 'MinMaxScalerCustom':
        X = np.asarray(X, dtype=float)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_[self.data_range_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        rescaled = (X - self.data_min_) / self.data_range_
        min_bound, max_bound = self.feature_range
        return rescaled * (max_bound - min_bound) + min_bound

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class RobustScalerCustom:
    """Scales features using statistics that are robust to outliers: median and Interquartile Range (IQR)."""
    def __init__(self):
        self.center_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> 'RobustScalerCustom':
        X = np.asarray(X, dtype=float)
        self.center_ = np.median(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        self.scale_ = q75 - q25
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X - self.center_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
