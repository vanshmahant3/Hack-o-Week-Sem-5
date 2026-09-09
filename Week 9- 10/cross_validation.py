import numpy as np
from typing import List, Tuple, Dict, Any

class StratifiedKFoldCustom:
    """Stratified K-Fold cross-validator providing train/test indices to split data in train/test sets,
    preserving the percentage of samples for each class in every fold.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        np.random.seed(self.random_state)
        n_samples = len(y)
        classes = np.unique(y)
        
        # Partition indices by class
        class_indices = {}
        for c in classes:
            c_idx = np.where(y == c)[0]
            if self.shuffle:
                np.random.shuffle(c_idx)
            class_indices[c] = c_idx

        # Distribute into k buckets
        fold_buckets = [[] for _ in range(self.n_splits)]
        for c, idxs in class_indices.items():
            splits = np.array_split(idxs, self.n_splits)
            for fold_i, split_part in enumerate(splits):
                fold_buckets[fold_i].extend(split_part)

        # Generate train and validation index arrays for each fold
        all_indices = np.arange(n_samples)
        folds = []
        for fold_i in range(self.n_splits):
            val_idx = np.array(fold_buckets[fold_i])
            train_idx = np.setdiff1d(all_indices, val_idx)
            if self.shuffle:
                np.random.shuffle(train_idx)
            folds.append((train_idx, val_idx))

        return folds


class CrossValidationEngine:
    """Orchestrates leak-free k-fold cross-validation, computing fold metrics and stability statistics."""

    @staticmethod
    def cross_validate(model_factory, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, Any]:
        """Runs Stratified k-Fold CV on a given model factory function, tracking all evaluation metrics."""
        skf = StratifiedKFoldCustom(n_splits=n_splits, shuffle=True, random_state=42)
        folds = skf.split(X, y)

        metrics_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }

        # Local import to avoid circular dependency
        from model_evaluator import ModelMetrics

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Re-initialize fresh model instance
            model = model_factory()
            model.fit(X_train, y_train)

            # Predict labels & calibrated probabilities
            y_pred = model.predict(X_val)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)
                # Handle 2D probability outputs
                if len(y_proba.shape) == 2:
                    y_proba = y_proba[:, 1]
            else:
                y_proba = y_pred.astype(float)

            acc = ModelMetrics.accuracy_score(y_val, y_pred)
            prec = ModelMetrics.precision_score(y_val, y_pred)
            rec = ModelMetrics.recall_score(y_val, y_pred)
            f1 = ModelMetrics.f1_score(y_val, y_pred)
            auc = ModelMetrics.roc_auc_score(y_val, y_proba)

            metrics_history['accuracy'].append(acc)
            metrics_history['precision'].append(prec)
            metrics_history['recall'].append(rec)
            metrics_history['f1'].append(f1)
            metrics_history['roc_auc'].append(auc)

        # Compute mean and standard deviation
        summary = {
            'n_splits': n_splits,
            'fold_metrics': metrics_history,
            'mean_accuracy': float(np.mean(metrics_history['accuracy'])),
            'std_accuracy': float(np.std(metrics_history['accuracy'])),
            'mean_precision': float(np.mean(metrics_history['precision'])),
            'std_precision': float(np.std(metrics_history['precision'])),
            'mean_recall': float(np.mean(metrics_history['recall'])),
            'std_recall': float(np.std(metrics_history['recall'])),
            'mean_f1': float(np.mean(metrics_history['f1'])),
            'std_f1': float(np.std(metrics_history['f1'])),
            'mean_roc_auc': float(np.mean(metrics_history['roc_auc'])),
            'std_roc_auc': float(np.std(metrics_history['roc_auc']))
        }
        return summary
