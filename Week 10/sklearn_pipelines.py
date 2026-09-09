import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from typing import List, Tuple

class ScikitLearnWorkflowPipelines:
    """Demonstrates modular Scikit-Learn Pipeline and ColumnTransformer architectures."""

    @staticmethod
    def build_preprocessing_pipeline(numerical_cols: List[str], categorical_cols: List[str],
                                    scaler_type: str = 'robust', apply_pca: bool = False,
                                    n_pca_components: int = 2) -> Pipeline:
        """Constructs an end-to-end, leak-free preprocessing Pipeline using ColumnTransformer."""
        # 1. Numerical Sub-Pipeline
        scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])

        # 2. Categorical Sub-Pipeline
        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])

        # 3. Combined ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, numerical_cols),
                ('cat', cat_pipeline, categorical_cols)
            ],
            remainder='drop'
        )

        # 4. Master Pipeline (with optional PCA dimensionality reduction)
        steps = [('preprocessor', preprocessor)]
        if apply_pca:
            steps.append(('pca', PCA(n_components=n_pca_components, random_state=42)))

        return Pipeline(steps=steps)

    @staticmethod
    def build_clustering_pipeline(preprocessor_pipeline: Pipeline, clustering_estimator) -> Pipeline:
        """Chains an arbitrary unsupervised clustering estimator to a preprocessing pipeline."""
        full_pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor_pipeline),
            ('clusterer', clustering_estimator)
        ])
        return full_pipeline
