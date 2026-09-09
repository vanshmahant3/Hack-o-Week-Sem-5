import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class FeatureEngineer:
    """Generates interaction ratios, applies log transformations to skewed variables,
    performs one-hot encoding, and conducts IQR outlier capping.
    """
    def __init__(self, log_transform_cols: List[str] = None, outlier_cols: List[str] = None):
        self.log_transform_cols = log_transform_cols or ['revolving_balance', 'annual_income']
        self.outlier_cols = outlier_cols or ['revolving_balance', 'annual_income']
        self.iqr_bounds = {}
        self.encoded_categorical_cols = []
        self.fitted_columns = []

    def _create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives financial interaction ratios and non-linear risk signals."""
        df_out = df.copy()

        # 1. Total Debt-to-Income (DTI) Proxy
        income_safe = np.maximum(df_out['annual_income'], 1000.0)
        df_out['debt_to_income_ratio'] = (
            (df_out['loan_amount'] * 0.35 + df_out['revolving_balance'] * 0.15) / income_safe
        ).round(4)

        # 2. Loan-to-Income Ratio
        df_out['loan_to_income_ratio'] = (df_out['loan_amount'] / income_safe).round(4)

        # 3. Estimated Annual Interest Burden
        df_out['annual_interest_burden'] = (df_out['loan_amount'] * (df_out['interest_rate'] / 100.0)).round(2)

        # 4. Total Line Utilization Intensity
        df_out['line_utilization_intensity'] = (
            df_out['total_credit_lines'] * df_out['revolving_utilization']
        ).round(3)

        # 5. High Delinquency Risk Flag
        df_out['is_high_delinquency'] = (df_out['delinquencies_2yrs'] >= 2).astype(int)

        return df_out

    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Learns outlier IQR bounds and categorical dummy columns strictly from training data."""
        df_temp = self._create_domain_features(df)

        # Compute IQR outlier bounds
        self.iqr_bounds = {}
        for col in self.outlier_cols:
            if col in df_temp.columns and pd.api.types.is_numeric_dtype(df_temp[col]):
                q1 = float(np.percentile(df_temp[col].dropna(), 25))
                q3 = float(np.percentile(df_temp[col].dropna(), 75))
                iqr = q3 - q1
                lower = max(0.0, q1 - 1.5 * iqr)
                upper = q3 + 1.5 * iqr
                self.iqr_bounds[col] = (lower, upper)

        # Determine one-hot encoded structure
        cat_cols = [c for c in df_temp.columns if not pd.api.types.is_numeric_dtype(df_temp[c])]
        df_encoded = pd.get_dummies(df_temp, columns=cat_cols, drop_first=True, dtype=float)
        self.fitted_columns = [c for c in df_encoded.columns if c != 'is_default']

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame using pre-computed training parameters."""
        df_out = self._create_domain_features(df)

        # 1. IQR Outlier Winsorization / Capping
        for col, (lower, upper) in self.iqr_bounds.items():
            if col in df_out.columns:
                df_out[col] = np.clip(df_out[col], lower, upper)

        # 2. Log Transformations for Skewed Features: log(1 + x)
        for col in self.log_transform_cols:
            if col in df_out.columns:
                df_out[f"log_{col}"] = np.log1p(np.maximum(0, df_out[col]))

        # 3. One-Hot Categorical Encoding
        cat_cols = [c for c in df_out.columns if not pd.api.types.is_numeric_dtype(df_out[c])]
        df_out = pd.get_dummies(df_out, columns=cat_cols, drop_first=True, dtype=float)

        # 4. Align columns with training fitted columns
        for col in self.fitted_columns:
            if col not in df_out.columns:
                df_out[col] = 0.0

        # Preserve target if present
        cols_to_keep = [c for c in self.fitted_columns]
        if 'is_default' in df.columns:
            df_out['is_default'] = df['is_default'].values
            cols_to_keep.append('is_default')

        return df_out[cols_to_keep]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
