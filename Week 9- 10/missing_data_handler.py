import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class MissingDataDiagnostics:
    """Analyzes missing data prevalence, patterns, and missingness mechanisms."""

    @staticmethod
    def inspect_missingness(df: pd.DataFrame) -> pd.DataFrame:
        """Returns a formatted summary table of missing values, counts, and percentages."""
        null_counts = df.isnull().sum()
        null_pcts = (null_counts / len(df)) * 100.0
        dtypes = df.dtypes

        summary = pd.DataFrame({
            'Column': df.columns,
            'Data_Type': dtypes.values,
            'Missing_Count': null_counts.values,
            'Missing_Percent': null_pcts.values
        })
        summary = summary[summary['Missing_Count'] > 0].sort_values(by='Missing_Percent', ascending=False).reset_index(drop=True)
        return summary

    @staticmethod
    def missingness_correlation(df: pd.DataFrame) -> pd.DataFrame:
        """Computes pairwise correlation between missingness indicator vectors (I_missing)."""
        null_indicators = df.isnull().astype(int)
        cols_with_nulls = [c for c in df.columns if df[c].isnull().sum() > 0]
        if len(cols_with_nulls) < 2:
            return pd.DataFrame()
        return null_indicators[cols_with_nulls].corr()


class MissingDataImputer:
    """Applies central tendency, frequent category, and indicator-augmented imputation.
    Strictly prevents data leakage by computing statistics solely on training partitions.
    """
    def __init__(self, strategy_num: str = 'median', strategy_cat: str = 'mode', add_indicators: bool = True):
        self.strategy_num = strategy_num  # 'mean', 'median'
        self.strategy_cat = strategy_cat  # 'mode'
        self.add_indicators = add_indicators
        self.num_stats = {}
        self.cat_stats = {}
        self.imputed_columns = []

    def fit(self, df: pd.DataFrame) -> 'MissingDataImputer':
        """Computes imputation statistics strictly on the training DataFrame."""
        self.num_stats = {}
        self.cat_stats = {}
        self.imputed_columns = [col for col in df.columns if df[col].isnull().sum() > 0]

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if self.strategy_num == 'median':
                        self.num_stats[col] = float(df[col].median())
                    elif self.strategy_num == 'mean':
                        self.num_stats[col] = float(df[col].mean())
                else:
                    # Categorical mode (most frequent)
                    mode_val = df[col].mode()
                    self.cat_stats[col] = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies pre-computed training statistics to any DataFrame (train, validation, or test)."""
        df_out = df.copy()

        # 1. Optionally add missingness indicator columns (e.g. income_was_missing)
        if self.add_indicators:
            for col in self.imputed_columns:
                if col in df_out.columns:
                    df_out[f"{col}_was_missing"] = df_out[col].isnull().astype(int)

        # 2. Impute numerical columns
        for col, fill_val in self.num_stats.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(fill_val)

        # 3. Impute categorical columns
        for col, fill_val in self.cat_stats.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(fill_val)

        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
