import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class DataLoader:
    """Class responsible for loading multi-table relational CSV datasets."""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_raw_data(self):
        retail_path = os.path.join(self.data_dir, 'online_retail.csv')
        customer_path = os.path.join(self.data_dir, 'customer_profiles.csv')
        catalog_path = os.path.join(self.data_dir, 'product_catalog.csv')

        if not all(os.path.exists(p) for p in [retail_path, customer_path, catalog_path]):
            raise FileNotFoundError("One or more required CSV files are missing from data directory.")

        print("\n[STEP 1] Loading Raw Relational Datasets from Disk...")
        retail_df = pd.read_csv(retail_path)
        cust_df = pd.read_csv(customer_path)
        catalog_df = pd.read_csv(catalog_path)

        print(f"  [+] Retail Transactions Loaded : {retail_df.shape[0]:,} rows x {retail_df.shape[1]} columns")
        print(f"  [+] Customer Profiles Loaded   : {cust_df.shape[0]:,} rows x {cust_df.shape[1]} columns")
        print(f"  [+] Product Catalog Loaded     : {catalog_df.shape[0]:,} rows x {catalog_df.shape[1]} columns")

        print("\n  Sample Raw Retail Transactions (First 3 Rows):")
        print(retail_df[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice', 'CustomerID', 'Country']].head(3).to_string(index=False))

        return retail_df, cust_df, catalog_df


class DataCleaner:
    """Class responsible for cleaning, transforming, and merging relational DataFrames."""
    def __init__(self, retail_df: pd.DataFrame, cust_df: pd.DataFrame, catalog_df: pd.DataFrame):
        self.retail_df = retail_df
        self.cust_df = cust_df
        self.catalog_df = catalog_df

    def clean_and_merge(self) -> pd.DataFrame:
        print("\n[STEP 2] Data Cleaning & String Standardization (Python Comprehensions)...")
        df = self.retail_df.copy()

        initial_rows = len(df)
        df = df.dropna(subset=['CustomerID', 'Description'])
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        cleaned_rows = len(df)
        print(f"  [+] Filtered invalid/missing entries: Removed {initial_rows - cleaned_rows:,} rows ({cleaned_rows:,} remaining)")

        # Date parsing & String standardization via List Comprehension
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df['Description'] = [str(desc).strip().upper() for desc in df['Description']]
        df['CustomerID'] = df['CustomerID'].astype(int)

        # Region Group Mapping via Dict Comprehension
        countries = df['Country'].unique()
        european_countries = {'United Kingdom', 'France', 'Germany', 'EIRE', 'Spain', 'Netherlands', 'Belgium', 'Switzerland', 'Portugal'}
        region_map = {c: ('Europe' if c in european_countries else 'International') for c in countries}
        df['RegionGroup'] = df['Country'].map(region_map)

        print("  [+] Executed Dict Comprehension: Country -> Region Group Mapping")
        print("  [+] Sample Region Mappings:", {k: region_map[k] for k in list(region_map.keys())[:5]})

        print("\n[STEP 3] Executing Multi-Table Relational Merges (SQL Inner Joins)...")
        merged_df = df.merge(self.cust_df, on='CustomerID', how='inner')
        merged_df = merged_df.merge(self.catalog_df, on='StockCode', how='inner')

        print(f"  [+] Merged Data Shape : {merged_df.shape[0]:,} rows x {merged_df.shape[1]} columns")
        print("  [+] Columns in Merged Dataset:", list(merged_df.columns))

        return merged_df


class SalesAnalyticsEngine:
    """Class executing NumPy vectorized operations, broadcasting, and Pandas GroupBy analytics."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def perform_vectorized_calculations(self) -> pd.DataFrame:
        print("\n[STEP 4A] NumPy Vectorized Computations & Broadcasting Operations...")
        
        # 1. NumPy Vectorized Operations for Revenue and Cost
        qty_array = self.df['Quantity'].to_numpy()
        price_array = self.df['UnitPrice'].to_numpy()
        margin_array = self.df['CostMarginRatio'].to_numpy()

        print(f"  [+] NumPy Quantity Array Shape : {qty_array.shape}")
        print(f"  [+] NumPy UnitPrice Array Shape: {price_array.shape}")

        total_revenue = np.multiply(qty_array, price_array)
        total_cost = np.multiply(total_revenue, margin_array)
        gross_profit = np.subtract(total_revenue, total_cost)

        self.df['TotalRevenue'] = np.round(total_revenue, 2)
        self.df['TotalCost'] = np.round(total_cost, 2)
        self.df['GrossProfit'] = np.round(gross_profit, 2)

        print(f"  [+] Sample Computed Total Revenue Vector (First 5): {total_revenue[:5]}")
        print(f"  [+] Sample Computed Gross Profit Vector (First 5) : {gross_profit[:5]}")

        # 2. NumPy Broadcasting: Z-Score Normalization of UnitPrice within Product Categories
        category_means = self.df.groupby('Category')['UnitPrice'].transform('mean').to_numpy()
        category_stds = self.df.groupby('Category')['UnitPrice'].transform('std').to_numpy()
        category_stds[category_stds == 0] = 1.0

        z_scores = np.divide(np.subtract(price_array, category_means), category_stds)
        self.df['Price_ZScore_Category'] = np.round(z_scores, 3)

        print("  [+] NumPy Broadcasting Z-Score Calculation Completed")
        print(f"  [+] Sample Z-Score Array Output (First 5): {z_scores[:5]}")

        return self.df

    def compute_rfm_scores(self) -> pd.DataFrame:
        print("\n[STEP 4B] NumPy Matrix Operations: Customer RFM Scoring...")
        rfm_df = self.df.groupby('CustomerID').agg({
            'InvoiceDate': 'max',
            'InvoiceNo': 'nunique',
            'TotalRevenue': 'sum'
        }).reset_index()

        rfm_df.columns = ['CustomerID', 'LastPurchase', 'Frequency', 'Monetary']
        max_date = self.df['InvoiceDate'].max()
        rfm_df['Recency'] = (max_date - rfm_df['LastPurchase']).dt.days

        r_norm = 1.0 - (rfm_df['Recency'] - rfm_df['Recency'].min()) / (rfm_df['Recency'].max() - rfm_df['Recency'].min() + 1e-5)
        f_norm = (rfm_df['Frequency'] - rfm_df['Frequency'].min()) / (rfm_df['Frequency'].max() - rfm_df['Frequency'].min() + 1e-5)
        m_norm = (rfm_df['Monetary'] - rfm_df['Monetary'].min()) / (rfm_df['Monetary'].max() - rfm_df['Monetary'].min() + 1e-5)

        feature_matrix = np.column_stack((r_norm, f_norm, m_norm))
        weights = np.array([0.3, 0.35, 0.35])

        rfm_df['RFM_Composite_Score'] = np.round(np.dot(feature_matrix, weights) * 100, 2)

        print(f"  [+] Formed Feature Matrix Shape: {feature_matrix.shape}")
        print("  [+] Executed Matrix Dot Product: Feature Matrix (N x 3) * Weights (3 x 1)")
        print("\n  Sample Customer RFM Composite Scores (Top 5 Customers):")
        print(rfm_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Composite_Score']].sort_values(by='RFM_Composite_Score', ascending=False).head(5).to_string(index=False))

        return rfm_df

    def get_groupby_summaries(self) -> dict:
        print("\n[STEP 5] Pandas Multi-Level GroupBy & Statistical Aggregations...")
        
        # GroupBy 1: Category Summary
        cat_summary = self.df.groupby('Category').agg({
            'TotalRevenue': ['sum', 'mean'],
            'GrossProfit': 'sum',
            'Quantity': 'sum'
        })
        cat_summary.columns = ['TotalRevenue_Sum', 'TotalRevenue_Mean', 'GrossProfit_Sum', 'Quantity_Sum']
        cat_summary = cat_summary.sort_values(by='TotalRevenue_Sum', ascending=False)

        print("\n  --- GroupBy 1: Performance by Product Category ---")
        print(cat_summary.to_string())

        # GroupBy 2: Loyalty Tier Summary
        tier_summary = self.df.groupby('LoyaltyTier').agg({
            'TotalRevenue': ['sum', 'mean', 'median'],
            'CustomerID': 'nunique'
        })
        tier_summary.columns = ['TotalRevenue_Sum', 'TotalRevenue_Mean', 'TotalRevenue_Median', 'UniqueCustomers']
        
        print("\n  --- GroupBy 2: Metrics by Customer Loyalty Tier ---")
        print(tier_summary.to_string())

        return {
            'cat_summary': cat_summary,
            'tier_summary': tier_summary
        }


class DataVisualizer:
    """Class responsible for generating Seaborn & Matplotlib data visualizations."""
    def __init__(self, df: pd.DataFrame, output_dir: str):
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    def generate_all_plots(self):
        print("\n[STEP 6] Generating Seaborn & Matplotlib Visualization Gallery...")
        
        # Plot 1: Heatmap
        plt.figure(figsize=(8, 6))
        numeric_cols = ['UnitPrice', 'Quantity', 'TotalRevenue', 'GrossProfit', 'CreditRating']
        sns.heatmap(self.df[numeric_cols].corr(), annot=True, cmap='Purples', fmt=".2f", linewidths=0.5)
        plt.title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
        plt.tight_layout()
        p1 = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(p1, dpi=300)
        plt.close()
        print(f"  [+] Saved Seaborn Correlation Heatmap        -> {p1}")

        # Plot 2: Violin Plot
        plt.figure(figsize=(9, 5))
        q95 = self.df['TotalRevenue'].quantile(0.95)
        sns.violinplot(data=self.df[self.df['TotalRevenue'] <= q95], x='LoyaltyTier', y='TotalRevenue', hue='LoyaltyTier', palette='Set2', legend=False)
        plt.title('Order Revenue Distribution by Loyalty Tier', fontsize=12, fontweight='bold')
        plt.tight_layout()
        p2 = os.path.join(self.output_dir, 'loyalty_tier_distribution.png')
        plt.savefig(p2, dpi=300)
        plt.close()
        print(f"  [+] Saved Seaborn Loyalty Tier Violin Plot   -> {p2}")

        # Plot 3: Bar Plot
        plt.figure(figsize=(11, 5))
        cat_s = self.df.groupby('Category')[['TotalRevenue', 'GrossProfit']].sum().reset_index().sort_values(by='TotalRevenue', ascending=False)
        x = np.arange(len(cat_s['Category']))
        w = 0.35
        plt.bar(x - w/2, cat_s['TotalRevenue']/1e3, w, label='Revenue ($K)', color='#702A8C')
        plt.bar(x + w/2, cat_s['GrossProfit']/1e3, w, label='Gross Profit ($K)', color='#25D366')
        plt.xticks(x, cat_s['Category'], rotation=15, ha='right')
        plt.title('Category Revenue & Gross Profit Breakdown', fontsize=12, fontweight='bold')
        plt.ylabel('Amount ($K)')
        plt.legend()
        plt.tight_layout()
        p3 = os.path.join(self.output_dir, 'category_performance.png')
        plt.savefig(p3, dpi=300)
        plt.close()
        print(f"  [+] Saved Matplotlib Category Bar Plot       -> {p3}")

        # Plot 4: Executive Subplot Dashboard
        m_df = self.df.groupby(self.df['InvoiceDate'].dt.to_period('M'))['TotalRevenue'].sum().reset_index()
        m_df['InvoiceDate'] = m_df['InvoiceDate'].astype(str)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
        ax1.plot(m_df['InvoiceDate'], m_df['TotalRevenue']/1e3, marker='o', color='#702A8C', linewidth=2)
        ax1.set_title('Monthly Revenue Trend ($K)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        top_c = self.df.groupby('Country')['TotalRevenue'].sum().nlargest(5).reset_index()
        sns.barplot(data=top_c, x='Country', y='TotalRevenue', hue='Country', palette='Blues_r', ax=ax2, legend=False)
        ax2.set_title('Top 5 Countries by Revenue', fontweight='bold')
        ax2.tick_params(axis='x', rotation=15)
        plt.suptitle('Global Sales Dashboard', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        p4 = os.path.join(self.output_dir, 'monthly_sales_trend.png')
        plt.savefig(p4, dpi=300)
        plt.close()
        print(f"  [+] Saved Executive Dashboard Line/Bar Subplots -> {p4}")
