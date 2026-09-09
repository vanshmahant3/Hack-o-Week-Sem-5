import os
import urllib.request
import pandas as pd
import numpy as np

def prepare_dataset():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    raw_csv_path = os.path.join(data_dir, 'online_retail.csv')
    url = "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv"
    
    if not os.path.exists(raw_csv_path) or os.path.getsize(raw_csv_path) < 1000:
        print("Downloading UCI Online Retail Dataset from raw source...")
        urllib.request.urlretrieve(url, raw_csv_path)
        print(f"Downloaded raw dataset ({os.path.getsize(raw_csv_path) / (1024*1024):.2f} MB)")
    else:
        print("Raw dataset already downloaded.")

    # Load initial sample to build supplementary relational tables
    print("Parsing dataset and generating relational schemas...")
    df = pd.read_csv(raw_csv_path)
    
    # 1. Clean basic columns & extract unique Customers and Products
    df = df.dropna(subset=['CustomerID', 'Description'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    unique_customers = df['CustomerID'].unique()
    unique_stock_codes = df['StockCode'].unique()
    
    np.random.seed(42)
    
    # Table 2: Customer Profiles (customer_profiles.csv)
    tiers = ['Silver', 'Gold', 'Platinum', 'Diamond']
    segments = ['Consumer', 'Corporate', 'Home Office']
    age_groups = ['18-25', '26-35', '36-50', '51+']
    
    cust_df = pd.DataFrame({
        'CustomerID': unique_customers,
        'LoyaltyTier': np.random.choice(tiers, size=len(unique_customers), p=[0.4, 0.3, 0.2, 0.1]),
        'Segment': np.random.choice(segments, size=len(unique_customers), p=[0.5, 0.3, 0.2]),
        'AgeGroup': np.random.choice(age_groups, size=len(unique_customers)),
        'CreditRating': np.random.randint(580, 850, size=len(unique_customers))
    })
    
    # Table 3: Product Catalog & Metadata (product_catalog.csv)
    categories = ['Home Decor', 'Kitchenware', 'Office Supplies', 'Toys & Crafts', 'Fashion Accessories', 'Garden & Outdoor']
    suppliers = ['Apex Retail Ltd', 'Global Global Imports', 'Nordic Goods Co', 'Vanguard Trading', 'Heritage Crafts']
    
    prod_df = pd.DataFrame({
        'StockCode': unique_stock_codes,
        'Category': np.random.choice(categories, size=len(unique_stock_codes)),
        'Supplier': np.random.choice(suppliers, size=len(unique_stock_codes)),
        'CostMarginRatio': np.round(np.random.uniform(0.45, 0.75, size=len(unique_stock_codes)), 2)
    })
    
    # Save relational CSVs
    cust_path = os.path.join(data_dir, 'customer_profiles.csv')
    prod_path = os.path.join(data_dir, 'product_catalog.csv')
    
    cust_df.to_csv(cust_path, index=False)
    prod_df.to_csv(prod_path, index=False)
    
    print("Dataset setup complete!")
    print(f"- Sales Transactions: {len(df):,} rows -> {raw_csv_path}")
    print(f"- Customer Profiles: {len(cust_df):,} rows -> {cust_path}")
    print(f"- Product Catalog: {len(prod_df):,} rows -> {prod_path}")

if __name__ == '__main__':
    prepare_dataset()
