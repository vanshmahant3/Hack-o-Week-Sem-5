import os
import sys
import pandas as pd

# Add current script directory to module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_dataset import prepare_dataset
from analytics_engine import DataLoader, DataCleaner, SalesAnalyticsEngine, DataVisualizer

def main():
    print("=" * 80)
    print("  GLOBAL E-COMMERCE DATA ANALYTICS & VISUALIZATION PIPELINE")
    print("  Stack: Python OOP | NumPy Broadcasting & Vectorization | Pandas GroupBy | Seaborn/Matplotlib")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    reports_dir = os.path.join(base_dir, 'reports')

    # Step 0: Ensure Datasets are Generated / Downloaded
    print("\n[STEP 0] Verifying Relational Dataset Files...")
    prepare_dataset()

    # Step 1: Load Data
    loader = DataLoader(data_dir=data_dir)
    retail_df, cust_df, catalog_df = loader.load_raw_data()

    # Step 2 & 3: Clean and Merge Data
    cleaner = DataCleaner(retail_df=retail_df, cust_df=cust_df, catalog_df=catalog_df)
    merged_df = cleaner.clean_and_merge()

    # Step 4 & 5: Execute Vectorized Calculations, Matrix RFM, and GroupBy Summaries
    engine = SalesAnalyticsEngine(df=merged_df)
    processed_df = engine.perform_vectorized_calculations()
    rfm_df = engine.compute_rfm_scores()
    summaries = engine.get_groupby_summaries()

    # Save processed merged CSV dataset
    processed_csv_path = os.path.join(data_dir, 'processed_ecommerce.csv')
    processed_df.to_csv(processed_csv_path, index=False)
    print(f"\n  [+] Exported Merged Dataset to Disk: {processed_csv_path} ({len(processed_df):,} rows)")

    # Step 6: Render Data Visualizations
    visualizer = DataVisualizer(df=processed_df, output_dir=reports_dir)
    visualizer.generate_all_plots()

    # Step 7: Executive Terminal Report Summary
    print("\n" + "=" * 80)
    print("  [STEP 7] EXECUTIVE DATA SUMMARY & ANALYTICS REPORT")
    print("=" * 80)
    print(f"  * Total Transactions Analyzed : {len(processed_df):,}")
    print(f"  * Total Unique Customers      : {processed_df['CustomerID'].nunique():,}")
    print(f"  * Total Gross Revenue         : ${processed_df['TotalRevenue'].sum():,.2f}")
    print(f"  * Total Gross Profit          : ${processed_df['GrossProfit'].sum():,.2f}")
    print(f"  * Overall Profit Margin       : {(processed_df['GrossProfit'].sum() / processed_df['TotalRevenue'].sum()) * 100:.2f}%\n")

    print("=" * 80)
    print("  [+] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  [+] Visualization reports saved in: {reports_dir}")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
