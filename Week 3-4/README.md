# 📊 Global E-Commerce Data Analytics & Insights (Week 3–4)

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized_Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Plots-3776AB?style=for-the-badge&logo=python&logoColor=white)

An end-to-end Python data engineering and visualization suite utilizing **Object-Oriented Programming (OOP)**, **NumPy vectorization & broadcasting**, **Pandas multi-table joins & GroupBy aggregations**, and **Seaborn & Matplotlib analytics plots**.

---

## 📈 Executive Analytics Results & Key Performance Indicators (KPIs)

### Key Metrics Summary
| Metric | Value |
| :--- | :--- |
| **Total Transactions Analyzed** | **397,884** |
| **Unique Active Customers** | **4,338** |
| **Total Gross Revenue** | **$8,911,407.90** |
| **Total Gross Profit** | **$3,492,723.77** |
| **Overall Profit Margin** | **39.19%** |

---

## 📊 Product Category Performance Results (Pandas GroupBy)

| Product Category | Total Revenue ($) | Mean Order Revenue ($) | Total Gross Profit ($) | Total Quantity Sold |
| :--- | :--- | :--- | :--- | :--- |
| **Kitchenware** | $1,731,609.84 | $25.87 | $690,096.38 | 1,029,349 |
| **Home Decor** | $1,683,073.82 | $24.97 | $637,999.52 | 894,337 |
| **Garden & Outdoor** | $1,549,696.39 | $23.77 | $592,742.60 | 925,291 |
| **Toys & Crafts** | $1,335,525.03 | $20.73 | $543,305.30 | 785,800 |
| **Office Supplies** | $1,315,279.76 | $19.39 | $522,511.20 | 767,999 |
| **Fashion Accessories** | $1,296,223.06 | $19.65 | $506,068.76 | 765,036 |

---

## 👑 Customer Loyalty Tier Analysis Results

| Loyalty Tier | Total Revenue ($) | Mean Order Revenue ($) | Median Order Revenue ($) | Unique Customers |
| :--- | :--- | :--- | :--- | :--- |
| **Silver** | $2,752,293.63 | $20.16 | $9.95 | 1,738 |
| **Gold** | $2,648,796.88 | $22.57 | $10.20 | 1,281 |
| **Platinum** | $2,193,246.61 | $24.63 | $10.50 | 862 |
| **Diamond** | $1,317,070.78 | $30.09 | $14.85 | 457 |

---

## 🎯 Top High-Value Customer Composite RFM Scores (NumPy Dot Product)

Computed via matrix dot product $(N \times 3 \text{ normalized feature matrix}) \cdot (\text{weight vector})$:

| Customer ID | Recency (Days) | Frequency (Orders) | Monetary Value ($) | Composite RFM Score (0–100) |
| :--- | :--- | :--- | :--- | :--- |
| **14911** | 0 | 201 | $143,825.06 | **81.62** |
| **14646** | 1 | 73 | $280,206.02 | **77.03** |
| **18102** | 0 | 60 | $259,657.30 | **72.36** |
| **12748** | 0 | 209 | $33,719.73 | **69.21** |
| **17450** | 7 | 46 | $194,550.79 | **61.31** |

---

## 🖼️ Data Visualization Gallery & Analytical Insights

### 1. Feature Correlation Matrix (Seaborn Heatmap)
Displays pairwise linear correlation coefficients across prices, quantities, revenue, profit, and credit ratings.
![Correlation Heatmap](reports/correlation_heatmap.png)

### 2. Revenue Distribution across Loyalty Tiers (Seaborn Violin Plot)
Highlights order value density distributions and quartiles across Silver, Gold, Platinum, and Diamond loyalty tiers.
![Loyalty Tier Distribution](reports/loyalty_tier_distribution.png)

### 3. Category Revenue & Gross Profit Breakdown (Matplotlib Multi-Bar Plot)
Comparative bar chart showing total revenue vs net gross profit for each of the 6 core product categories.
![Category Performance](reports/category_performance.png)

### 4. Executive Sales Dashboard Subplots (Matplotlib Line & Seaborn Bar)
Displays monthly sales revenue trends over time alongside top 5 revenue-generating countries.
![Monthly Sales Trend Dashboard](reports/monthly_sales_trend.png)

---

## 🛠️ Technical Concepts & Architecture

### 1. Python Essentials (OOP & Comprehensions)
- **Object-Oriented Architecture** ([analytics_engine.py](analytics_engine.py)):
  - `DataLoader`: Loads relational CSV datasets into memory.
  - `DataCleaner`: Handles null values, string sanitization, ISO date parsing, and region mappings.
  - `SalesAnalyticsEngine`: Encapsulates vectorized matrix calculations and GroupBy aggregations.
  - `DataVisualizer`: Renders and saves Seaborn/Matplotlib figures.
- **Comprehensions**:
  - Dict comprehension mapping European vs. International country regions.
  - List comprehension standardizing product description strings.

### 2. NumPy (Arrays, Vectorization & Broadcasting)
- **Vectorized Operations**: Computes total revenue, cost, and gross profit using `np.multiply` and `np.subtract` without slow Python loops across 397,000+ rows.
- **Z-Score Normalization via Broadcasting**: Standardizes unit prices per product category using NumPy broadcasting: `(price_array - category_means) / category_stds`.
- **Matrix Dot Product**: Calculates composite Customer RFM (Recency, Frequency, Monetary) scores via `np.dot(feature_matrix, weights)`.

### 3. Pandas (DataFrames, Merging & GroupBy)
- **Relational Joins**: Multi-table SQL-style inner joins merging `online_retail.csv`, `customer_profiles.csv`, and `product_catalog.csv`.
- **Multi-level GroupBy**: Aggregates total revenues, gross profits, and item quantities across `Category` × `LoyaltyTier` and `RegionGroup` × `Segment`.

---

## 🚀 How to Run

1. Navigate to project folder:
   ```bash
   cd "week 3-4"
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the complete pipeline:
   ```bash
   python main.py
   ```

---

Made by [Dhanish Ladwani](https://github.com/dhanish0711/)
