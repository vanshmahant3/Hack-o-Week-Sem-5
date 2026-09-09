# 🚗 Used Car Valuation & Deal Quality Intelligence Engine (Week 7–8)

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/NumPy-Linear_Algebra-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Plots-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

An end-to-end Machine Learning suite covering classical **Regression** and **Classification** algorithms applied directly to a realistic **Used Car Resale Market Dataset** (6,000 vehicle transactions).

---

## 🏗️ System Architecture & Machine Learning Pipeline

```mermaid
flowchart TD
    subgraph DataLayer["Dataset Layer (6,000 Used Vehicles)"]
        D["used_cars.csv<br>7 Automotive Features (Age, KM, Power, CC, Mileage, Owners, Seats)"]
    end

    subgraph DataPrep["Data Preprocessing & Train/Test Split (80/20)"]
        D --> PRE["StandardScaler: X_std = (X - mu) / sigma"]
    end

    subgraph RegressionEngine["Part 1: Regression Suite (Target: Resale Price in Lakhs ₹)"]
        PRE --> LR["1. Linear Regression (OLS & Normal Eq)<br>w = (X^T X)^-1 X^T y"]
        PRE --> PR["2. Polynomial Regression (d=2)<br>Captures Non-Linear Price Depreciation"]
        PRE --> RIDGE["3. Ridge Regression (L2 Penalty)<br>Loss = MSE + alpha ||w||_2^2"]
        PRE --> LASSO["4. Lasso Regression (L1 Penalty)<br>Loss = MSE + lambda ||w||_1 (Sparsity)"]
    end

    subgraph ClassificationEngine["Part 2: Classification Suite (Target: is_good_deal)"]
        PRE --> LOGREG["1. Logistic Regression<br>P(y=1) = 1 / (1 + e^-z)<br>Binary Cross-Entropy Loss"]
        PRE --> KNN["2. K-Nearest Neighbors (KNN)<br>Euclidean Minkowski Distance<br>k-Hyperparameter Search (k=1..29)"]
    end

    subgraph EvaluationTournament["Part 3: Evaluation & Benchmarks"]
        RegressionEngine --> EVAL_REG["R² Score, RMSE, MAE, Residuals"]
        ClassificationEngine --> EVAL_CLF["Confusion Matrix, Precision, Recall, F1, ROC-AUC"]
    end

    subgraph AdvisorCLI["Part 4: Real-World Decision Support"]
        EVAL_REG --> CLI["Interactive Car Valuation & Deal Advisor<br>(Evaluates Specific Buyer Scenarios)"]
        EVAL_CLF --> CLI
    end
```

---

## 🎯 Practical Outcome & Real-World Use Case

In the used automobile market, individual buyers and dealerships face asymmetrical information:
1. **Accurate Price Valuation**: What is the true fair market resale price of a car given its age, mileage, engine displacement, and power?
2. **Deal Quality Scoring**: Is a seller's asking price an **underpriced bargain (Great Deal)** or an **overpriced trap**?
3. **Depreciation Trajectory**: How rapidly will the vehicle lose value over subsequent ownership years?

Our platform combines **Polynomial Regression** (achieving an exceptional **$R^2 = 0.9318$**) with **Logistic Regression / KNN** (achieving **79.92% classification accuracy** and **0.835 ROC-AUC**) to deliver instant, automated valuation and deal advisories.

---

## 📊 Actionable Buyer Scenario Evaluation

```mermaid
flowchart LR
    A["Target Used Car Specs<br>(Age: 4 yrs, 38k km, 83 bhp, Asking: ₹ 5.40 Lakh)"] --> B["Polynomial Regression Engine"]
    A --> C["KNN & Logistic Classifiers"]
    B --> D["Predicted Fair Value: ₹ 8.95 Lakh<br>(Discount: ₹ 3.55 Lakh Below Market)"]
    C --> E["Deal Confidence Score: 75.9%"]
    D --> F{"Buyer Recommendation"}
    E --> F
    F --> G["RECOMMENDED BUY: EXCELLENT VALUE / GREAT DEAL"]
```

| Real-World Scenario | Vehicle Profile | Seller Asking Price | Model Fair Value Est | Confidence Score | Buyer Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Scenario 1: Daily Commuter Hatchback** | 4 yrs old, 38,000 km, 1.2L petrol, 83 bhp, 20.5 km/l, single owner | **₹ 5.40 Lakh** | **₹ 8.95 Lakh** | **75.9%** | `EXCELLENT VALUE / GREAT DEAL [RECOMMENDED BUY]` |
| **Scenario 2: Highway Family SUV** | 3 yrs old, 42,000 km, 2.0L diesel, 168 bhp, 15.2 km/l, 7-seater | **₹ 14.80 Lakh** | **₹ 14.03 Lakh** | **24.1%** | `OVERPRICED [NEGOTIATE DOWN OR WALK AWAY]` |
| **Scenario 3: High-Mileage Sedan** | 9 yrs old, 115,000 km, 1.5L petrol, 118 bhp, 3 previous owners | **₹ 4.10 Lakh** | **₹ 2.61 Lakh** | **13.8%** | `OVERPRICED [NEGOTIATE DOWN OR WALK AWAY]` |

---

## 🏆 Model Performance Leaderboard

### 1. Regression Tournament (Predicting Resale Price in ₹ Lakhs)

| Rank | Model Architecture | $R^2$ Score (Higher is Better) | RMSE (₹ Lakhs) | MAE (₹ Lakhs) | Key Takeaway |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **Polynomial Regression ($d=2$)** | **0.9318** | **₹ 1.29 Lakh** | **₹ 0.89 Lakh** | **Top Performer**: Captures non-linear early-year price depreciation. |
| 🥈 | **Ridge Regression ($L_2, \alpha=10$)** | **0.8467** | **₹ 1.94 Lakh** | **₹ 1.48 Lakh** | Prevents collinearity between engine displacement and horsepower. |
| 🥉 | **Linear Regression (OLS)** | **0.8467** | **₹ 1.94 Lakh** | **₹ 1.48 Lakh** | Closed-form Normal Equation baseline. |
| 4 | **Lasso Regression ($L_1, \alpha=0.08$)** | **0.8464** | **₹ 1.94 Lakh** | **₹ 1.46 Lakh** | Induces feature sparsity; isolates dominant pricing drivers. |

### 2. Classification Tournament (Predicting "Good Deal" Status)

| Rank | Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | $F_1$-Score (%) | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Logistic Regression** | **79.92%** | **82.08%** | **28.16%** | **41.93%** | **0.835** |
| 🥈 | **K-Nearest Neighbors ($k=29$)** | **77.00%** | **61.70%** | **28.16%** | **38.67%** | **0.811** |

---

## 🔬 Core Mathematical Principles & Algorithms

### 1. Linear Regression (Ordinary Least Squares)
* **Hypothesis**:
  $$\hat{y} = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^d w_i x_i + b$$
* **Closed-Form Normal Equation**:
  $$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### 2. Polynomial Regression & Bias-Variance Tradeoff
* **Basis Expansion**: Transforms features into higher degrees:
  $$\phi(\mathbf{x}) = [1, x_1, x_2, x_1^2, x_2^2, \dots]$$
* **Bias-Variance Dynamics**: Degree 1 underfits (high bias); Degree 2 achieves minimum test MSE; Degree 4+ overfits with erratic boundary predictions (high variance).

### 3. Regularization Showdown: Ridge ($L_2$) vs. Lasso ($L_1$)
* **Ridge Regression ($L_2$)**:
  $$\mathcal{L}_{\text{Ridge}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 + \alpha \|\mathbf{w}\|_2^2, \quad \mathbf{w} = (\mathbf{X}^T\mathbf{X} + \alpha \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$
  *Continuously shrinks coefficients toward zero without setting them to exact zero.*
* **Lasso Regression ($L_1$)**:
  $$\mathcal{L}_{\text{Lasso}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 + \lambda \|\mathbf{w}\|_1$$
  *Solved via Coordinate Descent with soft-thresholding ($S(z, \gamma) = \text{sign}(z)\max(0, |z|-\gamma)$); drives uninformative coefficients to exact zero for automatic feature selection.*

### 4. Classification: Logistic Regression & K-Nearest Neighbors (KNN)
* **Logistic Regression**:
  $$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$
  *Optimized using Binary Cross-Entropy loss via gradient descent.*
* **K-Nearest Neighbors (KNN)**:
  $$d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^d (x_j - x_{i,j})^2}, \quad \hat{y} = \text{mode}\left(\{y_{i} : \mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})\}\right)$$
  *Non-parametric instance-based voting; tuned across $k \in [1, 29]$ to balance localized noise with global trends.*

---

## 🖼️ Diagnostic Visualization Gallery

### 1. Used Car Price Depreciation: Linear vs. Polynomial Fits & Bias-Variance Curve
Visualizes non-linear age depreciation curve alongside training vs. testing MSE curves illustrating underfitting vs. overfitting.
![Depreciation Curve](reports/01_linear_vs_polynomial_depreciation.png)

### 2. Regularization Showdown: Ridge ($L_2$) vs. Lasso ($L_1$)
Compares continuous weight shrinkage in Ridge against exact feature sparsity in Lasso across logarithmic regularization strengths ($\alpha$).
![Ridge vs Lasso](reports/02_ridge_vs_lasso_regularization.png)

### 3. Logistic Regression Confusion Matrix & ROC-AUC Curve
Displays classification confusion matrix and Receiver Operating Characteristic (ROC) curve achieving an AUC of 0.835.
![Logistic Regression ROC](reports/03_logistic_regression_roc_confusion.png)

### 4. K-Nearest Neighbors (KNN) Decision Surface & $k$-Tuning
Shows 2D classification decision surface on Car Age vs. Engine Power alongside the neighborhood size ($k$) accuracy tuning curve.
![KNN Decision Boundary](reports/04_knn_decision_boundary_k_tuning.png)

### 5. Week 7–8 Model Tournament Leaderboard
Comprehensive benchmark comparison comparing all 6 models across $R^2$, RMSE, Accuracy, and $F_1$-score.
![Model Leaderboard](reports/05_model_leaderboard_comparison.png)

---

## 🎮 Interactive Vehicle Valuation CLI

You can interact with the valuation system using **preset vehicle archetypes** or by typing **custom vehicle specs**:

```bash
python interactive_advisor.py
```

### Input Options Menu:
```text
CHOOSE AN OPTION FOR INPUT:
  [1] Maruti Suzuki Swift (3 yrs, 32k km, 88 bhp, Asking Rs. 5.90 Lakh)
  [2] Hyundai Creta (4 yrs, 48k km, 113 bhp, Asking Rs. 10.40 Lakh)
  [3] Mahindra XUV700 7-Seater (2 yrs, 26k km, 182 bhp, Asking Rs. 17.50 Lakh)
  [4] Honda City Executive Sedan (6 yrs, 68k km, 119 bhp, Asking Rs. 5.80 Lakh)
  [5] Maruti Alto 800 Budget Car (7 yrs, 72k km, 47 bhp, Asking Rs. 2.10 Lakh)
  [6] ENTER CUSTOM VEHICLE DETAILS (Interactive Prompt: Age, KM, Power, Price, etc.)
  [7] EVALUATE ALL POPULAR PRESETS TOGETHER
  [0] Exit
```

### What You Get for Each Input:
1. **AI Fair Market Value Estimate** (in ₹ Lakhs).
2. **Value Variance**: Price difference and discount/markup percentage.
3. **Deal Quality Score**: Classification confidence (Great Deal vs. Fair/Overpriced).
4. **Negotiation Advisory**: Specific target purchase price to offer.
5. **3-Year Depreciation Forecast**: Estimated vehicle value and projected capital loss across Years 1, 2, and 3.

---

## 🚀 How to Run

1. Navigate to the project directory:
   ```bash
   cd "week 7-8"
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the complete training tournament and diagnostic report generation:
   ```bash
   python main.py
   ```
4. Run the interactive valuation advisor:
   ```bash
   python interactive_advisor.py
   ```

---

Made by [Dhanish Ladwani](https://github.com/dhanish0711/)
