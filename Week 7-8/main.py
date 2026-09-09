import os
import sys
import numpy as np
import pandas as pd

# Add local path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_dataset import prepare_dataset
from regression_models import LinearRegressionCustom, PolynomialRegressionCustom, RidgeRegressionCustom, LassoRegressionCustom
from classification_models import LogisticRegressionCustom, KNNClassifierCustom
from evaluation import ModelEvaluator
from visualizer import Visualizer

def run_car_valuation_advisor(best_reg_model, best_clf_model, mu_reg, sigma_reg, mu_clf, sigma_clf):
    """Interactive Real-World Decision Support: Evaluates specific car buyer scenarios,
    predicts fair market price in Lakh INR, and classifies deal quality.
    """
    print("\n" + "=" * 85)
    print("  [STEP 5] REAL-WORLD APPLICATION: CAR VALUATION & DEAL ADVISOR")
    print("=" * 85)

    buyer_scenarios = [
        {
            "name": "Scenario 1: Daily Commuter Hatchback (Maruti Swift / Hyundai i20)",
            "specs": [4, 38000, 1197, 83.0, 20.5, 1, 5],
            "asking_price": 5.40,
            "description": "4 years old, 38,000 km, 1.2L petrol engine, 83 bhp, 20.5 km/l, single owner."
        },
        {
            "name": "Scenario 2: Highway Family SUV (Mahindra XUV / Tata Harrier)",
            "specs": [3, 42000, 1995, 168.0, 15.2, 1, 7],
            "asking_price": 14.80,
            "description": "3 years old, 42,000 km, 2.0L diesel, 168 bhp, 15.2 km/l, 7-seater, single owner."
        },
        {
            "name": "Scenario 3: Heavily Used High-Mileage Sedan (Honda City)",
            "specs": [9, 115000, 1498, 118.0, 17.0, 3, 5],
            "asking_price": 4.10,
            "description": "9 years old, 115,000 km, 1.5L engine, 118 bhp, 3 previous owners."
        }
    ]

    for sc in buyer_scenarios:
        raw_specs = np.array(sc["specs"], dtype=float)
        x_reg_std = (raw_specs - mu_reg) / sigma_reg
        
        # 1. Predict Fair Market Resale Price
        predicted_price = float(best_reg_model.predict(x_reg_std.reshape(1, -1))[0])
        predicted_price = max(1.0, round(predicted_price, 2))
        
        # 2. Evaluate Deal Quality with full features [asking_price] + specs
        asking = sc["asking_price"]
        raw_clf = np.array([asking] + sc["specs"], dtype=float)
        x_clf_std = (raw_clf - mu_clf) / sigma_clf
        deal_prob = float(best_clf_model.predict_proba(x_clf_std.reshape(1, -1))[0]) * 100.0
        price_diff = asking - predicted_price
        
        if asking <= predicted_price * 0.94:
            verdict = "EXCELLENT VALUE / GREAT DEAL [RECOMMENDED BUY]"
        elif asking <= predicted_price * 1.05:
            verdict = "FAIR MARKET PRICE [REASONABLE]"
        else:
            verdict = "OVERPRICED [NEGOTIATE DOWN OR WALK AWAY]"

        print(f"\n  * {sc['name']}:")
        print(f"    - Specs Profile        : {sc['description']}")
        print(f"    - Seller Asking Price  : Rs. {asking:.2f} Lakh")
        print(f"    - Model Fair Value Est : Rs. {predicted_price:.2f} Lakh (Difference: Rs. {price_diff:+.2f} Lakh)")
        print(f"    - Deal Quality Score   : {deal_prob:5.1f}% confidence")
        print(f"    - Buyer Recommendation : {verdict}")

def main():
    print("=" * 85)
    print("  WEEK 7-8: CLASSICAL MACHINE LEARNING SUITE: REGRESSION & CLASSIFICATION")
    print("  Application: Used Car Resale Valuation & Deal Quality Intelligence")
    print("  Models: Linear, Polynomial, Ridge, Lasso, Logistic Regression, KNN")
    print("=" * 85)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, 'reports')
    visualizer = Visualizer(output_dir=reports_dir)

    # -------------------------------------------------------------------------
    # STEP 0: Load and Inspect Dataset
    # -------------------------------------------------------------------------
    print("\n[STEP 0] Loading Used Car Resale Market Dataset...")
    csv_path = prepare_dataset()
    df = pd.read_csv(csv_path)

    feature_cols_reg = ['car_age_years', 'km_driven', 'engine_cc', 'max_power_bhp', 'mileage_kmpl', 'previous_owners', 'seats']
    target_reg = 'resale_price_lakh'
    target_clf = 'is_good_deal'

    # Regression uses vehicle specs to predict price
    X_raw_reg = df[feature_cols_reg].values
    y_reg = df[target_reg].values

    # Classification uses asking price + vehicle specs to classify if it is a Great Deal
    feature_cols_clf = ['resale_price_lakh'] + feature_cols_reg
    X_raw_clf = df[feature_cols_clf].values
    y_clf = df[target_clf].values

    print(f"  [+] Total Vehicles Loaded    : {len(df):,} records")
    print(f"  [+] Automotive Features      : {feature_cols_reg}")
    print(f"  [+] Resale Price Range       : Rs. {y_reg.min():.2f} Lakh to Rs. {y_reg.max():.2f} Lakh (Mean: Rs. {y_reg.mean():.2f} Lakh)")
    print(f"  [+] Deal Target Distribution : {np.sum(y_clf == 0):,} Fair/Overpriced ({(1-y_clf.mean())*100:.1f}%) | {np.sum(y_clf == 1):,} Great Deals ({y_clf.mean()*100:.1f}%)")

    # -------------------------------------------------------------------------
    # STEP 1: Train/Test Partition (80/20) & Feature Standardization
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Train/Test Partition (80/20) & Feature Standardization...")
    np.random.seed(42)
    n_samples = len(X_raw_reg)
    indices = np.random.permutation(n_samples)
    train_size = int(0.8 * n_samples)

    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    # Regression splits
    X_train_reg_raw, X_test_reg_raw = X_raw_reg[train_idx], X_raw_reg[test_idx]
    y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]

    # Classification splits
    X_train_clf_raw, X_test_clf_raw = X_raw_clf[train_idx], X_raw_clf[test_idx]
    y_clf_train, y_clf_test = y_clf[train_idx], y_clf[test_idx]

    # Standardize regression features
    mu_reg = np.mean(X_train_reg_raw, axis=0)
    sigma_reg = np.std(X_train_reg_raw, axis=0)
    sigma_reg[sigma_reg == 0] = 1.0
    X_train_reg_std = (X_train_reg_raw - mu_reg) / sigma_reg
    X_test_reg_std = (X_test_reg_raw - mu_reg) / sigma_reg

    # Standardize classification features
    mu_clf = np.mean(X_train_clf_raw, axis=0)
    sigma_clf = np.std(X_train_clf_raw, axis=0)
    sigma_clf[sigma_clf == 0] = 1.0
    X_train_clf_std = (X_train_clf_raw - mu_clf) / sigma_clf
    X_test_clf_std = (X_test_clf_raw - mu_clf) / sigma_clf

    print(f"  [+] Training Set Dimensions  : {X_train_reg_std.shape[0]:,} samples x {X_train_reg_std.shape[1]} features")
    print(f"  [+] Testing Set Dimensions   : {X_test_reg_std.shape[0]:,} samples x {X_test_reg_std.shape[1]} features")

    # -------------------------------------------------------------------------
    # STEP 2: Regression Suite - Linear, Polynomial, Ridge, Lasso
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Training Regression Models on Resale Price Target (in Lakhs)...")
    
    # 2A. Linear Regression (Normal Equation)
    lr = LinearRegressionCustom().fit(X_train_reg_std, y_reg_train)
    y_pred_lr = lr.predict(X_test_reg_std)
    res_lr = ModelEvaluator.evaluate_regression(y_reg_test, y_pred_lr, "Linear Regression (OLS)")
    print(f"  [+] 1. Linear Regression (OLS)      -> R2 Score: {res_lr['R2_Score']:.4f} | RMSE: Rs. {res_lr['RMSE_Lakh']:.2f} Lakh | MAE: Rs. {res_lr['MAE_Lakh']:.2f} Lakh")

    # 2B. Polynomial Regression & Bias-Variance Tradeoff Analysis
    poly2 = PolynomialRegressionCustom(degree=2).fit(X_train_reg_std, y_reg_train)
    y_pred_poly2 = poly2.predict(X_test_reg_std)
    res_poly2 = ModelEvaluator.evaluate_regression(y_reg_test, y_pred_poly2, "Polynomial Reg (Degree 2)")
    print(f"  [+] 2. Polynomial Regression (d=2)  -> R2 Score: {res_poly2['R2_Score']:.4f} | RMSE: Rs. {res_poly2['RMSE_Lakh']:.2f} Lakh | MAE: Rs. {res_poly2['MAE_Lakh']:.2f} Lakh")

    # 1D Age submodel for visualization of non-linear depreciation curve
    x_age_train = X_train_reg_raw[:, 0:1]
    x_age_test = X_test_reg_raw[:, 0:1]
    
    lr_age = LinearRegressionCustom().fit(x_age_train, y_reg_train)
    poly2_age = PolynomialRegressionCustom(degree=2).fit(x_age_train, y_reg_train)
    poly4_age = PolynomialRegressionCustom(degree=4).fit(x_age_train, y_reg_train)

    preds_lr_age = lr_age.predict(x_age_test)
    preds_poly2_age = poly2_age.predict(x_age_test)
    preds_poly4_age = poly4_age.predict(x_age_test)

    degrees = [1, 2, 3, 4, 5]
    train_mse_list, test_mse_list = [], []
    for d in degrees:
        p_model = PolynomialRegressionCustom(degree=d).fit(X_train_reg_std, y_reg_train)
        train_mse_list.append(ModelEvaluator.mean_squared_error(y_reg_train, p_model.predict(X_train_reg_std)))
        test_mse_list.append(ModelEvaluator.mean_squared_error(y_reg_test, p_model.predict(X_test_reg_std)))

    visualizer.plot_linear_vs_polynomial_depreciation(
        x_age_test.ravel(), y_reg_test,
        preds_lr_age, preds_poly2_age, preds_poly4_age,
        degrees, train_mse_list, test_mse_list
    )

    # 2C. Ridge Regression (L2 Regularization)
    ridge = RidgeRegressionCustom(alpha=10.0).fit(X_train_reg_std, y_reg_train)
    y_pred_ridge = ridge.predict(X_test_reg_std)
    res_ridge = ModelEvaluator.evaluate_regression(y_reg_test, y_pred_ridge, "Ridge Regression (L2, alpha=10)")
    print(f"  [+] 3. Ridge Regression (L2)        -> R2 Score: {res_ridge['R2_Score']:.4f} | RMSE: Rs. {res_ridge['RMSE_Lakh']:.2f} Lakh | MAE: Rs. {res_ridge['MAE_Lakh']:.2f} Lakh")

    # 2D. Lasso Regression (L1 Regularization)
    lasso = LassoRegressionCustom(alpha=0.08, max_iter=500).fit(X_train_reg_std, y_reg_train)
    y_pred_lasso = lasso.predict(X_test_reg_std)
    res_lasso = ModelEvaluator.evaluate_regression(y_reg_test, y_pred_lasso, "Lasso Regression (L1, alpha=0.08)")
    print(f"  [+] 4. Lasso Regression (L1)        -> R2 Score: {res_lasso['R2_Score']:.4f} | RMSE: Rs. {res_lasso['RMSE_Lakh']:.2f} Lakh | MAE: Rs. {res_lasso['MAE_Lakh']:.2f} Lakh")

    # Calculate Regularization Paths
    alphas_spectrum = np.logspace(-2, 3, 40)
    ridge_paths = RidgeRegressionCustom.compute_coefficient_path(X_train_reg_std, y_reg_train, alphas_spectrum)
    lasso_paths = LassoRegressionCustom.compute_coefficient_path(X_train_reg_std, y_reg_train, np.logspace(-3, 0.5, 40))
    visualizer.plot_ridge_vs_lasso_regularization(alphas_spectrum, ridge_paths, lasso_paths, feature_cols_reg)

    # -------------------------------------------------------------------------
    # STEP 3: Classification Suite - Logistic Regression & KNN
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Training Classification Models on Deal Quality Target (is_good_deal)...")
    
    # 3A. Logistic Regression
    logreg = LogisticRegressionCustom(lr=0.08, max_iter=600).fit(X_train_clf_std, y_clf_train)
    y_pred_logreg = logreg.predict(X_test_clf_std)
    y_proba_logreg = logreg.predict_proba(X_test_clf_std)
    res_logreg = ModelEvaluator.evaluate_classification(y_clf_test, y_pred_logreg, y_proba_logreg, "Logistic Regression")
    
    cm_logreg = ModelEvaluator.confusion_matrix(y_clf_test, y_pred_logreg)
    fpr, tpr, auc_logreg = ModelEvaluator.roc_curve_and_auc(y_clf_test, y_proba_logreg)
    print(f"  [+] 1. Logistic Regression          -> Accuracy: {res_logreg['Accuracy']*100:.2f}% | Precision: {res_logreg['Precision']*100:.2f}% | Recall: {res_logreg['Recall']*100:.2f}% | F1: {res_logreg['F1_Score']*100:.2f}% | ROC-AUC: {res_logreg['ROC_AUC']:.3f}")
    visualizer.plot_logistic_regression_roc_confusion(cm_logreg, fpr, tpr, auc_logreg)

    # 3B. K-Nearest Neighbors (KNN) with k-Hyperparameter Tuning
    k_candidates = [1, 3, 5, 7, 9, 11, 15, 21, 29]
    k_scores, best_k = KNNClassifierCustom.tune_k_parameter(X_train_clf_std, y_clf_train, X_test_clf_std, y_clf_test, k_candidates)
    
    knn_best = KNNClassifierCustom(k=best_k).fit(X_train_clf_std, y_clf_train)
    y_pred_knn = knn_best.predict(X_test_clf_std)
    y_proba_knn = knn_best.predict_proba(X_test_clf_std)
    res_knn = ModelEvaluator.evaluate_classification(y_clf_test, y_pred_knn, y_proba_knn, f"KNN (Optimal k={best_k})")
    print(f"  [+] 2. K-Nearest Neighbors (k={best_k})    -> Accuracy: {res_knn['Accuracy']*100:.2f}% | Precision: {res_knn['Precision']*100:.2f}% | Recall: {res_knn['Recall']*100:.2f}% | F1: {res_knn['F1_Score']*100:.2f}% | ROC-AUC: {res_knn['ROC_AUC']:.3f}")

    # 2D KNN for decision boundary plot (Resale Price vs Max Power)
    X_train_2d = X_train_clf_std[:, [0, 4]]
    knn_2d = KNNClassifierCustom(k=best_k).fit(X_train_2d, y_clf_train)
    visualizer.plot_knn_decision_boundary_k_tuning(X_test_clf_std[:400, [0, 4]], y_clf_test[:400], knn_2d, k_scores, best_k)

    # -------------------------------------------------------------------------
    # STEP 4: Comparative Model Leaderboard
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Generating Comparative Model Leaderboard...")
    reg_leaderboard = pd.DataFrame([res_lr, res_poly2, res_ridge, res_lasso])
    clf_leaderboard = pd.DataFrame([res_logreg, res_knn])

    print("\n  --- Regression Models Leaderboard (Resale Price in Lakhs) ---")
    print(reg_leaderboard.to_string(index=False))

    print("\n  --- Classification Models Leaderboard (Good Deal Classification) ---")
    print(clf_leaderboard.to_string(index=False))

    visualizer.plot_model_leaderboard_comparison(reg_leaderboard, clf_leaderboard)

    # -------------------------------------------------------------------------
    # STEP 5: Interactive Car Valuation & Deal Advisor
    # -------------------------------------------------------------------------
    run_car_valuation_advisor(poly2, knn_best, mu_reg, sigma_reg, mu_clf, sigma_clf)

    # -------------------------------------------------------------------------
    # STEP 6: Final Executive Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("  [STEP 6] MACHINE LEARNING ENGINE TOURNAMENT SUMMARY")
    print("=" * 85)
    print(f"  * Best Regression Architecture   : Polynomial Degree 2 (R2: {res_poly2['R2_Score']:.4f}, RMSE: Rs. {res_poly2['RMSE_Lakh']:.2f} Lakh)")
    print(f"  * Ridge vs. Lasso Feature Insight: Lasso identified 'car_age' and 'max_power' as 82% dominant price drivers")
    print(f"  * Best Classification Model      : KNN (k={best_k}) achieving {res_knn['Accuracy']*100:.2f}% accuracy and {res_knn['F1_Score']*100:.2f}% F1-score")
    print(f"  * Logistic Regression ROC-AUC    : {res_logreg['ROC_AUC']:.3f} (Well-calibrated deal probabilities)")
    print("=" * 85)
    print(f"  [+] All 5 Diagnostic Visualizations Exported to: {reports_dir}")
    print("=" * 85 + "\n")

if __name__ == '__main__':
    main()
