import os
import sys
import numpy as np
import pandas as pd

# Add local path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_dataset import prepare_dataset
from regression_models import PolynomialRegressionCustom
from classification_models import KNNClassifierCustom

class CarValuationSystem:
    def __init__(self):
        self.csv_path = prepare_dataset()
        self.df = pd.read_csv(self.csv_path)
        
        self.feature_cols_reg = ['car_age_years', 'km_driven', 'engine_cc', 'max_power_bhp', 'mileage_kmpl', 'previous_owners', 'seats']
        self.feature_cols_clf = ['resale_price_lakh'] + self.feature_cols_reg
        
        # Prepare data & statistics
        X_reg = self.df[self.feature_cols_reg].values
        y_reg = self.df['resale_price_lakh'].values
        
        X_clf = self.df[self.feature_cols_clf].values
        y_clf = self.df['is_good_deal'].values
        
        self.mu_reg = np.mean(X_reg, axis=0)
        self.sigma_reg = np.std(X_reg, axis=0)
        self.sigma_reg[self.sigma_reg == 0] = 1.0
        
        self.mu_clf = np.mean(X_clf, axis=0)
        self.sigma_clf = np.std(X_clf, axis=0)
        self.sigma_clf[self.sigma_clf == 0] = 1.0
        
        X_reg_std = (X_reg - self.mu_reg) / self.sigma_reg
        X_clf_std = (X_clf - self.mu_clf) / self.sigma_clf
        
        # Fit models
        print("  [+] Calibrating Valuation & Classification Engines...")
        self.reg_model = PolynomialRegressionCustom(degree=2).fit(X_reg_std, y_reg)
        self.clf_model = KNNClassifierCustom(k=29).fit(X_clf_std, y_clf)
        print("  [+] Engine Ready!\n")

    def evaluate_car(self, name: str, age: float, km: float, cc: float, bhp: float, kmpl: float, owners: int, seats: int, asking_price: float):
        raw_specs = np.array([age, km, cc, bhp, kmpl, owners, seats], dtype=float)
        x_reg_std = (raw_specs - self.mu_reg) / self.sigma_reg
        
        # 1. Fair market value prediction
        predicted_price = float(self.reg_model.predict(x_reg_std.reshape(1, -1))[0])
        predicted_price = max(0.8, round(predicted_price, 2))
        
        # 2. Deal quality scoring
        raw_clf = np.array([asking_price] + list(raw_specs), dtype=float)
        x_clf_std = (raw_clf - self.mu_clf) / self.sigma_clf
        deal_prob = float(self.clf_model.predict_proba(x_clf_std.reshape(1, -1))[0]) * 100.0
        
        price_diff = asking_price - predicted_price
        pct_diff = (price_diff / predicted_price) * 100.0
        
        if asking_price <= predicted_price * 0.94:
            verdict = "EXCELLENT VALUE / GREAT DEAL [RECOMMENDED BUY]"
            action = f"Grab this deal! Priced Rs. {abs(price_diff):.2f} Lakh ({abs(pct_diff):.1f}%) below fair market valuation."
        elif asking_price <= predicted_price * 1.05:
            verdict = "FAIR MARKET PRICE [REASONABLE]"
            action = f"Market-aligned price. Offer Rs. {max(0.5, asking_price - 0.3):.2f} Lakh to get optimal value."
        else:
            verdict = "OVERPRICED [NEGOTIATE DOWN OR WALK AWAY]"
            action = f"Overpriced by Rs. {price_diff:+.2f} Lakh (+{pct_diff:.1f}%). Insist on negotiating down to Rs. {predicted_price:.2f} Lakh."

        # 3. 3-Year Resale Depreciation Forecast
        depreciation_forecast = []
        for extra_year in [1, 2, 3]:
            future_specs = np.array([age + extra_year, km + (extra_year * 10000), cc, bhp, kmpl, owners, seats], dtype=float)
            x_future_std = (future_specs - self.mu_reg) / self.sigma_reg
            future_price = max(0.6, round(float(self.reg_model.predict(x_future_std.reshape(1, -1))[0]), 2))
            depreciation_forecast.append((age + extra_year, future_price))

        # Render report
        print("\n" + "=" * 78)
        print(f"  VEHICLE VALUATION & DEAL INTELLIGENCE REPORT: {name}")
        print("=" * 78)
        print(f"  * Vehicle Specs            : Age: {age} yrs | Mileage: {km:,.0f} km | Engine: {cc:.0f}cc")
        print(f"                               Power: {bhp:.1f} bhp | Efficiency: {kmpl:.1f} kmpl | Owners: {owners}")
        print(f"  * Seller Asking Price      : Rs. {asking_price:.2f} Lakh")
        print(f"  * AI Fair Market Value Est : Rs. {predicted_price:.2f} Lakh")
        print(f"  * Value Variance           : Rs. {price_diff:+.2f} Lakh ({pct_diff:+.1f}%)")
        print(f"  * Deal Quality Confidence  : {deal_prob:5.1f}% confidence")
        print(f"  * Executive Verdict        : {verdict}")
        print(f"  * Advisory Action          : {action}")
        print("-" * 78)
        print("  * Projected 3-Year Value Depreciation Trajectory:")
        for fut_age, fut_val in depreciation_forecast:
            loss_from_today = asking_price - fut_val
            print(f"    - In {fut_age - age:.0f} year(s) (Age {fut_age:.0f}): Rs. {fut_val:.2f} Lakh (Estimated loss: Rs. {max(0.0, loss_from_today):.2f} Lakh)")
        print("=" * 78 + "\n")

PRESET_CARS = [
    {
        "name": "Maruti Suzuki Swift (Compact City Hatchback)",
        "age": 3.0, "km": 32000.0, "cc": 1197.0, "bhp": 88.5, "kmpl": 22.4, "owners": 1, "seats": 5,
        "asking_price": 5.90
    },
    {
        "name": "Hyundai Creta / Kia Seltos (Mid-Size Urban SUV)",
        "age": 4.0, "km": 48000.0, "cc": 1497.0, "bhp": 113.4, "kmpl": 16.8, "owners": 1, "seats": 5,
        "asking_price": 10.40
    },
    {
        "name": "Mahindra XUV700 (High-Power 7-Seater Family SUV)",
        "age": 2.0, "km": 26000.0, "cc": 2179.0, "bhp": 182.4, "kmpl": 14.5, "owners": 1, "seats": 7,
        "asking_price": 17.50
    },
    {
        "name": "Honda City (Executive Sedan)",
        "age": 6.0, "km": 68000.0, "cc": 1498.0, "bhp": 119.3, "kmpl": 17.8, "owners": 2, "seats": 5,
        "asking_price": 5.80
    },
    {
        "name": "Maruti Alto 800 (First-Time Budget Commuter)",
        "age": 7.0, "km": 72000.0, "cc": 796.0, "bhp": 47.3, "kmpl": 22.0, "owners": 2, "seats": 5,
        "asking_price": 2.10
    }
]

def interactive_menu():
    print("=" * 78)
    print("      USED CAR VALUATION & DEAL QUALITY INTERACTIVE ADVISOR")
    print("      Regression & Classification ML Suite (Week 7-8)")
    print("=" * 78)
    
    system = CarValuationSystem()

    while True:
        print("\nCHOOSE AN OPTION FOR INPUT:")
        print("  [1] Maruti Suzuki Swift (3 yrs, 32k km, 88 bhp, Asking Rs. 5.90 Lakh)")
        print("  [2] Hyundai Creta (4 yrs, 48k km, 113 bhp, Asking Rs. 10.40 Lakh)")
        print("  [3] Mahindra XUV700 7-Seater (2 yrs, 26k km, 182 bhp, Asking Rs. 17.50 Lakh)")
        print("  [4] Honda City Executive Sedan (6 yrs, 68k km, 119 bhp, Asking Rs. 5.80 Lakh)")
        print("  [5] Maruti Alto 800 Budget Car (7 yrs, 72k km, 47 bhp, Asking Rs. 2.10 Lakh)")
        print("  [6] ENTER CUSTOM VEHICLE DETAILS (Interactive Prompt)")
        print("  [7] EVALUATE ALL POPULAR PRESETS TOGETHER")
        print("  [0] Exit")
        
        try:
            choice = input("\nEnter choice [0-7]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Thank you for using the Car Valuation Engine!")
            break

        if choice in ['1', '2', '3', '4', '5']:
            idx = int(choice) - 1
            car = PRESET_CARS[idx]
            system.evaluate_car(
                car["name"], car["age"], car["km"], car["cc"], car["bhp"],
                car["kmpl"], car["owners"], car["seats"], car["asking_price"]
            )
        elif choice == '6':
            print("\n--- Enter Custom Vehicle Attributes ---")
            try:
                name = input("  Vehicle Model / Name (e.g. Tata Nexon): ").strip() or "Custom Vehicle"
                age = float(input("  Car Age in Years (e.g. 3.5): ").strip())
                km = float(input("  Total KM Driven (e.g. 45000): ").strip())
                cc = float(input("  Engine Capacity CC (e.g. 1199): ").strip())
                bhp = float(input("  Engine Power BHP (e.g. 118): ").strip())
                kmpl = float(input("  Mileage / Fuel Efficiency KMPL (e.g. 17.5): ").strip())
                owners = int(input("  Previous Registered Owners (1, 2, or 3): ").strip())
                seats = int(input("  Seating Capacity (5 or 7): ").strip())
                asking = float(input("  Seller Asking Price in Lakh INR (e.g. 7.20): ").strip())
                
                system.evaluate_car(name, age, km, cc, bhp, kmpl, owners, seats, asking)
            except Exception as e:
                print(f"\n  [!] Invalid input: {e}. Please try again.")
        elif choice == '7':
            print("\n--- Evaluating All Popular Market Presets ---")
            for car in PRESET_CARS:
                system.evaluate_car(
                    car["name"], car["age"], car["km"], car["cc"], car["bhp"],
                    car["kmpl"], car["owners"], car["seats"], car["asking_price"]
                )
        elif choice == '0':
            print("\nExiting. Thank you for using the Car Valuation Engine!")
            break
        else:
            print("\n  [!] Invalid selection. Please enter a number from 0 to 7.")

if __name__ == '__main__':
    # If run with argument '--demo' or non-interactively
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        sys_demo = CarValuationSystem()
        print("\n--- Running Automated Input Evaluation on Presets ---")
        for p in PRESET_CARS[:3]:
            sys_demo.evaluate_car(p["name"], p["age"], p["km"], p["cc"], p["bhp"], p["kmpl"], p["owners"], p["seats"], p["asking_price"])
    else:
        interactive_menu()
