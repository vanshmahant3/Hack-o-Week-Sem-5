import os
import numpy as np
import pandas as pd

def prepare_dataset():
    """Acquires or synthesizes the Used Car Resale Valuation & Deal Quality Dataset."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'used_cars.csv')

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 50000:
        print(f"  [+] Dataset already exists: {csv_path} ({os.path.getsize(csv_path) / 1024:.1f} KB)")
        return csv_path

    print("  [+] Generating realistic Used Car Resale Market Dataset (6,000 vehicles)...")
    np.random.seed(42)
    n_samples = 6000

    # 1. Primary automotive features
    car_age = np.random.randint(1, 15, size=n_samples)
    km_driven = np.clip(np.random.normal(loc=car_age * 9500 + 8000, scale=12000), 5000, 180000).astype(int)
    engine_cc = np.random.choice([998, 1197, 1373, 1498, 1798, 1995, 2179, 2498], size=n_samples, p=[0.18, 0.28, 0.16, 0.18, 0.08, 0.06, 0.04, 0.02])
    
    # Power correlates with engine displacement
    base_power = engine_cc * 0.072 + np.random.normal(0, 7, size=n_samples)
    max_power_bhp = np.clip(base_power, 52.0, 210.0).round(1)

    # Fuel efficiency inversely correlates with engine size and power
    mileage_kmpl = np.clip(28.0 - (engine_cc / 160.0) + np.random.normal(0, 1.5, size=n_samples), 10.5, 26.5).round(1)
    previous_owners = np.random.choice([1, 2, 3], size=n_samples, p=[0.72, 0.22, 0.06])
    seats = np.random.choice([5, 7], size=n_samples, p=[0.85, 0.15])

    # 2. Continuous Target: Resale Price in Lakhs INR (Non-linear depreciation)
    # Original showroom price estimated from power and engine
    original_price = 4.2 + (max_power_bhp * 0.11) + (engine_cc * 0.0035)
    
    # Non-linear age depreciation: Rapid in early years, flattens out later
    age_depreciation_factor = np.exp(-0.16 * car_age)
    
    # Mileage and owner discount
    km_penalty = 1.0 - np.clip(km_driven / 350000.0, 0.0, 0.35)
    owner_penalty = 1.0 - (previous_owners - 1) * 0.08

    # Intrinsic fair market price
    fair_price = original_price * age_depreciation_factor * km_penalty * owner_penalty
    
    # Market price with realistic seller variation noise (+/- 12%)
    seller_markup = np.random.normal(1.0, 0.10, size=n_samples)
    resale_price_lakh = np.clip(fair_price * seller_markup, 1.2, 38.0).round(2)

    # 3. Binary Target: Is it a "Good Deal" (Underpriced relative to fair specs)?
    # Well-balanced criterion: price is below fair market value and car is not excessively old
    price_discount = (fair_price - resale_price_lakh) / fair_price
    # High-value deal if asking price is discounted relative to specs
    is_good_deal = ((price_discount > 0.02) & (car_age <= 9)).astype(int)

    df = pd.DataFrame({
        'car_age_years': car_age,
        'km_driven': km_driven,
        'engine_cc': engine_cc,
        'max_power_bhp': max_power_bhp,
        'mileage_kmpl': mileage_kmpl,
        'previous_owners': previous_owners,
        'seats': seats,
        'resale_price_lakh': resale_price_lakh,
        'is_good_deal': is_good_deal
    })

    df.to_csv(csv_path, index=False)
    print(f"  [+] Saved {len(df):,} vehicle records to: {csv_path}")
    return csv_path

if __name__ == '__main__':
    prepare_dataset()
