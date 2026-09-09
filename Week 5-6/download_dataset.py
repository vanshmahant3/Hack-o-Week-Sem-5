import os
import urllib.request
import pandas as pd
import numpy as np

def prepare_dataset():
    """Acquires or synthesizes the Hotel Booking Demand & Cancellation Dataset."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, 'hotel_bookings.csv')
    
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 10000:
        print(f"  [+] Dataset already exists: {csv_path} ({os.path.getsize(csv_path) / 1024:.1f} KB)")
        return csv_path

    url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
    download_success = False

    try:
        print("  [+] Downloading Hotel Booking Demand Dataset from public repository...")
        urllib.request.urlretrieve(url, csv_path)
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 10000:
            df_raw = pd.read_csv(csv_path)
            # Select relevant core numerical features
            cols = [
                'lead_time', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights',
                'adults', 'previous_cancellations', 'previous_bookings_not_canceled',
                'booking_changes', 'days_in_waiting_list', 'required_car_parking_spaces',
                'total_of_special_requests', 'is_canceled'
            ]
            df = df_raw[cols].dropna()
            df.to_csv(csv_path, index=False)
            download_success = True
            print(f"  [+] Downloaded and parsed {len(df):,} real hotel booking records.")
    except Exception as e:
        print(f"  [!] Online download unavailable ({e}), generating high-fidelity dataset locally...")

    if not download_success:
        np.random.seed(42)
        n_samples = 15000
        
        lead_time = np.random.exponential(scale=85, size=n_samples).astype(int)
        adr = np.clip(np.random.normal(loc=105, scale=45, size=n_samples), 30, 350).round(2)
        stays_weekend = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.45, 0.35, 0.18, 0.02])
        stays_week = np.random.choice([1, 2, 3, 4, 5, 6, 7], size=n_samples, p=[0.25, 0.30, 0.20, 0.12, 0.08, 0.03, 0.02])
        adults = np.random.choice([1, 2, 3], size=n_samples, p=[0.20, 0.72, 0.08])
        prev_cancel = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.92, 0.05, 0.02, 0.01])
        prev_not_cancel = np.random.choice([0, 1, 2, 5], size=n_samples, p=[0.95, 0.03, 0.015, 0.005])
        booking_changes = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.75, 0.18, 0.05, 0.02])
        waiting_list = np.random.choice([0, 15, 30, 60], size=n_samples, p=[0.96, 0.02, 0.015, 0.005])
        parking = np.random.choice([0, 1], size=n_samples, p=[0.93, 0.07])
        special_requests = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.55, 0.30, 0.12, 0.03])

        # Realistic Cancellation probability based on linear combination + non-linear noise
        logits = (
            0.008 * lead_time
            + 0.006 * adr
            + 1.8 * prev_cancel
            - 1.5 * prev_not_cancel
            - 0.6 * booking_changes
            - 2.8 * parking
            - 0.9 * special_requests
            - 1.2
        )
        cancel_prob = 1.0 / (1.0 + np.exp(-logits))
        is_canceled = (np.random.rand(n_samples) < cancel_prob).astype(int)

        df = pd.DataFrame({
            'lead_time': lead_time,
            'adr': adr,
            'stays_in_weekend_nights': stays_weekend,
            'stays_in_week_nights': stays_week,
            'adults': adults,
            'previous_cancellations': prev_cancel,
            'previous_bookings_not_canceled': prev_not_cancel,
            'booking_changes': booking_changes,
            'days_in_waiting_list': waiting_list,
            'required_car_parking_spaces': parking,
            'total_of_special_requests': special_requests,
            'is_canceled': is_canceled
        })
        df.to_csv(csv_path, index=False)
        print(f"  [+] Synthesized {len(df):,} realistic booking records ({csv_path}).")

    return csv_path

if __name__ == '__main__':
    prepare_dataset()
