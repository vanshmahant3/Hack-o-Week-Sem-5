"""
Data Generation & Ingestion Module for Week 15-16:
Clinical Heart Disease & Cardiovascular Risk Dataset + Non-Linear Toy Manifolds.

Author: Dhanish Ladwani
Curriculum: Machine Learning & Pattern Recognition — Week 15-16
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons


def generate_heart_disease_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Synthesizes a realistic, non-linear clinical cardiovascular risk dataset
    based on standard cardiology diagnostic protocols (e.g., Cleveland/Framingham criteria).
    
    Parameters:
        n_samples (int): Number of patient records (default: 1500).
        random_state (int): Deterministic random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Clean clinical dataset with 8 biomarkers and 1 binary target.
    """
    rng = np.random.RandomState(random_state)
    
    # 1. Age (years): mean 54, std 9, bounded [28, 78]
    age = np.clip(rng.normal(loc=54.5, scale=9.2, size=n_samples), 28, 78).round(0).astype(int)
    
    # 2. Resting Blood Pressure (mmHg): mean 131, std 17, bounded [95, 190]
    resting_bp = np.clip(rng.normal(loc=131.5, scale=17.5, size=n_samples), 95, 190).round(0).astype(int)
    
    # 3. Serum Cholesterol (mg/dL): mean 246, std 48, bounded [135, 370]
    cholesterol = np.clip(rng.normal(loc=246.0, scale=48.0, size=n_samples), 135, 370).round(0).astype(int)
    
    # 4. Maximum Heart Rate (bpm): decreases with age (220 - age baseline with clinical variance)
    expected_max_hr = 208 - 0.7 * age
    max_heart_rate = np.clip(rng.normal(loc=expected_max_hr, scale=16.0), 72, 202).round(0).astype(int)
    
    # 5. ST Depression induced by exercise relative to rest (0.0 to 4.5)
    st_depression = np.clip(rng.exponential(scale=1.1, size=n_samples), 0.0, 4.5).round(2)
    
    # 6. Exercise Induced Angina (0 = No, 1 = Yes)
    angina_prob = 1.0 / (1.0 + np.exp(-(0.04 * (age - 50) + 0.6 * st_depression - 1.2)))
    exercise_angina = (rng.uniform(size=n_samples) < angina_prob).astype(int)
    
    # 7. Fasting Blood Sugar > 120 mg/dL (0 = Normal, 1 = High, ~15% baseline)
    fbs_prob = 0.10 + 0.002 * (age - 30) + 0.0003 * (cholesterol - 150)
    fasting_blood_sugar = (rng.uniform(size=n_samples) < np.clip(fbs_prob, 0.05, 0.40)).astype(int)
    
    # 8. Number of Major Vessels colored by fluoroscopy (0 to 3)
    vessels_weights = [0.55, 0.23, 0.15, 0.07]
    vessels_colored = rng.choice([0, 1, 2, 3], size=n_samples, p=vessels_weights)
    
    # Latent Non-Linear Cardiovascular Risk Function
    # Combines linear clinical markers with strong non-linear interactions:
    # - Low max heart rate coupled with high cholesterol (atherosclerosis proxy)
    # - Non-linear ST depression exponent (ischemia indicator)
    # - Angina-vessel interaction
    z = (
        0.035 * (age - 50)
        + 0.020 * (resting_bp - 120)
        + 0.012 * (cholesterol - 200)
        - 0.038 * (max_heart_rate - 140)
        + 1.100 * (st_depression ** 1.3)
        + 0.850 * exercise_angina
        + 0.450 * fasting_blood_sugar
        + 0.650 * vessels_colored
        + 0.750 * ((cholesterol / 240.0) * (140.0 / np.maximum(max_heart_rate, 80)) - 1.0)
        + 0.500 * (exercise_angina * vessels_colored)
        - 2.800  # Calibration intercept
        + rng.normal(0, 0.35, size=n_samples)  # Physiological irreducible noise
    )
    
    # Probability via Sigmoid
    risk_prob = 1.0 / (1.0 + np.exp(-z))
    heart_disease_risk = (risk_prob >= 0.50).astype(int)
    
    df = pd.DataFrame({
        "age": age,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "max_heart_rate": max_heart_rate,
        "st_depression": st_depression,
        "exercise_angina": exercise_angina,
        "fasting_blood_sugar": fasting_blood_sugar,
        "vessels_colored": vessels_colored,
        "heart_disease_risk": heart_disease_risk
    })
    
    return df


def prepare_dataset(data_dir: str = "week 15-16/data") -> str:
    """
    Generates and saves the dataset CSV if not already present.
    Returns the absolute path to the dataset CSV.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "heart_disease_risk.csv")
    
    if not os.path.exists(csv_path):
        df = generate_heart_disease_dataset(n_samples=1500, random_state=42)
        df.to_csv(csv_path, index=False)
        print(f"Generated and cached clinical dataset at: {csv_path}")
    else:
        print(f"Using existing clinical dataset at: {csv_path}")
        
    return os.path.abspath(csv_path)


def load_preprocessed_data(csv_path: str = None, test_size: float = 0.20, random_state: int = 42):
    """
    Loads clinical heart disease dataset, scales numerical features via StandardScaler,
    and returns train/test splits.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    if csv_path is None:
        csv_path = prepare_dataset()
        
    df = pd.read_csv(csv_path)
    feature_names = [col for col in df.columns if col != "heart_disease_risk"]
    
    X = df[feature_names].values
    y = df["heart_disease_risk"].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def get_toy_datasets(random_state: int = 42):
    """
    Returns non-linear and linear benchmark toy datasets:
    1. AND gate: Linearly separable
    2. OR gate: Linearly separable
    3. XOR gate: Non-linearly separable (Minsky & Papert limitation)
    4. Moons: 2D interleaved half-circles (Scikit-Learn make_moons)
    """
    # Logic Truth Tables
    and_data = {
        "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([0, 0, 0, 1]).reshape(-1, 1),
        "name": "Logical AND Gate"
    }
    
    or_data = {
        "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([0, 1, 1, 1]).reshape(-1, 1),
        "name": "Logical OR Gate"
    }
    
    xor_data = {
        "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([0, 1, 1, 0]).reshape(-1, 1),
        "name": "Logical XOR Gate"
    }
    
    # Interleaved Moons
    X_moons, y_moons = make_moons(n_samples=400, noise=0.20, random_state=random_state)
    moons_data = {
        "X": X_moons,
        "y": y_moons.reshape(-1, 1),
        "name": "Interleaved Moons (2D)"
    }
    
    return and_data, or_data, xor_data, moons_data


if __name__ == "__main__":
    csv_file = prepare_dataset()
    df_sample = pd.read_csv(csv_file)
    print("--- Clinical Dataset Preview ---")
    print(df_sample.info())
    print("\nTarget Class Distribution:")
    print(df_sample["heart_disease_risk"].value_counts(normalize=True).round(3) * 100)
    print("\nFirst 5 rows:")
    print(df_sample.head())
