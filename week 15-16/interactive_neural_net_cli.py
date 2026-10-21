"""
Interactive CLI Tool: Clinical Cardiovascular Risk Scoring & Model Deployment Inspector.
Allows practitioners to test custom patient biomarkers, inspect live neural net layer
activations, verify artifact SHA-256 checksums, and evaluate real-time deployment latency.

Author: Dhanish Ladwani
Curriculum: Machine Learning & Pattern Recognition — Week 15-16
"""

import os
import sys
import json
import numpy as np

# Ensure module visibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deployment_engine import RealTimeScoringMicroservice, DeploymentEngine


PRESET_PATIENTS = {
    "healthy_athlete": {
        "name": "Healthy Young Adult / Endurance Athlete",
        "data": {
            "age": 32,
            "resting_bp": 110,
            "cholesterol": 165,
            "max_heart_rate": 185,
            "st_depression": 0.0,
            "exercise_angina": 0,
            "fasting_blood_sugar": 0,
            "vessels_colored": 0
        }
    },
    "borderline_executive": {
        "name": "Borderline / Moderate Risk Patient",
        "data": {
            "age": 52,
            "resting_bp": 138,
            "cholesterol": 242,
            "max_heart_rate": 142,
            "st_depression": 1.1,
            "exercise_angina": 0,
            "fasting_blood_sugar": 1,
            "vessels_colored": 1
        }
    },
    "critical_cardiac_patient": {
        "name": "Critical Ischemic Patient (High Urgency)",
        "data": {
            "age": 67,
            "resting_bp": 168,
            "cholesterol": 315,
            "max_heart_rate": 105,
            "st_depression": 2.8,
            "exercise_angina": 1,
            "fasting_blood_sugar": 1,
            "vessels_colored": 3
        }
    }
}


def print_banner():
    print("=" * 78)
    print("  HEARTGUARD AI: CLINICAL CARDIOVASCULAR RISK & NEURAL NET DEPLOYMENT")
    print("  NumPy Deep Multi-Layer Perceptron from Scratch | Joblib/Pickle Serving")
    print("  Author: Dhanish Ladwani")
    print("=" * 78)


def inspect_layer_activations(service: RealTimeScoringMicroservice, patient_dict: dict):
    """Passes input through model and inspects internal hidden layer representations."""
    raw_vec = np.array([[patient_dict[f] for f in service.feature_names]], dtype=float)
    scaled_vec = service.scaler.transform(raw_vec)
    
    A_L, cache = service.model.forward(scaled_vec)
    
    print("\n--- Internal Neural Network Layer Representations ---")
    print(f"  Input Vector (Scaled 8-dim):")
    print("   ", np.array2string(scaled_vec[0], precision=2, suppress_small=True))
    
    L = len(service.model.layer_dims) - 1
    for l in range(1, L):
        a_l = cache["A"][l][0]
        active_ratio = np.mean(a_l > 0.0) * 100.0
        print(f"  Layer {l} (Hidden: {service.model.layer_dims[l]} neurons, {service.model.hidden_activation_name.upper()}):")
        print(f"    Activations: {np.array2string(a_l, precision=3, suppress_small=True)}")
        print(f"    Active Neurons: {np.sum(a_l > 0.0)}/{len(a_l)} ({active_ratio:.1f}%)")
        
    print(f"  Output Layer (Sigmoid Probability): {A_L[0, 0]:.4f}")


def run_cli():
    print_banner()
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deployed_models", "model_artifact_joblib_compressed.joblib")
    if not os.path.exists(model_path):
        print(f"\n[!] Model artifact not found at: {model_path}")
        print("    Please run main.py first to train and serialize the production model bundle.")
        return
        
    # SHA-256 verification
    checksum = DeploymentEngine.compute_sha256(model_path)
    print(f"\n[OK] Loaded Deployed Model Artifact: {os.path.basename(model_path)}")
    print(f"    Cryptographic SHA-256: {checksum[:16]}...{checksum[-8:]}")
    
    service = RealTimeScoringMicroservice(model_path, format_type="joblib")
    
    while True:
        print("\n" + "-" * 78)
        print("Select an option:")
        print("  1. Evaluate Preset Patient: Healthy Endurance Athlete")
        print("  2. Evaluate Preset Patient: Borderline Risk Patient")
        print("  3. Evaluate Preset Patient: Critical Ischemic Patient")
        print("  4. Enter Custom Patient Biomarkers")
        print("  5. Run 1,000-Request Production Latency Benchmark")
        print("  6. Exit")
        print("-" * 78)
        
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == "1":
            p = PRESET_PATIENTS["healthy_athlete"]
            print(f"\nEvaluating: {p['name']}")
            res = service.score_patient(p["data"])
            print(json.dumps(res, indent=2))
            inspect_layer_activations(service, p["data"])
            
        elif choice == "2":
            p = PRESET_PATIENTS["borderline_executive"]
            print(f"\nEvaluating: {p['name']}")
            res = service.score_patient(p["data"])
            print(json.dumps(res, indent=2))
            inspect_layer_activations(service, p["data"])
            
        elif choice == "3":
            p = PRESET_PATIENTS["critical_cardiac_patient"]
            print(f"\nEvaluating: {p['name']}")
            res = service.score_patient(p["data"])
            print(json.dumps(res, indent=2))
            inspect_layer_activations(service, p["data"])
            
        elif choice == "4":
            print("\nEnter Patient Clinical Biomarkers:")
            try:
                age = float(input("  Age (years, e.g. 55): "))
                bp = float(input("  Resting Blood Pressure (mmHg, e.g. 135): "))
                chol = float(input("  Cholesterol (mg/dL, e.g. 240): "))
                max_hr = float(input("  Max Heart Rate (bpm, e.g. 140): "))
                st_dep = float(input("  ST Depression (0.0 to 4.5, e.g. 1.2): "))
                angina = int(input("  Exercise Angina (0 = No, 1 = Yes): "))
                fbs = int(input("  Fasting Blood Sugar > 120 (0 = No, 1 = Yes): "))
                vessels = int(input("  Major Vessels Colored (0 to 3): "))
                
                custom_patient = {
                    "age": age,
                    "resting_bp": bp,
                    "cholesterol": chol,
                    "max_heart_rate": max_hr,
                    "st_depression": st_dep,
                    "exercise_angina": angina,
                    "fasting_blood_sugar": fbs,
                    "vessels_colored": vessels
                }
                res = service.score_patient(custom_patient)
                print("\n=== Clinical Risk Assessment Result ===")
                print(json.dumps(res, indent=2))
                inspect_layer_activations(service, custom_patient)
            except Exception as e:
                print(f"[!] Invalid input: {e}")
                
        elif choice == "5":
            print("\nRunning 1,000-request production inference latency benchmark...")
            sample = PRESET_PATIENTS["borderline_executive"]["data"]
            lat_stats = service.benchmark_inference_latency(sample, n_requests=1000)
            print(f"  Total Requests:  {lat_stats['n_requests']}")
            print(f"  Mean Latency:    {lat_stats['mean_us']:.2f} µs ({lat_stats['mean_us']/1000.0:.3f} ms)")
            print(f"  Median (P50):    {lat_stats['median_p50_us']:.2f} µs ({lat_stats['median_p50_us']/1000.0:.3f} ms)")
            print(f"  95th Percentile: {lat_stats['p95_us']:.2f} µs ({lat_stats['p95_us']/1000.0:.3f} ms)")
            print(f"  99th Percentile: {lat_stats['p99_us']:.2f} µs ({lat_stats['p99_us']/1000.0:.3f} ms)")
            print(f"  Peak Throughput: {lat_stats['throughput_req_per_sec']:.0f} requests / sec")
            
        elif choice == "6":
            print("\nExiting HeartGuard AI CLI. Stay healthy!\n")
            break
        else:
            print("[!] Invalid choice. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    run_cli()
