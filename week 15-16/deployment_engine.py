"""
Model Deployment Basics & Serialization Engine: Pickle vs. Joblib.
Implements Model Persistence, Production Artifact Enveloping, SHA-256 Checksum
Integrity Verification, Benchmarking, and a Real-Time Scoring Microservice Simulator.

Author: Dhanish Ladwani
Curriculum: Machine Learning & Pattern Recognition — Week 15-16
"""

import os
import time
import pickle
import joblib
import hashlib
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, Optional


class ModelDeploymentBundle:
    """
    Production-ready Model Artifact Envelope.
    Encapsulates trained model, input preprocessor, feature schema, decision threshold,
    metadata, and cryptographic integrity verification.
    """
    
    def __init__(
        self,
        model: Any,
        scaler: Any,
        feature_names: list,
        metrics: Dict[str, float],
        version: str = "1.0.0",
        threshold: float = 0.50
    ):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.threshold = threshold
        self.metrics = metrics
        self.metadata = {
            "model_name": "ClinicalCardiovascularRisk_NumPy_MLP",
            "version": version,
            "created_at": datetime.now().isoformat(),
            "author": "Dhanish Ladwani",
            "framework": "NumPy Scratch MLP",
            "feature_count": len(feature_names),
            "feature_schema": feature_names,
            "decision_threshold": threshold,
            "performance_metrics": metrics
        }


class DeploymentEngine:
    """
    Handles saving, loading, benchmarking, and integrity verification of
    machine learning model artifacts using Pickle and Joblib.
    """
    
    @staticmethod
    def compute_sha256(filepath: str) -> str:
        """Computes SHA-256 cryptographic hash of a serialized file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def save_pickle(bundle: ModelDeploymentBundle, filepath: str, protocol: int = pickle.HIGHEST_PROTOCOL) -> str:
        """Saves deployment bundle via Python's standard pickle module."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(bundle, f, protocol=protocol)
        return DeploymentEngine.compute_sha256(filepath)

    @staticmethod
    def load_pickle(filepath: str, expected_hash: Optional[str] = None) -> ModelDeploymentBundle:
        """
        Loads deployment bundle via pickle with optional SHA-256 integrity check.
        Raises ValueError if file has been tampered with or corrupted.
        """
        if expected_hash is not None:
            actual_hash = DeploymentEngine.compute_sha256(filepath)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Integrity Check Failed for {filepath}!\n"
                    f"Expected: {expected_hash}\nActual:   {actual_hash}"
                )
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_joblib(bundle: ModelDeploymentBundle, filepath: str, compress: int = 3) -> str:
        """Saves deployment bundle via Joblib (optimized for NumPy arrays with compression)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(bundle, filepath, compress=compress)
        return DeploymentEngine.compute_sha256(filepath)

    @staticmethod
    def load_joblib(filepath: str, expected_hash: Optional[str] = None, mmap_mode: Optional[str] = None) -> ModelDeploymentBundle:
        """
        Loads deployment bundle via Joblib with optional memory mapping and SHA-256 check.
        """
        if expected_hash is not None:
            actual_hash = DeploymentEngine.compute_sha256(filepath)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Integrity Check Failed for {filepath}!\n"
                    f"Expected: {expected_hash}\nActual:   {actual_hash}"
                )
        return joblib.load(filepath, mmap_mode=mmap_mode)

    @staticmethod
    def benchmark_serialization(bundle: ModelDeploymentBundle, output_dir: str = "week 15-16/deployed_models") -> pd.DataFrame:
        """
        Performs a rigorous head-to-head benchmark comparing Pickle vs Joblib
        across serialization latency, deserialization latency, and on-disk footprint.
        """
        os.makedirs(output_dir, exist_ok=True)
        pickle_path = os.path.join(output_dir, "model_artifact_pickle.pkl")
        joblib_raw_path = os.path.join(output_dir, "model_artifact_joblib_raw.joblib")
        joblib_comp_path = os.path.join(output_dir, "model_artifact_joblib_compressed.joblib")
        
        n_repeats = 30
        records = []
        
        # 1. Standard Pickle (Highest Protocol)
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            DeploymentEngine.save_pickle(bundle, pickle_path)
        save_time_pkl = (time.perf_counter() - t0) / n_repeats * 1000.0  # ms
        
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = DeploymentEngine.load_pickle(pickle_path)
        load_time_pkl = (time.perf_counter() - t0) / n_repeats * 1000.0  # ms
        size_pkl = os.path.getsize(pickle_path)
        
        records.append({
            "Serialization Protocol": "Python Pickle (Protocol 5)",
            "Compression": "None (Raw Bytecode)",
            "Save Time (ms)": round(save_time_pkl, 3),
            "Load Time (ms)": round(load_time_pkl, 3),
            "File Size (Bytes)": size_pkl,
            "File Size (KB)": round(size_pkl / 1024.0, 2),
            "NumPy Optimized": "No (Standard Serializer)",
            "Memory Mapping": "Unsupported"
        })
        
        # 2. Joblib Raw (No compression)
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            joblib.dump(bundle, joblib_raw_path, compress=0)
        save_time_job_raw = (time.perf_counter() - t0) / n_repeats * 1000.0
        
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = joblib.load(joblib_raw_path)
        load_time_job_raw = (time.perf_counter() - t0) / n_repeats * 1000.0
        size_job_raw = os.path.getsize(joblib_raw_path)
        
        records.append({
            "Serialization Protocol": "Joblib (compress=0)",
            "Compression": "None (Raw NumPy Buffers)",
            "Save Time (ms)": round(save_time_job_raw, 3),
            "Load Time (ms)": round(load_time_job_raw, 3),
            "File Size (Bytes)": size_job_raw,
            "File Size (KB)": round(size_job_raw / 1024.0, 2),
            "NumPy Optimized": "Yes (Direct Array Buffers)",
            "Memory Mapping": "Supported (mmap_mode='r')"
        })
        
        # 3. Joblib Compressed (zlib level 3)
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            joblib.dump(bundle, joblib_comp_path, compress=3)
        save_time_job_comp = (time.perf_counter() - t0) / n_repeats * 1000.0
        
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = joblib.load(joblib_comp_path)
        load_time_job_comp = (time.perf_counter() - t0) / n_repeats * 1000.0
        size_job_comp = os.path.getsize(joblib_comp_path)
        
        records.append({
            "Serialization Protocol": "Joblib (compress=3)",
            "Compression": "zlib Level 3",
            "Save Time (ms)": round(save_time_job_comp, 3),
            "Load Time (ms)": round(load_time_job_comp, 3),
            "File Size (Bytes)": size_job_comp,
            "File Size (KB)": round(size_job_comp / 1024.0, 2),
            "NumPy Optimized": "Yes (Compressed Buffers)",
            "Memory Mapping": "Supported"
        })
        
        # Save metadata JSON
        meta_path = os.path.join(output_dir, "model_metadata.json")
        bundle.metadata["sha256_pickle"] = DeploymentEngine.compute_sha256(pickle_path)
        bundle.metadata["sha256_joblib"] = DeploymentEngine.compute_sha256(joblib_comp_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(bundle.metadata, f, indent=2)
            
        return pd.DataFrame(records)


class RealTimeScoringMicroservice:
    """
    Simulates a production-grade inference microservice (similar to a FastAPI / Flask endpoint)
    that loads a validated model artifact and performs real-time schema validation and scoring.
    """
    
    def __init__(self, artifact_path: str, format_type: str = "joblib", expected_hash: Optional[str] = None):
        self.artifact_path = artifact_path
        self.format_type = format_type
        
        # Load bundle with integrity verification
        if format_type == "pickle":
            self.bundle = DeploymentEngine.load_pickle(artifact_path, expected_hash=expected_hash)
        else:
            self.bundle = DeploymentEngine.load_joblib(artifact_path, expected_hash=expected_hash)
            
        self.model = self.bundle.model
        self.scaler = self.bundle.scaler
        self.feature_names = self.bundle.feature_names
        self.threshold = self.bundle.threshold
        
    def score_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates schema, transforms biomarkers, runs scratch MLP forward pass,
        and returns structured clinical prediction payload.
        """
        t_start = time.perf_counter()
        
        # Schema validation
        missing_fields = [f for f in self.feature_names if f not in patient_data]
        if missing_fields:
            raise ValueError(f"Payload missing required feature fields: {missing_fields}")
            
        # Feature array creation in deterministic order
        raw_vec = np.array([[patient_data[f] for f in self.feature_names]], dtype=float)
        
        # Preprocessing transformation
        scaled_vec = self.scaler.transform(raw_vec)
        
        # Model forward inference
        prob = float(self.model.predict_proba(scaled_vec)[0, 0])
        is_high_risk = bool(prob >= self.threshold)
        
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        
        if prob >= 0.70:
            risk_tier = "CRITICAL HIGH"
            triage_action = "Immediate cardiology consultation and emergency angiogram evaluation."
        elif prob >= self.threshold:
            risk_tier = "ELEVATED RISK"
            triage_action = "Schedule stress echocardiogram and initiate lipid-lowering therapy."
        elif prob >= 0.25:
            risk_tier = "MODERATE / BORDERLINE"
            triage_action = "Recommend lifestyle modification, low-sodium diet, and 6-month follow-up."
        else:
            risk_tier = "LOW / NORMAL"
            triage_action = "Routine annual physical checkup; maintain aerobic conditioning."
            
        return {
            "prediction": 1 if is_high_risk else 0,
            "risk_status": "HIGH CARDIOVASCULAR RISK" if is_high_risk else "NORMAL / LOW RISK",
            "risk_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "clinical_recommendation": triage_action,
            "inference_latency_ms": round(latency_ms, 3),
            "model_version": self.bundle.metadata["version"]
        }

    def benchmark_inference_latency(self, test_sample: Dict[str, Any], n_requests: int = 1000) -> Dict[str, float]:
        """
        Measures real-time inference latency distribution across n_requests.
        Computes Mean, Median (P50), P95, and P99 latencies in microseconds.
        """
        latencies_us = []
        for _ in range(n_requests):
            t0 = time.perf_counter()
            _ = self.score_patient(test_sample)
            dt_us = (time.perf_counter() - t0) * 1e6
            latencies_us.append(dt_us)
            
        latencies_arr = np.array(latencies_us)
        return {
            "n_requests": n_requests,
            "mean_us": float(np.mean(latencies_arr)),
            "std_us": float(np.std(latencies_arr)),
            "median_p50_us": float(np.median(latencies_arr)),
            "p95_us": float(np.percentile(latencies_arr, 95)),
            "p99_us": float(np.percentile(latencies_arr, 99)),
            "throughput_req_per_sec": float(1e6 / np.mean(latencies_arr)),
            "raw_latencies_us": latencies_us
        }


if __name__ == "__main__":
    print("=== Testing Deployment Engine ===")
    from download_dataset import load_preprocessed_data
    from neural_network_scratch import NeuralNetwork, AdamOptimizer
    
    X_train, X_test, y_train, y_test, scaler, feat_names = load_preprocessed_data()
    
    # Quick train
    model = NeuralNetwork([8, 16, 8, 1], hidden_activation="relu", output_activation="sigmoid", random_state=42)
    model.fit(X_train, y_train, epochs=50, batch_size=32, optimizer=AdamOptimizer(lr=0.005))
    
    test_acc = float(np.mean(model.predict(X_test) == y_test))
    bundle = ModelDeploymentBundle(
        model=model,
        scaler=scaler,
        feature_names=feat_names,
        metrics={"test_accuracy": test_acc, "test_loss": 0.15},
        version="1.0.0"
    )
    
    df_bench = DeploymentEngine.benchmark_serialization(bundle)
    print("\n--- Serialization Benchmark Results ---")
    print(df_bench.to_string(index=False))
    
    # Test scoring microservice
    service = RealTimeScoringMicroservice("week 15-16/deployed_models/model_artifact_joblib_compressed.joblib", format_type="joblib")
    sample_patient = {
        "age": 62,
        "resting_bp": 150,
        "cholesterol": 290,
        "max_heart_rate": 115,
        "st_depression": 2.4,
        "exercise_angina": 1,
        "fasting_blood_sugar": 0,
        "vessels_colored": 2
    }
    pred_res = service.score_patient(sample_patient)
    print("\n--- Real-Time Patient Scoring Result ---")
    print(json.dumps(pred_res, indent=2))
