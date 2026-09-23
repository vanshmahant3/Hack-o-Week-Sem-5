import os
import numpy as np
import pandas as pd
from typing import Tuple

def generate_student_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generates a realistic, intuitive Student Lifestyle & Academic Performance dataset

    Contains 8 numerical features across 3 distinct student tiers:
    - Tier 1: High Achievers (n = 250)
    - Tier 2: Balanced Mainstream (n = 500)
    - Tier 3: At-Risk / Distracted (n = 250)
    """
    np.random.seed(random_state)
    
    cohorts = [
        {
            "name": "High Achievers",
            "tier_id": 1,
            "size": int(0.25 * n_samples),
            "study": (28.0, 4.0),
            "attendance": (94.0, 3.5),
            "sleep": (7.2, 0.7),
            "screen": (2.2, 0.6),
            "extra": (6.5, 2.0),
            "stress": (4.2, 1.2),
            "gpa": (3.82, 0.12),
            "assignments": (96.0, 2.5)
        },
        {
            "name": "Balanced Mainstream",
            "tier_id": 2,
            "size": int(0.50 * n_samples),
            "study": (16.0, 3.5),
            "attendance": (82.0, 5.0),
            "sleep": (7.5, 0.8),
            "screen": (4.5, 1.0),
            "extra": (8.0, 2.5),
            "stress": (5.5, 1.4),
            "gpa": (3.18, 0.22),
            "assignments": (84.0, 5.0)
        },
        {
            "name": "At-Risk / Distracted",
            "tier_id": 3,
            "size": int(0.25 * n_samples),
            "study": (7.0, 2.5),
            "attendance": (64.0, 7.0),
            "sleep": (5.6, 1.1),
            "screen": (7.5, 1.4),
            "extra": (3.2, 1.8),
            "stress": (8.2, 1.0),
            "gpa": (2.35, 0.30),
            "assignments": (62.0, 8.0)
        }
    ]
    
    rows = []
    student_id = 1001
    
    for cohort in cohorts:
        size = cohort["size"]
        for _ in range(size):
            study = float(np.clip(np.random.normal(*cohort["study"]), 1.0, 45.0))
            attendance = float(np.clip(np.random.normal(*cohort["attendance"]), 40.0, 100.0))
            sleep = float(np.clip(np.random.normal(*cohort["sleep"]), 3.5, 11.0))
            screen = float(np.clip(np.random.normal(*cohort["screen"]), 0.5, 14.0))
            extra = float(np.clip(np.random.normal(*cohort["extra"]), 0.0, 20.0))
            stress = float(np.clip(np.random.normal(*cohort["stress"]), 1.0, 10.0))
            gpa = float(np.clip(np.random.normal(*cohort["gpa"]), 1.0, 4.0))
            assignments = float(np.clip(np.random.normal(*cohort["assignments"]), 30.0, 100.0))
            
            rows.append({
                "student_id": f"STU_{student_id}",
                "study_hours_weekly": round(study, 1),
                "attendance_rate": round(attendance, 1),
                "sleep_hours_daily": round(sleep, 1),
                "screen_time_daily": round(screen, 1),
                "extracurricular_hours": round(extra, 1),
                "stress_level": round(stress, 1),
                "prior_gpa": round(gpa, 2),
                "assignment_completion_rate": round(assignments, 1),
                "academic_tier": cohort["name"],
                "tier_code": cohort["tier_id"]
            })
            student_id += 1
            
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

def prepare_dataset() -> str:
    """Acquires, caches, and returns the path to the Student dataset CSV."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "student_lifestyle_academic.csv")
    
    if not os.path.exists(csv_path):
        print(f"[*] Generating Student Lifestyle & Academic Performance Dataset (1,000 records)...")
        df = generate_student_dataset(n_samples=1000)
        df.to_csv(csv_path, index=False)
        print(f"[+] Dataset saved to: {csv_path}")
    else:
        print(f"[+] Dataset already exists: {csv_path}")
        df = pd.read_csv(csv_path)
        
    print(f"    - Records: {len(df)}")
    print(f"    - Features: {df.shape[1] - 3} numerical predictors")
    print(f"    - Target Tiers: {df['academic_tier'].value_counts().to_dict()}")
    return csv_path

if __name__ == "__main__":
    prepare_dataset()
