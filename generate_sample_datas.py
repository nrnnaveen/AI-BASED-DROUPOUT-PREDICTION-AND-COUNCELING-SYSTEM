# generate_sample_data.py
import pandas as pd
import numpy as np
import random

def generate(n: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    random.seed(42)

    student_ids = [f"S{i:05d}" for i in range(1, n+1)]
    genders = np.random.choice(["Male", "Female"], size=n)
    scholarships = np.random.choice([0, 1], size=n, p=[0.7, 0.3])  # 30% scholarship holders

    attendance = np.clip(np.random.normal(75, 15, n), 30, 100)
    assignment_scores = np.clip(np.random.normal(70, 12, n), 20, 100)
    test_scores = np.clip(np.random.normal(65, 15, n), 10, 100)
    fee_delay = np.abs(np.random.poisson(3, n))
    attempts = np.random.randint(1, 5, n)
    arrears = np.random.randint(0, 4, n)
    engagement = np.clip(np.random.normal(60, 20, n), 0, 100)

    # Dropout Risk Logic (synthetic)
    dropout_probs = (
        (100 - attendance) * 0.02 +
        (50 - test_scores) * 0.02 +
        (fee_delay) * 0.01 +
        (arrears) * 0.05
    )
    dropout_probs = np.clip(dropout_probs, 0, 1)
    dropout_labels = np.random.binomial(1, dropout_probs)

    df = pd.DataFrame({
        "student_id": student_ids,
        "gender": genders,
        "scholarship": scholarships,
        "attendance_pct": attendance.round(2),
        "avg_assignment_pct": assignment_scores.round(2),
        "avg_test_pct": test_scores.round(2),
        "fee_delay_days": fee_delay,
        "num_attempts": attempts,
        "prior_arrears": arrears,
        "engagement_score": engagement.round(2),
        "dropout_risk": dropout_labels
    })

    return df
