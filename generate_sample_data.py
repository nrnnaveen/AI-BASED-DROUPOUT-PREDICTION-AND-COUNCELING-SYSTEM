# generate_sample_data.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate(n=2000, seed=42):
    np.random.seed(seed)
    ids = np.arange(100000, 100000 + n)
    # features
    attendance = np.clip(np.random.normal(80, 12, n), 30, 100).round(1)
    avg_assignment = np.clip(np.random.normal(65, 18, n), 0, 100).round(1)
    avg_test = np.clip(np.random.normal(60, 20, n), 0, 100).round(1)
    fee_delay_days = np.random.poisson(2, n)  # days late in fee payment
    attempts = np.random.choice([1,1,1,2,2,3,4,5], size=n, p=[0.45,0.2,0.1,0.1,0.06,0.05,0.03,0.01])
    prior_arrears = np.random.poisson(0.2, n)
    engagement = np.clip(np.random.normal(70, 20, n), 0, 100).round(1)
    gender = np.random.choice(['M','F'], size=n, p=[0.6,0.4])
    scholarship = np.random.choice([0,1], size=n, p=[0.8,0.2])

    # Create synthetic risk label: higher risk when attendance low, test low, fee delay high, attempts high, engagement low
    risk_score = (
        (100 - attendance) * 0.25 +
        (100 - avg_test) * 0.25 +
        (fee_delay_days * 2.5) +
        (attempts - 1) * 5 +
        (50 - engagement) * 0.2 +
        prior_arrears * 3
    )
    # add noise
    risk_score += np.random.normal(0, 5, n)
    # threshold to binary label: 1 -> dropout_risk
    label = (risk_score > np.percentile(risk_score, 70)).astype(int)

    df = pd.DataFrame({
        "student_id": ids,
        "gender": gender,
        "scholarship": scholarship,
        "attendance_pct": attendance,
        "avg_assignment_pct": avg_assignment,
        "avg_test_pct": avg_test,
        "fee_delay_days": fee_delay_days,
        "num_attempts": attempts,
        "prior_arrears": prior_arrears,
        "engagement_score": engagement,
        "dropout_risk": label
    })
    return df

if __name__ == "__main__":
    df = generate(2000)
    df.to_csv("sample_students.csv", index=False)
    print("Wrote sample_students.csv (2000 rows)")
