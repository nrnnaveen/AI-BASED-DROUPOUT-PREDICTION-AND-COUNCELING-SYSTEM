# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

MODEL_PATH = "dropout_model.joblib"

# ----------------------------
# Train & Evaluate
# ----------------------------
def train_and_evaluate(df: pd.DataFrame):
    # Clean column names
    df.columns = df.columns.str.strip()

    # Target column
    target_col = "dropout_risk"
    if target_col not in df.columns:
        raise ValueError(f"Dataset missing '{target_col}' column for training")

    # Features and target
    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]

    # Convert categorical columns to numeric
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Predictions & metrics
    preds = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 3),
        "precision": round(precision_score(y_test, preds, zero_division=0), 3),
        "recall": round(recall_score(y_test, preds, zero_division=0), 3),
        "f1_score": round(f1_score(y_test, preds, zero_division=0), 3)
    }

    return model, metrics

# ----------------------------
# Save & Load Model
# ----------------------------
def save_model(model, path: str = MODEL_PATH):
    joblib.dump(model, path)

def load_model(path: str = MODEL_PATH):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

# ----------------------------
# Predict on Dataset
# ----------------------------
def predict_df(model, df: pd.DataFrame):
    df.columns = df.columns.str.strip()
    X = df.copy()

    # Keep student_id if available
    student_ids = X["student_id"] if "student_id" in X.columns else None

    # Drop label if exists
    X = X.drop(columns=["dropout_risk"], errors="ignore")

    # Convert categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Make sure columns match training set
    # This prevents errors if df has extra/missing columns
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        for col in model_features:
            if col not in X.columns:
                X[col] = 0
        X = X[model_features]  # reorder columns

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(X))

    result = df.copy()
    result["predicted_dropout"] = preds
    result["risk_proba"] = probs

    # Restore student_id column if it existed
    if student_ids is not None:
        result["student_id"] = student_ids

    return result
