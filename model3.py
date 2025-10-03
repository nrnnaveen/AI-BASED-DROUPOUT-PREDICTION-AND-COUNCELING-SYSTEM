# mode3.py
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
    target_col = "dropout_risk"
    if target_col not in df.columns:
        raise ValueError(f"Dataset missing '{target_col}' column for training")

    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]

    # Convert categorical to numeric safely
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    metrics = {
        "accuracy": round(acc, 3),
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
    except:
        return None

# ----------------------------
# Predict on Dataset
# ----------------------------
def predict_df(model, df: pd.DataFrame):
    X = df.copy()
    
    # Keep student_id if available
    student_ids = X["student_id"] if "student_id" in X.columns else None

    # Drop label if exists
    X = X.drop(columns=["dropout_risk"], errors="ignore")

    # Safe categorical handling
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Align features with model
    if hasattr(model, "feature_names_in_"):
        missing_cols = set(model.feature_names_in_) - set(X.columns)
        for c in missing_cols:
            X[c] = 0
        X = X[model.feature_names_in_]

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["predicted_dropout"] = preds
    result["risk_proba"] = probs

    if student_ids is not None:
        result["student_id"] = student_ids

    return result

