# model3.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

MODEL_FILE = "dropout_model.joblib"
FEATURE_FILE = "features.joblib"
ENCODER_FILE = "label_encoder.joblib"

# ---------------- Train & Evaluate ----------------
def train_and_evaluate(df: pd.DataFrame):
    target = "dropout_risk"
    features = [col for col in df.columns if col not in ["student_id", target]]

    # Encode target
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )

    # Save everything
    save_model(model, features, le)

    metrics = {
        "accuracy": acc,
        "report": report,
        "classes": list(le.classes_)
    }
    return model, metrics


# ---------------- Save & Load ----------------
def save_model(model, features, label_encoder):
    joblib.dump(model, MODEL_FILE)
    joblib.dump(features, FEATURE_FILE)
    joblib.dump(label_encoder, ENCODER_FILE)


def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None


def load_features():
    if os.path.exists(FEATURE_FILE):
        return joblib.load(FEATURE_FILE)
    return []


def load_encoder():
    if os.path.exists(ENCODER_FILE):
        return joblib.load(ENCODER_FILE)
    return None


# ---------------- Prediction ----------------
def predict_df(model, df: pd.DataFrame, feature_columns=None):
    if feature_columns is None:
        feature_columns = load_features()

    df = df.copy()

    # Ensure all required features exist (fill missing with 0)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Drop any extra columns not used in training
    X = df[feature_columns]

    # Predict
    y_pred = model.predict(X)

    # Decode labels if encoder exists
    le = load_encoder()
    if le:
        df["prediction"] = le.inverse_transform(y_pred)
    else:
        df["prediction"] = y_pred

    return df


