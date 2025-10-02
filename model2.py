# model_alt.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, classification_report
)
import joblib

def build_pipeline(
    numeric_features=None,
    categorical_features=None,
    n_estimators=200,
    random_state=42
):
    """
    Build preprocessing + RandomForest pipeline
    """
    if numeric_features is None:
        numeric_features = [
            "attendance_pct", "avg_assignment_pct", "avg_test_pct",
            "fee_delay_days", "num_attempts", "prior_arrears", "engagement_score"
        ]
    if categorical_features is None:
        categorical_features = ["gender", "scholarship"]

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        ))
    ])

    return clf

def train_and_evaluate(df, label_col="dropout_risk", test_size=0.2, random_state=42):
    """
    Train model and return trained pipeline + metrics
    """
    df = df.copy()
    X = df.drop(columns=[label_col, "student_id"], errors='ignore')
    y = df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    }

    return model, metrics

def save_model(model, path="dropout_model.joblib"):
    """Save model to disk"""
    joblib.dump(model, path)
    return path

def load_model(path="dropout_model.joblib"):
    """Load model from disk"""
    return joblib.load(path)

def predict_df(model, df, threshold=0.5):
    """
    Predict risk probabilities and binary label for a given DataFrame.
    Returns df with 'risk_proba' and 'risk_label'.
    """
    df_copy = df.copy()
    if "student_id" in df_copy.columns:
        X = df_copy.drop(columns=["student_id"])
    else:
        X = df_copy

    proba = model.predict_proba(X)[:, 1]
    label = (proba >= threshold).astype(int)

    df_copy["risk_proba"] = proba
    df_copy["risk_label"] = label
    return df_copy
