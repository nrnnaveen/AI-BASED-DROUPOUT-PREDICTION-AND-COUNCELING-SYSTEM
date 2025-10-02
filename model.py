# model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    classification_report
)
import joblib

def build_pipeline():
    numeric_features = [
        "attendance_pct",
        "avg_assignment_pct",
        "avg_test_pct",
        "fee_delay_days",
        "num_attempts",
        "prior_arrears",
        "engagement_score"
    ]
    categorical_features = ["gender", "scholarship"]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    clf = Pipeline(steps=[
        ("pre", preprocessor),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    return clf

def train_and_evaluate(df, label_col="dropout_risk", test_size=0.2, random_state=42):
    df = df.copy()
    X = df.drop(columns=[label_col, "student_id"]) if "student_id" in df.columns else df.drop(columns=[label_col])
    y = df[label_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "report": classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    }
    return model, metrics

def save_model(model, path="dropout_model.joblib"):
    joblib.dump(model, path)
    return path

def load_model(path="dropout_model.joblib"):
    return joblib.load(path)

def predict_df(model, df):
    # returns df with risk_proba and risk_label (binary threshold 0.5)
    X = df.copy()
    if "student_id" in X.columns:
        ids = X["student_id"]
        X = X.drop(columns=["student_id"])
    else:
        ids = None

    proba = model.predict_proba(X)[:, 1]
    label = (proba >= 0.5).astype(int)
    out = df.copy()
    out["risk_proba"] = proba
    out["risk_label"] = label
    return out
