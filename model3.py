# model3.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

MODEL_PATH = "dropout_model.joblib"
FEATURES_PATH = "model_features.joblib"  # Save feature list separately

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

    # Save feature names
    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, FEATURES_PATH)

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

def load_features(path: str = FEATURES_PATH):
    try:
        return joblib.load(path)
    except:
        return None

# ----------------------------
# Predict on Dataset
# ----------------------------
def predict_df(model, df, feature_columns=None):
    """
    Predict using a trained model, safely handling missing or extra columns in the input DataFrame.

    Args:
        model: Trained scikit-learn model
        df: pandas DataFrame with input data
        feature_columns: list of features used during training (optional)

    Returns:
        DataFrame with an additional 'prediction' column
    """
    if feature_columns is None:
        # Try to load saved feature list
        feature_columns = load_features()
        if feature_columns is None:
            raise ValueError("Feature columns not provided and could not be loaded from disk.")

    # Add missing columns with default value 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Select only the features used in training
    df_aligned = df[feature_columns]

    # Make predictions
    df['prediction'] = model.predict(df_aligned)

    return df
