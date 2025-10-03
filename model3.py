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
def predict_df(model, df):
    """
    Predict using a trained model, safely handling missing or extra columns in the input DataFrame.

    Args:
        model: Trained scikit-learn model.
        df: pandas DataFrame with input data.

    Returns:
        DataFrame with an additional 'prediction' column.
    """
    import pandas as pd
    import numpy as np

    # Get expected features from the model
    try:
        expected_features = model.feature_names_in_
    except AttributeError:
        raise ValueError("The model does not have 'feature_names_in_'. Make sure it is a scikit-learn model.")

    # Add any missing columns with default value 0
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    # Select only the expected features (ignore extra columns)
    df_aligned = df[expected_features]

    # Make predictions
    predictions = model.predict(df_aligned)

    # Add predictions to the original DataFrame
    df['prediction'] = predictions

    return df
