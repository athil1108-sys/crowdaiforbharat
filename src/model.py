"""
ML Model Module
===============
Binary classification model to predict congestion 10-15 minutes ahead.

Architecture: Logistic Regression (fast, interpretable, perfect for hackathon)
- Chosen over LSTM for speed of training and explainability
- Still captures the key signal: rising density + falling velocity → congestion

Pipeline:
  1. Load simulated data
  2. Engineer features
  3. Train/test split (80/20, stratified)
  4. Train Logistic Regression with class weights (handles imbalance)
  5. Evaluate: Accuracy, Precision, Recall, F1, Confusion Matrix
  6. Save model to disk for dashboard use
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

# Allow imports from both direct run and package import
try:
    from src.simulate_data import generate_training_dataset
    from src.features import engineer_features, get_feature_columns
except ImportError:
    from simulate_data import generate_training_dataset
    from features import engineer_features, get_feature_columns

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "congestion_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def prepare_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data → engineer features → split into train/test.
    Returns X_train, X_test, y_train, y_test.
    """
    print("📊 Generating training dataset...")
    raw_df = generate_training_dataset(seed=seed)

    print("🔧 Engineering features...")
    featured_df = engineer_features(raw_df)

    # Drop rows with NaN (from rolling/shift operations)
    featured_df = featured_df.dropna()

    feature_cols = get_feature_columns()
    X = featured_df[feature_cols].values
    y = featured_df["pre_congestion_label"].values

    print(f"   Total samples: {len(X)}")
    print(f"   Positive (pre-congestion): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Negative (normal): {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> tuple[LogisticRegression, StandardScaler]:
    """
    Train a Logistic Regression model with:
    - StandardScaler for feature normalization
    - class_weight='balanced' to handle class imbalance
    - C=1.0 regularization to prevent overfitting
    """
    print("\n🧠 Training Logistic Regression model...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",  # Handles imbalanced classes
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(X_train_scaled, y_train)

    print("   ✅ Model trained successfully!")
    return model, scaler


def evaluate_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate model and print comprehensive metrics.
    Returns dict of metric values.
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("📈 MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   {cm[0][0]:5d} (TN)  {cm[0][1]:5d} (FP)")
    print(f"   {cm[1][0]:5d} (FN)  {cm[1][1]:5d} (TP)")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Pre-Congestion"]))

    # Feature importance (Logistic Regression coefficients)
    feature_cols = get_feature_columns()
    coefficients = model.coef_[0]
    print("   Feature Importance (coefficients):")
    for name, coef in sorted(zip(feature_cols, coefficients), key=lambda x: abs(x[1]), reverse=True):
        direction = "↑ risk" if coef > 0 else "↓ risk"
        print(f"     {name:30s} → {coef:+.4f} ({direction})")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def save_model(model: LogisticRegression, scaler: StandardScaler):
    """Save trained model and scaler to disk, then upload to S3 if available."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n💾 Model saved to {MODEL_PATH}")
    print(f"💾 Scaler saved to {SCALER_PATH}")

    # Upload to S3 if available
    try:
        from src.aws_storage import upload_model_to_s3, is_s3_available
        if is_s3_available():
            s3_model = upload_model_to_s3(MODEL_PATH, "congestion_model.pkl")
            s3_scaler = upload_model_to_s3(SCALER_PATH, "scaler.pkl")
            if s3_model and s3_scaler:
                print("☁️  Model artifacts uploaded to Amazon S3")
            else:
                print("⚠️  S3 upload partially failed (local copy is fine)")
        else:
            print("ℹ️  S3 not available — using local storage only")
    except ImportError:
        print("ℹ️  AWS storage module not found — using local storage only")


def load_model() -> tuple[LogisticRegression, StandardScaler]:
    """Load trained model and scaler from disk."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def train_and_save(seed: int = 42) -> dict:
    """
    Full pipeline: generate data → train → evaluate → save.
    Returns evaluation metrics.
    """
    X_train, X_test, y_train, y_test = prepare_data(seed)
    model, scaler = train_model(X_train, y_train)
    metrics = evaluate_model(model, scaler, X_test, y_test)
    save_model(model, scaler)
    return metrics


if __name__ == "__main__":
    metrics = train_and_save()
    print("\n✅ Training pipeline complete!")
