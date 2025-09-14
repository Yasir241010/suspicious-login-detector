# train_login_model.py
"""
Train a Logistic Regression pipeline for suspicious login detection
and save the trained pipeline to models/login_pipeline.joblib
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
import joblib

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "login_pipeline.joblib")
RANDOM_STATE = 42

def generate_synthetic_logins(n=5000, seed=RANDOM_STATE):
    np.random.seed(seed)
    # features
    failed_attempts = np.random.poisson(1, n)           # mostly 0-3
    time_of_login = np.random.randint(0, 24, n)         # hour of day 0-23
    geo_distance = np.random.exponential(200, n)        # km (skewed)
    device_known = np.random.choice([0,1], n, p=[0.3,0.7])  # 1=known

    # create a realistic "malicious score" and label
    suspicious_time = ((time_of_login < 6) | (time_of_login > 22)).astype(int)
    long_distance = (geo_distance > 1000).astype(int)
    # weighted sum (tunable)
    score = (
        0.45 * failed_attempts +
        0.6 * suspicious_time +
        0.5 * long_distance +
        0.8 * (device_known == 0)
    )
    # convert to probability with sigmoid
    prob = 1 / (1 + np.exp(-(score - 0.8)))  # shift so baseline not too high
    labels = (prob > np.random.rand(n)).astype(int)

    df = pd.DataFrame({
        "failed_attempts": failed_attempts,
        "time_of_login": time_of_login,
        "geo_distance": geo_distance,
        "device_known": device_known,
        "malicious": labels
    })
    return df

def train_and_save(n_samples=5000):
    df = generate_synthetic_logins(n_samples)
    X = df[["failed_attempts", "time_of_login", "geo_distance", "device_known"]]
    y = df["malicious"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE))
    ])

    pipeline.fit(X_train, y_train)

    # evaluation
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test
    }, MODEL_PATH)
    print(f"Saved pipeline to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save()
