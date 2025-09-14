# login_detector.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

MODEL_PATH = "models/login_pipeline.joblib"

st.set_page_config(page_title="Suspicious Login Detector", page_icon="🔐", layout="centered")
st.title("🔐 Suspicious Login Detector (Logistic Regression)")

def load_pipeline(path=MODEL_PATH):
    if os.path.exists(path):
        bundle = joblib.load(path)
        return bundle["pipeline"]
    return None

pipeline = load_pipeline()

if pipeline is None:
    st.warning("No trained model found. Please run `train_login_model.py` first (or wait while training).")
    if st.button("Train model now (quick demo)"):
        # quick inline training step (small sample) -- for beginners convenience
        from train_login_model import train_and_save
        train_and_save(n_samples=2000)
        pipeline = load_pipeline()
        st.experimental_rerun()
else:
    st.success("Model loaded.")

st.markdown("Enter a login attempt and the model will predict the probability it's malicious.")

st.sidebar.header("Input features")
failed_attempts = st.sidebar.number_input("Failed attempts", min_value=0, max_value=50, value=0, step=1)
time_of_login = st.sidebar.slider("Login hour", 0, 23, 14)
geo_distance = st.sidebar.number_input("Geo distance from last known location (km)", min_value=0.0, value=10.0, step=1.0, format="%.1f")
device_known = st.sidebar.selectbox("Device known?", ["Yes", "No"])
device_known = 1 if device_known == "Yes" else 0

col1, col2 = st.columns(2)
with col1:
    st.write("### Input summary")
    st.write(f"Failed attempts: **{failed_attempts}**")
    st.write(f"Login hour: **{time_of_login}**")
with col2:
    st.write(f"Geo distance (km): **{geo_distance:.1f}**")
    st.write(f"Device known: **{'Yes' if device_known==1 else 'No'}**")

if pipeline is not None:
    input_df = pd.DataFrame([{
        "failed_attempts": failed_attempts,
        "time_of_login": time_of_login,
        "geo_distance": geo_distance,
        "device_known": device_known
    }])

    prob = pipeline.predict_proba(input_df)[0, 1]
    default_threshold = 0.5
    st.metric("Malicious probability", f"{prob:.2f}")

    thr = st.slider("Decision threshold", 0.0, 1.0, default_threshold, 0.01)
    pred_label = "🚨 Suspicious" if prob >= thr else "✅ Legitimate"
    if prob >= thr:
        st.error(f"Predicted: {pred_label}")
    else:
        st.success(f"Predicted: {pred_label}")

    # Show model internals: approximate contributions
    st.write("### Approximate feature contributions (on scaled features):")
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    scaled_vals = scaler.transform(input_df)[0]
    coefs = clf.coef_[0]
    contributions = coefs * scaled_vals
    contrib_df = pd.DataFrame({
        "feature": ["failed_attempts", "time_of_login", "geo_distance", "device_known"],
        "raw_value": input_df.iloc[0].values,
        "scaled_value": np.round(scaled_vals, 3),
        "coef": np.round(coefs, 3),
        "contribution": np.round(contributions, 3)
    }).sort_values(by="contribution", key=abs, ascending=False)
    st.table(contrib_df)

    if st.checkbox("Show test set evaluation (internal)"):
        bundle = joblib.load(MODEL_PATH)
        X_test = bundle["X_test"]
        y_test = bundle["y_test"]
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred_default = (y_proba >= default_threshold).astype(int)
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        st.text("Classification report (threshold=0.5):")
        st.text(classification_report(y_test, y_pred_default))
        st.write("ROC AUC:", roc_auc_score(y_test, y_proba))
        st.write("Confusion matrix:")
        st.write(confusion_matrix(y_test, y_pred_default))
