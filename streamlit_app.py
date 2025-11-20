# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# --- Config / Title ---
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Credit Card Fraud Detection — Demo")
st.markdown("Upload transactions CSV and the model will predict fraud (1) or not fraud (0).")

# --- Load model & scaler ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/fraud_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../models/scaler.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error("Model or scaler not found. Make sure ../models/fraud_model.pkl and ../models/scaler.pkl exist.")
    st.stop()

# --- Sidebar: About + sample download ---
st.sidebar.header("About")
st.sidebar.write("Model: RandomForest (trained on creditcard.csv)")
st.sidebar.write("Scaler: StandardScaler (must be applied to features)")

if st.sidebar.button("Download sample input CSV"):
    # create sample with required columns (exact columns from training)
    cols = list(pd.read_csv("../data/creditcard.csv", nrows=1).columns)
    sample_df = pd.DataFrame(columns=cols)  # empty sample with columns
    sample_csv = sample_df.to_csv(index=False)
    st.sidebar.download_button("Download sample", sample_csv, file_name="sample_creditcard_input.csv", mime="text/csv")

uploaded_file = st.file_uploader("Upload your transaction data (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File uploaded successfully!")
        # your prediction code here
        
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a file to continue")
    st.stop()

# --- Read data & preprocess ---
df = pd.read_csv(uploaded_file)
st.write("Uploaded data preview:", df.head())

# Check presence of Class column
has_label = "Class" in df.columns

# Keep original copy for display
orig_df = df.copy()

# Drop target if present
if has_label:
    y_true = df["Class"].values
    X = df.drop("Class", axis=1)
else:
    X = df.copy()

# Remove non-feature columns if exist (like 'Time' if you didn't use it)
# NOTE: Use the same features as used during training.
# Here, we assume you trained with all columns except 'Class'.
# If you trained dropping 'Time', uncomment the next line.
# X = X.drop(columns=['Time'], errors='ignore')

# --- Scale features ---
try:
    X_scaled = scaler.transform(X)
except Exception as e:
    st.error("Scaling failed. Make sure uploaded CSV has same columns and order as training data.")
    st.stop()

# --- Predict ---
preds = model.predict(X_scaled)
# If model supports predict_proba, show probability of class 1
probs = None
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_scaled)[:, 1]

# Append results
result_df = orig_df.copy()
result_df["Predicted_Class"] = preds
if probs is not None:
    result_df["Fraud_Probability"] = probs.round(4)

st.write("Predictions preview:")
st.dataframe(result_df.head().style.background_gradient(subset=["Predicted_Class"], cmap="coolwarm"))

# --- If true labels present: show metrics ---
if has_label:
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    st.markdown("### Model Performance on uploaded data (if labels provided)")
    st.write(f"- Precision: **{precision:.4f}**")
    st.write(f"- Recall: **{recall:.4f}**")
    st.write(f"- F1-score: **{f1:.4f}**")
    st.write("Full classification report:")
    st.text(classification_report(y_true, preds, zero_division=0))

# --- Allow user to download predictions CSV ---
@st.cache_data
def convert_df(df_in):
    return df_in.to_csv(index=False).encode('utf-8')

csv = convert_df(result_df)
st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")

st.success("Done — predictions generated ✅")
