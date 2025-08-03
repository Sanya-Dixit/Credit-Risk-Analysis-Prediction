import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Credit Risk Prediction", layout="centered")

st.title("Credit Risk Prediction Demo")
st.write(
    """
    This app predicts whether a loan applicant is a **low risk** or **high risk** for credit default,
    based on their financial information, using a machine learning model trained on LendingClub data.
    """
)

# Load trained model
@st.cache_resource
def load_model():
    with open("../models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Define input fields
st.subheader("Applicant Financial Information")
annual_inc = st.number_input("Annual Income ($)", min_value=0.0, max_value=1e7, step=1000.0)
fico_score = st.number_input("FICO Score", min_value=300, max_value=900, step=1)
loan_amnt = st.number_input("Loan Amount ($)", min_value=0.0, max_value=1e6, step=500.0)
purpose = st.selectbox("Loan Purpose", [
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
    "small_business", "car", "wedding", "medical", "vacation", "house", "other"
])
home_ownership = st.selectbox("Home Ownership", [
    "RENT", "OWN", "MORTGAGE", "OTHER", "NONE", "ANY"
])

# Prediction logic
if st.button("Predict Credit Risk"):
    # Prepare input for model: order must match training!
    input_dict = {
        'annual_inc': annual_inc,
        'loan_amnt': loan_amnt,
        'fico_score': fico_score,
        'purpose': purpose,
        'home_ownership': home_ownership
    }
    # One-hot encoding & scaling should be same as training
    # For demo, assume model expects [annual_inc, loan_amnt, fico_score] + one-hot purpose/home_ownership
    # You may need to use the same preprocessing pipeline as in scripts/preprocess.py
    # Here, show a simple version:
    input_df = pd.DataFrame([input_dict])
    
    # If you saved the pipeline, load and transform input
    try:
        with open("../models/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        X_input = preprocessor.transform(input_df)
    except Exception:
        # If no pipeline, try with numeric only
        X_input = np.array([[annual_inc, loan_amnt, fico_score]])
    
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][pred]
    risk_label = "Low Risk" if pred == 0 else "High Risk"
    
    st.markdown(f"## Prediction: **{risk_label}**")
    st.markdown(f"Model confidence: **{proba:.2f}**")
    if risk_label == "High Risk":
        st.error("Warning: Applicant is likely to default.")
    else:
        st.success("Applicant is unlikely to default.")

st.markdown("---")
st.caption("Built with LendingClub data & scikit-learn/XGBoost. For demonstration purposes only.")
