import streamlit as st

st.title("Credit Risk Prediction Demo")

income = st.number_input("Annual Income")
credit_score = st.number_input("FICO Score")
loan_amount = st.number_input("Loan Amount")

if st.button("Predict"):
    # Dummy prediction logic
    risk = "Low Risk" if income > 50000 and credit_score > 700 else "High Risk"
    confidence = 0.8 if risk == "Low Risk" else 0.6
    st.markdown(f"### Prediction: **{risk}**")
    st.markdown(f"Model confidence: {confidence:.2f}")
