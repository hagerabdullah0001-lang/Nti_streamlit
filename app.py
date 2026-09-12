import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
features = joblib.load("features.pkl")

st.title("Churn Prediction App")

tenure = st.number_input("Tenure", min_value=0)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "No internet service", "Yes"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

if st.button("Predict"):
    tech_support_encoded = encoder["TechSupport"].transform([tech_support])[0]
    contract_encoded = encoder["Contract"].transform([contract])[0]

    input_data = pd.DataFrame([[
        tenure,
        tech_support_encoded,
        contract_encoded
    ]], columns=features)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is not likely to Churn")