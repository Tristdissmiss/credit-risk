import streamlit as st 
import pandas as pd
import joblib

# Load trained model
model = joblib.load("credit_model.pkl")

st.title("💳 Credit Risk Predictor")
st.write("Enter customer details below to assess default risk.")

# Input sliders
RevolvingUtilization = st.slider("Revolving Utilization of Unsecured Lines", 0.0, 2.0, 0.5)
age = st.slider("Age", 18, 100, 35) 
past_due_30_59 = st.slider("Times 30-59 Days Past Due", 0, 10, 0)
DebtRatio = st.slider("Debt Ratio", 0.0, 5.0, 1.0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
OpenCreditLines = st.slider("Open Credit Lines and Loans", 0, 30, 5)
past_due_90 = st.slider("Times 90+ Days Late", 0, 10, 0)
RealEstateLoans = st.slider("Real Estate Loans or Lines", 0, 10, 1)
past_due_60_89 = st.slider("Times 60-89 Days Past Due", 0, 10, 0)
Dependents = st.slider("Number of Dependents", 0, 10, 0)

# Predict button
if st.button("Predict Credit Risk"): 
    input_data = pd.DataFrame([[
        RevolvingUtilization,
        age,
        past_due_30_59,
        DebtRatio,
        MonthlyIncome,
        OpenCreditLines,
        past_due_90,
        RealEstateLoans,
        past_due_60_89,
        Dependents
    ]], columns=[
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ High Risk — {probability*100:.2f}% chance of default")
    else:
        st.success(f"✅ Low Risk — {probability*100:.2f}% chance of default")
