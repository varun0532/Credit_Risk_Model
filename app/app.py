import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.inference import CreditRiskPredictor

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

st.title("Credit Risk Prediction System")

predictor = CreditRiskPredictor()

# -------------------------
# Personal & Financial
# -------------------------
st.subheader("Personal & Financial")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=75, value=28)

with col2:
    annual_income = st.number_input(
    "Annual Income (₹)",
    min_value=100000,
    max_value=100000000,
    value=1200000,
    step=50000
)

with col3:
    residence_type = st.selectbox("Residence Type", ["Owned", "Rented"])

# -------------------------
# Loan Details
# -------------------------
st.subheader("Loan Details")

col4, col5, col6, col7 = st.columns(4)

with col4:
    loan_amount = st.number_input("Loan Amount (₹)", value=2500000)

with col5:
    loan_tenure = st.number_input("Tenure (Months)", value=36)

with col6:
    loan_purpose = st.selectbox("Loan Purpose", ["Education", "Personal", "Business"])

with col7:
    loan_type = st.selectbox("Loan Type", ["Unsecured", "Secured"])


# -------------------------
# Derived Feature: Loan-to-Income
# -------------------------
loan_to_income_raw = loan_amount / annual_income
loan_to_income = min(round(loan_to_income_raw, 2), 10.0)

st.markdown(f"**Loan-to-Income Ratio:** {loan_to_income}")

if loan_to_income_raw > 10:
    st.error("Loan-to-Income ratio is extremely high. Application flagged as HIGH RISK.")
    st.stop()


# -------------------------
# Credit History
# -------------------------
st.subheader("Credit History")

col8, col9, col10, col11 = st.columns(4)

with col8:
    avg_dpd = st.number_input("Avg DPD", value=20)

with col9:
    delinquency_ratio = st.number_input("Delinquency Ratio (%)", value=30)

with col10:
    credit_utilization = st.number_input("Credit Utilization (%)", value=30)

with col11:
    open_accounts = st.number_input("Open Accounts", value=2)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Credit Risk"):
    input_data = {
        "age": age,
        "gender": "Male",
        "marital_status": "Single",
        "employment_status": "Salaried",
        "number_of_dependants": 1,
        "residence_type": residence_type,
        "years_at_current_address": 5,
        "city": "Bangalore",
        "state": "Karnataka",
        "zipcode": 560001,
        "loan_purpose": loan_purpose,
        "loan_type": loan_type,
        "sanction_amount": loan_amount,
        "processing_fee": 5000,
        "gst": 900,
        "net_disbursement": loan_amount - 5900,
        "loan_tenure_months": loan_tenure,
        "principal_outstanding": loan_amount,
        "bank_balance_at_application": 150000,
        "number_of_open_accounts": open_accounts,
        "number_of_closed_accounts": 4,
        "enquiry_count": 1,
        "credit_utilization_ratio": credit_utilization / 100,
        "loan_to_income": loan_to_income,
        "delinquency_ratio": delinquency_ratio / 100,
        "avg_dpd_per_delinquency": avg_dpd
    }

    result = predictor.predict(input_data)

    st.success(f"Risk Level: {result['risk_label']}")
    st.info(f"Probability of Default: {result['probability_of_default']}")
    st.metric("Credit Score", result["credit_score"])
    st.metric("Rating", result["rating"])
