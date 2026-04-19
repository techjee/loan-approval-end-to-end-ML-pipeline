import streamlit as st
import requests

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="💰",
    layout="wide"
)

API_URL = "http://localhost:8000/predict"

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }
    .sub-text {
        font-size: 18px;
        color: #4f4f4f;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .approved {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .rejected {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .info-card {
        background-color: #f7f9fc;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #d9e2ec;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">💰 Loan Approval Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Enter applicant details below to predict whether the loan is likely to be approved or rejected.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📋 Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=2000, step=500)
        loan_amount = st.number_input("Loan Amount", min_value=1, value=150, step=10)
        loan_term = st.number_input("Loan Amount Term", min_value=1, value=360, step=12)
        credit_history = st.selectbox("Credit History", [1, 0], help="1 = Good credit history, 0 = Poor credit history")

    predict_btn = st.button("🔍 Predict Loan Status", use_container_width=True)

with right_col:
    st.subheader("ℹ️ Quick Info")

    total_income = applicant_income + coapplicant_income
    ratio = round(total_income / loan_amount, 2) if loan_amount > 0 else 0

    st.markdown(f"""
        <div class="info-card">
            <b>Total Income</b><br>
            {total_income}
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="info-card">
            <b>Income to Loan Ratio</b><br>
            {ratio}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="info-card">
            <b>How prediction works</b><br>
            The system sends applicant details to the FastAPI backend, where the trained machine learning model predicts loan approval status.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="info-card">
            <b>Tip</b><br>
            Higher credit history and stable income usually improve approval chances.
        </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Prediction Logic
# -----------------------------
if predict_btn:
    payload = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            prediction_text = result.get("result", "Unknown")

            if prediction_text.lower() == "approved":
                st.markdown(
                    f'<div class="result-box approved">✅ Loan Status: {prediction_text}</div>',
                    unsafe_allow_html=True
                )
                st.balloons()
            else:
                st.markdown(
                    f'<div class="result-box rejected">❌ Loan Status: {prediction_text}</div>',
                    unsafe_allow_html=True
                )

            with st.expander("📦 View Submitted Data"):
                st.json(payload)

            with st.expander("🧠 Backend Response"):
                st.json(result)

        else:
            st.error(f"Prediction failed. Server returned status code {response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI backend. Make sure the API is running on http://localhost:8000")
    except requests.exceptions.Timeout:
        st.error("The request timed out. Please try again.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")