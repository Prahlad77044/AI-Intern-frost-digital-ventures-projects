# app.py
import streamlit as st
import requests

st.title("Churn Prediction")
st.write("Fill in customer details to predict churn probability.")

# -------------------------------
# Input form
# -------------------------------
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.number_input("Senior Citizen (0 or 1)", min_value=0, max_value=1, value=0, step=1)
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1, step=1)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0, step=0.1)

    submitted = st.form_submit_button("Predict")

# -------------------------------
# Call API when submitted
# -------------------------------
if submitted:
    st.success("Form submitted successfully! Sending request to backend...")

    api_url = "http://127.0.0.1:8000/predict"
    payload = {
        "gender": gender,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges)
    }

    try:
        response = requests.post(api_url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        probability = data.get("probability", 0.0)
        prediction = data.get("prediction", "Unknown")
        threshold = data.get("threshold", 0.5)

        st.success(f"Prediction: **{prediction}**")
        st.metric("Churn Probability", f"{probability:.2%}")
        st.info(f"Threshold used: {threshold}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend API. Is the FastAPI server running on http://127.0.0.1:8000?")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The backend is taking too long to respond.")
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")