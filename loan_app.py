# loan_app_pro.py
import streamlit as st
st.set_page_config(page_title="Smart Loan Approval System Pro", layout="wide")  # Must be first

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import numpy as np

# ----------------------
# 1️⃣ Load and preprocess dataset
# ----------------------
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("train.csv")  # Replace with your path

    # Drop Loan_ID
    df = df.drop("Loan_ID", axis=1)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Features and target
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoders, df

X_scaled, y, scaler, label_encoders, df = load_and_preprocess()

# ----------------------
# 2️⃣ Train SVM models
# ----------------------
@st.cache_data
def train_svms(X, y):
    svm_linear = SVC(kernel="linear", probability=True, random_state=42).fit(X, y)
    svm_poly = SVC(kernel="poly", degree=3, probability=True, random_state=42).fit(X, y)
    svm_rbf = SVC(kernel="rbf", probability=True, random_state=42).fit(X, y)
    return svm_linear, svm_poly, svm_rbf

svm_linear, svm_poly, svm_rbf = train_svms(X_scaled, y)

# ----------------------
# 3️⃣ Streamlit App UI
# ----------------------
st.title("💡 Smart Loan Approval System ")
st.write("Compare Linear, Polynomial, and RBF SVM predictions in real-time.")

st.sidebar.header("Enter Applicant Details")

# Input fields
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Term (Months)", min_value=12)
credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ----------------------
# 4️⃣ Prepare input for prediction
# ----------------------
input_dict = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": 1 if credit_history == "Yes" else 0,
    "Property_Area": property_area
}

input_df = pd.DataFrame([input_dict])

# Encode categorical features
for col in label_encoders:
    if col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

# Scale numeric features
input_scaled = scaler.transform(input_df)

# ----------------------
# 5️⃣ Make predictions for all models
# ----------------------
if st.sidebar.button("Check Loan Eligibility"):
    models = {
        "Linear SVM": svm_linear,
        "Polynomial SVM": svm_poly,
        "RBF SVM": svm_rbf
    }

    results = {}
    for name, model in models.items():
        pred = model.predict(input_scaled)[0]
        conf = np.max(model.predict_proba(input_scaled))
        results[name] = {"Prediction": pred, "Confidence": conf}

    # ----------------------
    # 6️⃣ Display side-by-side results
    # ----------------------
    col1, col2, col3 = st.columns(3)

    for i, (name, res) in enumerate(results.items()):
        col = [col1, col2, col3][i]
        pred_text = "✅ Approved" if res["Prediction"]==1 else "❌ Rejected"
        color_func = col.success if res["Prediction"]==1 else col.error
        color_func(f"{name}\n{pred_text}\nConfidence: {res['Confidence']*100:.2f}%")
        
        # Business explanation
        expl = "Applicant likely to repay the loan." if res["Prediction"]==1 else "Applicant unlikely to repay the loan."
        col.info(f"💬 {expl}")

    # Optional: Show summary table
    st.subheader("Prediction Summary")
    summary_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Prediction": ["Approved" if r["Prediction"]==1 else "Rejected" for r in results.values()],
        "Confidence (%)": [round(r["Confidence"]*100,2) for r in results.values()]
    })
    st.table(summary_df)
