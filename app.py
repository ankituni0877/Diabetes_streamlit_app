import streamlit as st
import numpy as np
import pickle

# ---------- Load model and scaler ----------
model = pickle.load(open("log_reg_diabetes.pkl", "rb"))
scaler = pickle.load(open("scaler_diabetes.pkl", "rb"))

st.title("Diabetes Prediction App")

st.write("Enter patient details to estimate diabetes probability.")

# ---------- User inputs ----------
Pregnancies   = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
Glucose       = st.number_input("Glucose", min_value=0, max_value=300, value=120)
BloodPressure = st.number_input("BloodPressure", min_value=0, max_value=200, value=70)
SkinThickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
Insulin       = st.number_input("Insulin", min_value=0, max_value=900, value=80)
BMI           = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0, step=0.1)
DPF           = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
Age           = st.number_input("Age", min_value=1, max_value=120, value=33)

# ---------- Prediction ----------
if st.button("Predict"):
    x = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                   Insulin, BMI, DPF, Age]])
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0, 1]
    pred = model.predict(x_scaled)[0]

    st.write(f"Predicted probability of diabetes: {prob:.2f}")
    st.write("Prediction:", "Diabetic" if pred == 1 else "Non‑Diabetic")
