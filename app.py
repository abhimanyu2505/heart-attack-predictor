import streamlit as st
import pandas as pd
import numpy as np
import pickle
from fpdf import FPDF
import matplotlib.pyplot as plt
import os

# ------------------------------
# Load trained model
# ------------------------------
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Features must match your training
features = ['Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Family History',
            'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week',
            'Sedentary Hours Per Day', 'bmi', 'Triglycerides']

# ------------------------------
# Streamlit App Setup
# ------------------------------
st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")
st.title("💓 Heart Attack Risk Predictor")
st.markdown("Enter your health data to check your heart attack risk.")

# ------------------------------
# User Input Fields
# ------------------------------
age = st.slider("Age", 18, 100, 30)
sex_str = st.selectbox("Sex", ["Male", "Female"])
cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
heart_rate = st.slider("Resting Heart Rate", 50, 200, 80)
diabetes_str = st.selectbox("Do you have diabetes?", ["No", "Yes"])
family_history_str = st.selectbox("Family history of heart disease?", ["No", "Yes"])
smoking_str = st.selectbox("Do you smoke?", ["No", "Yes"])
obesity_str = st.selectbox("Are you obese?", ["No", "Yes"])
alcohol_str = st.selectbox("Do you consume alcohol?", ["No", "Yes"])
exercise_hours = st.slider("Exercise Hours Per Week", 0, 20, 3)
sedentary_hours = st.slider("Sedentary Hours Per Day", 0, 24, 8)
bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
triglycerides = st.slider("Triglycerides (mg/dL)", 50, 1000, 150)

# ------------------------------
# Prepare User Input Dictionary (User-Friendly)
# ------------------------------
user_input = {
    "Age": age,
    "Sex": 1 if sex_str == "Male" else 0,
    "Cholesterol": cholesterol,
    "Heart Rate": heart_rate,
    "Diabetes": 1 if diabetes_str == "Yes" else 0,
    "Family History": 1 if family_history_str == "Yes" else 0,
    "Smoking": 1 if smoking_str == "Yes" else 0,
    "Obesity": 1 if obesity_str == "Yes" else 0,
    "Alcohol Consumption": 1 if alcohol_str == "Yes" else 0,
    "Exercise Hours Per Week": exercise_hours,
    "Sedentary Hours Per Day": sedentary_hours,
    "BMI (Body Mass Index)": bmi,
    "Triglycerides": triglycerides
}

# ------------------------------
# Build model input from the dictionary
# ------------------------------
model_input = pd.DataFrame([[
    user_input["Age"],
    user_input["Sex"],
    user_input["Cholesterol"],
    user_input["Heart Rate"],
    user_input["Diabetes"],
    user_input["Family History"],
    user_input["Smoking"],
    user_input["Obesity"],
    user_input["Alcohol Consumption"],
    user_input["Exercise Hours Per Week"],
    user_input["Sedentary Hours Per Day"],
    user_input["BMI (Body Mass Index)"],  # note: model was trained on column 'bmi'
    user_input["Triglycerides"]
]], columns=features)

# ------------------------------
# Prediction and Result Display
# ------------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(model_input)[0]
    if prediction == 1:
        st.error("⚠️ High risk of heart attack. Please consult a doctor.")
        risk_result = "High Risk"
    else:
        st.success("✅ Low risk of heart attack. Stay healthy!")
        risk_result = "Low Risk"
    
    st.markdown(f"### Risk Prediction: {risk_result}")

    # ------------------------------
    # Generate Chart Function
    # ------------------------------
    def generate_user_chart(user_input):
        # Use the same keys as in the user_input dictionary
        labels = list(user_input.keys())
        user_vals = list(user_input.values())

        plt.figure(figsize=(10, 6))
        bars = plt.barh(labels, user_vals, color='teal')
        plt.xlabel("Value")
        plt.title("Patient Health Parameters")
        plt.tight_layout()

        # Annotate bar values
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}', va='center')

        chart_path = "user_chart.png"
        plt.savefig(chart_path)
        plt.close()
        return chart_path

    # ------------------------------
    # PDF Report Generation Function
    # ------------------------------
    def generate_pdf_report(user_input, prediction, chart_path):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Heart Attack Risk Report", ln=True, align='C')
        pdf.ln(10)

        # Write user inputs in the PDF
        for key, value in user_input.items():
            pdf.cell(200, 8, txt=f"{key}: {value}", ln=True)

        pdf.ln(10)
        pdf.set_text_color(220, 50, 50)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=f"Risk Prediction: {prediction}", ln=True)
        pdf.ln(10)

        # Add chart image to PDF if it exists
        if os.path.exists(chart_path):
            pdf.image(chart_path, x=10, y=None, w=pdf.w - 20)

        output_path = "patient_report.pdf"
        pdf.output(output_path)
        return output_path

    # ------------------------------
    # Generate Report and Provide Download
    # ------------------------------
    chart_path = generate_user_chart(user_input)
    pdf_path = generate_pdf_report(user_input, risk_result, chart_path)

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download Patient Report (PDF)",
            data=f,
            file_name="heart_attack_report.pdf",
            mime="application/pdf"
        )
