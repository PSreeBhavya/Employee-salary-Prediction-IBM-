import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("salary_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Enter employee details below to predict their **Monthly Income**:")

# Input fields matching the trained model
age = st.slider("Age", 18, 60, 30)
education = st.selectbox("Education Level", [1, 2, 3, 4, 5])  # 1: Below College, 5: Doctor
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
distance_from_home = st.slider("Distance From Home (km)", 1, 50, 10)
total_working_years = st.slider("Total Working Years", 0, 40, 5)
job_role = st.selectbox("Job Role (Encoded)", list(range(0, 9)))  # 0 to 8 after LabelEncoding
department = st.selectbox("Department (Encoded)", list(range(0, 3)))  # 0 to 2 after LabelEncoding

# Combine into input array
input_data = np.array([[age, education, job_level, distance_from_home, total_working_years, job_role, department]])

# Predict salary
if st.button("Predict Salary"):
    predicted_salary = model.predict(input_data)
    st.success(f"💰 Predicted Monthly Income: **₹ {predicted_salary[0]:,.2f}**")

st.markdown("---")
st.caption("Built with 💡 using Streamlit · Trained on IBM HR Attrition Dataset")
