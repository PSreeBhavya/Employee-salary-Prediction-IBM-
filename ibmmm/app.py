import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("salary_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Enter employee details below to predict monthly income:")

# Input fields
work_year = st.slider("Work Year", 2015, 2025, 2022)
experience_level = st.selectbox("Experience Level", [0, 1, 2, 3])  # Map: Entry, Mid, Senior, Exec
employment_type = st.selectbox("Employment Type", [0, 1, 2, 3])  # Map: PT, FT, etc.
job_title = st.selectbox("Job Title (Encoded)", list(range(0, 30)))
remote_ratio = st.slider("Remote Ratio (%)", 0, 100, 50)
company_location = st.selectbox("Company Location (Encoded)", list(range(0, 10)))
company_size = st.selectbox("Company Size (Encoded)", [0, 1, 2])  # Map: S, M, L

# Convert inputs into numpy array
input_data = np.array([[work_year, experience_level, employment_type, job_title,
                        remote_ratio, company_location, company_size]])

# Predict
if st.button("Predict Salary"):
    predicted_salary = model.predict(input_data)
    st.success(f"💰 Predicted Monthly Salary: **${predicted_salary[0]:,.2f}**")

st.markdown("---")
st.caption("Built using Streamlit · Model trained on IBM HR Analytics Dataset")
