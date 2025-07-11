import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and encoders
with open("salary_model.pkl", "rb") as f:
    model, label_encoders = pickle.load(f)

# Page setup
st.set_page_config(page_title="Salary Predictor", layout="centered")

# Title
st.title(" Salary Prediction using Ensemble Learning")
st.markdown("### Fill in the details below to get your predicted salary:")

# Input form
with st.form("salary_form"):
    st.subheader(" Personal Information")
    education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
    age = st.slider("Age", 18, 65, 30)

    st.subheader("Job Details")
    job_title = st.selectbox("Job Title", ["Data Scientist", "Software Engineer", "Analyst", "Manager"])
    industry = st.selectbox("Industry", ["IT", "Finance", "Education", "Healthcare"])
    location = st.selectbox("Location", ["New York", "London", "Bangalore"])
    company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])

    st.subheader(" Experience & Skills")
    experience = st.slider("Years of Experience", 0.0, 30.0, 5.0)
    certifications = st.slider("Number of Certifications", 0, 10, 1)
    working_hours = st.slider("Weekly Working Hours", 20.0, 80.0, 40.0)

    crucial_code = st.selectbox("Select Crucial Code", label_encoders['crucial_code'].classes_)

    submit_btn = st.form_submit_button(" Predict Salary")

# On form submit
if submit_btn:
    # Prepare input
    input_dict = {
        'education_level': education,
        'years_experience': experience,
        'job_title': job_title,
        'industry': industry,
        'location': location,
        'company_size': company_size,
        'certifications': certifications,
        'age': age,
        'working_hours': working_hours,
        'crucial_code': crucial_code
    }

    # Encode categorical features
    for col in input_dict:
        if col in label_encoders:
            if input_dict[col] not in label_encoders[col].classes_:
                st.error(f"Invalid input for '{col}': {input_dict[col]}")
                st.stop()
            input_dict[col] = label_encoders[col].transform([input_dict[col]])[0]

    # Create DataFrame
    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_df)[0]

    # Display result with large green text
    st.markdown(f"""
    <div style="text-align: center; font-size: 32px; font-weight: bold; color: green;">
     Predicted Salary: ₹{int(prediction):,}
    </div>
    """, unsafe_allow_html=True)
