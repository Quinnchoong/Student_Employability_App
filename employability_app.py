# Import Necessary Libraries
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os

# Page Config
st.set_page_config(page_title="🎓 Student Employability Predictor", layout="centered")

# Styling
st.markdown("""
<style>
.stApp {
    background-color: #e6f2ff;
}
html, body, [class*="css"] {
    font-size: 14px;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_resources():
    model_path = "employability_predictor.pkl"
    scaler_path = "scaler.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    return joblib.load(model_path), joblib.load(scaler_path)

model, scaler = load_resources()

if model is None or scaler is None:
    st.error("⚠️ Model or scaler file not found. Please ensure 'employability_predictor.pkl' and 'scaler.pkl' exist.")
    st.stop()

# Header image
try:
    image = Image.open("business_people.png")
    st.image(image, use_container_width=True)
except FileNotFoundError:
    st.warning("Header image not found. Skipping image display.")

# App title
st.markdown("<h2 style='text-align: center;'>🎓 Student Employability Predictor — SVM Model</h2>", unsafe_allow_html=True)
st.markdown("Fill in the input features to predict employability.")

# Feature Inputs
feature_columns = [
    'GENDER', 'GENERAL_APPEARANCE', 'GENERAL_POINT_AVERAGE',
    'MANNER_OF_SPEAKING', 'PHYSICAL_CONDITION', 'MENTAL_ALERTNESS',
    'SELF-CONFIDENCE', 'ABILITY_TO_PRESENT_IDEAS', 'COMMUNICATION_SKILLS',
    'STUDENT_PERFORMANCE_RATING', 'NO_SKILLS', 'Year_of_Graduate'
]

def get_user_inputs():
    col1, col2, col3 = st.columns(3)
    inputs = {}

    with col1:
        inputs['GENDER'] = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=1)
        inputs['GENERAL_APPEARANCE'] = st.slider("Appearance (2–5)", 2, 5, 3, help="Overall presentation and grooming.")
        inputs['GENERAL_POINT_AVERAGE'] = st.slider("GPA (2-5)", 2, 5, 3, help="General point average (GPA).")
        inputs['MANNER_OF_SPEAKING'] = st.slider("Speaking Skills (2–5)", 2, 5, 3, help="Clarity and fluency of speech.")

    with col2:
        inputs['PHYSICAL_CONDITION'] = st.slider("Physical Condition (2–5)", 2, 5, 3, help="Health and physical fitness.")
        inputs['MENTAL_ALERTNESS'] = st.slider("Mental Alertness (2–5)", 2, 5, 3, help="Level of attentiveness and response.")
        inputs['SELF-CONFIDENCE'] = st.slider("Confidence (2–5)", 2, 5, 3, help="Self-confidence in various situations.")
        inputs['ABILITY_TO_PRESENT_IDEAS'] = st.slider("Idea Presentation (2–5)", 2, 5, 3, help="Ability to present and explain ideas.")

    with col3:
        inputs['COMMUNICATION_SKILLS'] = st.slider("Communication Skills (2-5)", 2, 5, 3, help="Effectiveness of verbal and written communication.")
        inputs['STUDENT_PERFORMANCE_RATING'] = st.slider("Performance Rating (3-4)", 3, 4, 5, help="Overall academic and behavioral performance.")
        inputs['NO_SKILLS'] = st.selectbox("Number of Skills", [2, 3, 4, 5], index=0, help="Number of essential skills the student lacks.")
        inputs['Year_of_Graduate'] = st.number_input("Graduation Year", 2019, 2022, 2022, help="Year in which the student graduated.")

    return pd.DataFrame([inputs])[feature_columns]

# Predict Function
def predict_employability(model, scaler, input_df):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0]
    return prediction, proba

# Get Inputs
input_df = get_user_inputs()

# Prediction
if st.button("Predict"):
    prediction, probability = predict_employability(model, scaler, input_df)

    if prediction == 1:
        st.success("🎉 The student is predicted to be **Employable**!")
        st.balloons()
    else:
        st.warning("⚠️ The student is predicted to be **Less Employable**.")

    st.info(f"📈 Employable Probability: **{probability[1] * 100:.2f}%**")
    st.info(f"📉 Less Employable Probability: **{probability[0] * 100:.2f}%**")

# Footer
st.markdown("---")
st.caption("""
Disclaimer: This prediction model is for research and informational purposes only.  
Version 1.0, © 2025 CHOONG MUH IN (TP068331) — Last updated: August 2025.  
""")

