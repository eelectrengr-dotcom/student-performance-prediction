import streamlit as st
import pickle
import numpy as np

# Load models
lr = pickle.load(open("logistic_model.pkl", "rb"))
rf = pickle.load(open("rf_model.pkl", "rb"))

# Title
st.title("🎓 Student Performance Prediction")

st.write("Enter student details below:")

# Inputs
hours = st.slider("📘 Hours Studied", 0, 12)
attendance = st.slider("📅 Attendance (%)", 0, 100)
sleep = st.slider("😴 Sleep Hours", 0, 10)
score = st.slider("📊 Previous Score", 0, 100)

# Model selection
model_choice = st.selectbox(
    "🤖 Select Model",
    ["Logistic Regression", "Random Forest"]
)

# Prediction button
if st.button("Predict"):
    data = np.array([[hours, attendance, sleep, score]])

    if model_choice == "Logistic Regression":
        prediction = lr.predict(data)
    else:
        prediction = rf.predict(data)

    if prediction[0] == 1:
        st.success("✅ Student will PASS")
    else:
        st.error("❌ Student will FAIL")