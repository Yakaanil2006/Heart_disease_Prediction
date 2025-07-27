import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# STEP 1: LOAD DATA
# -----------------
df = pd.read_csv("heart.csv")  # Load the dataset
st.title("❤️ Heart Disease Prediction App")
st.markdown("Predict if a person has heart disease based on medical features.")

# STEP 2: PREPROCESSING (here, we directly use numeric input)
# -----------------------------------------------------------
# No major preprocessing required since data is clean and numeric
X = df.drop("target", axis=1)
y = df["target"]

# STEP 3: MODEL TRAINING
# ----------------------
model = RandomForestClassifier()
model.fit(X, y)  # Fit the model (In real-world, load a pre-trained model)

# STEP 4: USER INPUT + PREDICTION
# -------------------------------

# Sidebar inputs
st.sidebar.header("Enter Patient Details")

def user_input_features():
    age = st.sidebar.slider("Age", 29, 77, 55)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of ST Segment", [0, 1, 2])
    ca = st.sidebar.selectbox("Major Vessels Colored", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia Type", [1, 2, 3])

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    return pd.DataFrame([data])

# Get input
input_df = user_input_features()

# Show input
st.subheader("Entered Patient Data")
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

# Output
st.subheader("Prediction Result")
if prediction == 1:
    st.error("⚠️ The patient is likely to have heart disease.")
else:
    st.success("✅ The patient is unlikely to have heart disease.")

st.subheader("Prediction Probability")
st.write(f"🟢 No Heart Disease: {prediction_proba[0]*100:.2f}%")
st.write(f"🔴 Heart Disease: {prediction_proba[1]*100:.2f}%")
