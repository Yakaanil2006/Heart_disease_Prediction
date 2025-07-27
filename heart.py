import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("heart.csv")

# Train model (for demo, you can replace with your trained model)
X = df.drop("target", axis=1)
y = df["target"]
model = RandomForestClassifier()
model.fit(X, y)

# Title
st.title("❤️ Heart Disease Prediction App")
st.markdown("Predict if someone is likely to have heart disease based on health attributes.")

# User input via sidebar
st.sidebar.header("Input Features")

def user_input_features():
    age = st.sidebar.slider("Age", 29, 77, 55)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (chol)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 (fbs)", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG (restecg)", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate (thalach)", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of ST segment", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of major vessels (ca)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia (thal)", [1, 2, 3])

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

# Collect input
input_df = user_input_features()

# Show input
st.subheader("User Input:")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

# Result
st.subheader("Prediction Result:")
if prediction == 1:
    st.error("⚠️ The model predicts that the person **has heart disease**.")
else:
    st.success("✅ The model predicts that the person **does not have heart disease**.")

# Probability
st.subheader("Prediction Probability:")
st.write(f"Probability of No Heart Disease: {prediction_proba[0]:.2f}")
st.write(f"Probability of Heart Disease: {prediction_proba[1]:.2f}")
