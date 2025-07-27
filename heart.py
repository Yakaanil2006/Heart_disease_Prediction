import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# STEP 1: LOAD DATA
df = pd.read_csv("heart.csv")

# Page Title
st.title("❤️ Heart Disease Prediction App")
st.markdown("Predict whether a person has heart disease based on medical attributes.")

# STEP 2: PREPROCESSING
X = df.drop("target", axis=1)
y = df["target"]

# STEP 3: MODEL TRAINING
model = RandomForestClassifier()
model.fit(X, y)

# STEP 4: USER INPUT + PREDICTION
st.sidebar.header("Enter Patient Information")

def user_input_features():
    age = st.sidebar.slider("Age", 29, 77, 55)
    
    sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    
    cp = st.sidebar.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }[x]
    )
    
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    
    fbs = st.sidebar.selectbox(
        "Fasting Blood Sugar > 120 mg/dl?",
        [0, 1],
        format_func=lambda x: "No (≤120 mg/dl)" if x == 0 else "Yes (>120 mg/dl)"
    )
    
    restecg = st.sidebar.selectbox(
        "Resting ECG Results",
        [0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }[x]
    )
    
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 70, 210, 150)
    
    exang = st.sidebar.selectbox(
        "Exercise Induced Angina?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    
    slope = st.sidebar.selectbox(
        "Slope of ST Segment",
        [0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping (best)",
            1: "Flat",
            2: "Downsloping (worst)"
        }[x]
    )
    
    ca = st.sidebar.selectbox(
        "Number of Major Vessels Colored (0–3)",
        [0, 1, 2, 3]
    )
    
    thal = st.sidebar.selectbox(
        "Thalassemia Type",
        [1, 2, 3],
        format_func=lambda x: {
            1: "Normal",
            2: "Fixed Defect",
            3: "Reversible Defect"
        }[x]
    )

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

# Display input
st.subheader("Entered Patient Data")
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

# Display result
st.subheader("Prediction Result")
if prediction == 1:
    st.error("⚠️ The model predicts the person **has heart disease**.")
else:
    st.success("✅ The model predicts the person **does not have heart disease**.")

# Show probabilities
st.subheader("Prediction Probability")
st.write(f"🟢 No Heart Disease: {prediction_proba[0] * 100:.2f}%")
st.write(f"🔴 Heart Disease: {prediction_proba[1] * 100:.2f}%")
