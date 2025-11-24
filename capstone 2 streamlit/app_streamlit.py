import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
MODEL_PATH = 'random_forest_model.joblib' # This model expects FULL feature names
PAGE_TITLE = "Heart Disease Prediction (RF)"

# --- Load Model ---
try:
    pipeline = joblib.load(MODEL_PATH)
    # --- MODIFICATION: Use FULL feature names ---
    expected_features = [
        'age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol',
        'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
        'exercise_induced_angina', 'st_depression', 'st_slope',
        'num_major_vessels', 'thalassemia'
    ]
    # ---------------------------------------------
    print("Model loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading the model: {e}")
    st.stop()

# --- Page Setup ---
st.set_page_config(page_title=PAGE_TITLE, layout="centered")
st.title(PAGE_TITLE + " ❤️")
st.write("Enter patient details to predict the likelihood of heart disease using a tuned Random Forest model.")
st.write("_Disclaimer: This prediction is for educational purposes only and not a substitute for professional medical advice._")

# --- Input Features ---
st.sidebar.header("Patient Data")

# Options for categorical-like features (assuming standard encodings)
sex_options = {0: "Female", 1: "Male"}
cp_options = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
fbs_options = {0: "False (< 120 mg/dl)", 1: "True (> 120 mg/dl)"}
restecg_options = {0: "Normal", 1: "ST-T wave abnormality", 2: "Probable/Definite LVH"}
exang_options = {0: "No", 1: "Yes"}
slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
thal_options = {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}
ca_options = [0, 1, 2, 3] # Num major vessels

# Use columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Vitals")
    age = st.slider("Age (age)", 29, 77, 54)
    sex_display = st.radio("Sex (sex)", list(sex_options.values()), index=1)
    sex = [k for k, v in sex_options.items() if v == sex_display][0]
    trestbps = st.slider("Resting Blood Pressure (resting_blood_pressure)", 94, 200, 130)
    chol = st.slider("Serum Cholesterol (cholesterol)", 126, 564, 246)
    fbs_display = st.radio("Fasting Blood Sugar > 120 mg/dl (fasting_blood_sugar)", list(fbs_options.values()), index=0)
    fbs = [k for k, v in fbs_options.items() if v == fbs_display][0]

with col2:
    st.subheader("Clinical Measurements")
    cp_display = st.selectbox("Chest Pain Type (chest_pain_type)", list(cp_options.values()), index=0)
    cp = [k for k, v in cp_options.items() if v == cp_display][0]
    restecg_display = st.selectbox("Resting ECG Results (resting_ecg)", list(restecg_options.values()), index=0)
    restecg = [k for k, v in restecg_options.items() if v == restecg_display][0]
    thalach = st.slider("Max Heart Rate Achieved (max_heart_rate)", 71, 202, 150)
    exang_display = st.radio("Exercise Induced Angina (exercise_induced_angina)", list(exang_options.values()), index=0)
    exang = [k for k, v in exang_options.items() if v == exang_display][0]
    oldpeak = st.slider("ST Depression (st_depression)", 0.0, 6.2, 1.0, step=0.1)
    slope_display = st.selectbox("ST Slope (st_slope)", list(slope_options.values()), index=1)
    slope = [k for k, v in slope_options.items() if v == slope_display][0]
    ca = st.selectbox("Num Major Vessels (num_major_vessels)", ca_options, index=0)
    thal_display = st.selectbox("Thalassemia (thalassemia)", list(thal_options.values()), index=2)
    thal = [k for k, v in thal_options.items() if v == thal_display][0]


# --- Prediction ---
if st.button("Predict Heart Disease Risk", type="primary"):
    
    # --- MODIFICATION: Create dictionary with FULL feature names ---
    input_data = {
        'age': [age],
        'sex': [sex],
        'chest_pain_type': [cp],
        'resting_blood_pressure': [trestbps],
        'cholesterol': [chol],
        'fasting_blood_sugar': [fbs],
        'resting_ecg': [restecg],
        'max_heart_rate': [thalach],
        'exercise_induced_angina': [exang],
        'st_depression': [oldpeak],
        'st_slope': [slope],
        'num_major_vessels': [ca],
        'thalassemia': [thal]
    }
    # -----------------------------------------------------------
    
    input_df = pd.DataFrame(input_data)

    # Ensure column order matches the 'expected_features' list
    try:
        input_df = input_df[expected_features]
    except KeyError as e:
        st.error(f"Column mismatch error: {e}. Ensure all input features are provided and names match the training data.")
        st.stop()
    except Exception as e:
        st.error(f"Error reordering columns: {e}")
        st.stop()

    try:
        # Make prediction (probability and class)
        prediction_proba = pipeline.predict_proba(input_df)[0] # Probabilities [P(0), P(1)]
        prediction = pipeline.predict(input_df)[0]        # Class prediction [0 or 1]
        prob_disease = prediction_proba[1] # Probability of class 1 (Disease)

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"Prediction: **Heart Disease Likely** (Class 1)")
            st.write("Based on the input data, the Random Forest model predicts a higher likelihood of heart disease.")
        else:
            st.success(f"Prediction: **Heart Disease Unlikely** (Class 0)")
            st.write("Based on the input data, the Random Forest model predicts a lower likelihood of heart disease.")

        st.progress(float(prob_disease))
        st.metric(label="Predicted Probability of Heart Disease", value=f"{prob_disease:.1%}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please check input values and ensure the model file is compatible.")

# --- Footer/Info ---
st.markdown("---")
st.caption(f"Model used: Tuned Random Forest Pipeline. Loaded from: {MODEL_PATH}")