import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
# Ensure this matches the saved LassoCV model trained WITHOUT polynomials
MODEL_PATH = 'manufacturing_output_model.pkl'
PAGE_TITLE = "Manufacturing Output Prediction"

# --- Load Model ---
# This loads the entire pipeline (preprocessor + model)
try:
    pipeline = joblib.load(MODEL_PATH)
    # Extract feature names required by the preprocessor
    # This relies on the structure saved within the pipeline, adjust names if needed
    num_features = pipeline.named_steps['preprocessor'].transformers_[0][2]
    cat_features = pipeline.named_steps['preprocessor'].transformers_[1][2]
    training_cols_order = num_features + cat_features
    print("Model and feature names loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the same directory.")
    st.stop() # Stop execution if model doesn't load
except AttributeError:
     st.error("Error: Could not extract feature names automatically from the loaded pipeline.")
     st.warning("Ensure the pipeline structure ('preprocessor', 'transformers_') is correct.")
     st.stop()
except Exception as e:
    st.error(f"An error occurred loading the model or extracting features: {e}")
    st.stop()

# --- Page Setup ---
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE + " 🏭")
st.write("Enter the machine parameters below to predict the hourly parts output using a tuned LassoCV model.")

# --- Input Features ---
st.sidebar.header("Input Features")

# Define categorical options (based on typical data/encoding)
# Include 'missing' as it was handled by the imputer
shift_options = ['Day', 'Evening', 'Night', 'missing']
machine_type_options = ['Type_A', 'Type_B', 'Type_C', 'missing']
material_grade_options = ['Economy', 'Standard', 'Premium', 'missing']
day_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'missing']

# Use two columns for better layout
col1, col2 = st.columns(2)

# Input fields with default/typical values
with col1:
    st.subheader("Operational Parameters")
    inj_temp = st.slider("Injection Temperature (°C)", min_value=180.0, max_value=300.0, value=215.3, step=0.1)
    inj_press = st.slider("Injection Pressure (bar)", min_value=80.0, max_value=150.0, value=116.1, step=0.1)
    cycle_time = st.slider("Cycle Time (s)", min_value=16.0, max_value=60.0, value=35.9, step=0.1)
    cool_time = st.slider("Cooling Time (s)", min_value=8.0, max_value=20.0, value=11.9, step=0.1)
    # For imputed features, provide guidance or use median from training data if known (approx. 242.7)
    mat_visc = st.number_input("Material Viscosity (Pa·s)", min_value=100.0, max_value=1000.0, value=242.7, step=0.1, help="Median value (242.7) used for missing data during training.")
    # Approx median for amb_temp: 22.9
    amb_temp = st.number_input("Ambient Temperature (°C)", min_value=18.0, max_value=28.0, value=22.9, step=0.1, help="Median value (22.9) used for missing data during training.")
    maint_hours = st.number_input("Maintenance Hours (last month)", min_value=0, max_value=500, value=50)

with col2:
    st.subheader("Machine & Context")
    mach_age = st.slider("Machine Age (years)", min_value=1.0, max_value=15.0, value=7.9, step=0.1)
    # Approx median for op_exp: 22.1
    op_exp = st.number_input("Operator Experience (years)", min_value=1.0, max_value=120.0, value=22.1, step=0.1, help="Median value (22.1) used for missing data during training.")
    shift = st.selectbox("Shift", shift_options, index=0) # Default to 'Day'
    machine_type = st.selectbox("Machine Type", machine_type_options, index=0) # Default to Type_A
    material_grade = st.selectbox("Material Grade", material_grade_options, index=1) # Default to Standard
    day_of_week = st.selectbox("Day of Week", day_options, index=0) # Default to Monday

st.subheader("Calculated/Efficiency Metrics")
col3, col4, col5, col6 = st.columns(4)
with col3:
    temp_press_ratio = st.number_input("Temp/Pressure Ratio", min_value=1.0, max_value=3.0, value=1.89, format="%.3f", step=0.01)
with col4:
    total_cycle_time = st.number_input("Total Cycle Time (s)", min_value=20.0, max_value=70.0, value=47.7, step=0.1)
with col5:
    efficiency_score = st.slider("Efficiency Score", min_value=0.0, max_value=1.0, value=0.19, step=0.01)
with col6:
    machine_util = st.slider("Machine Utilization (%)", min_value=0.0, max_value=1.0, value=0.36, step=0.01)


# --- Prediction ---
if st.button("Predict Hourly Output", type="primary"):
    # Create input DataFrame
    input_data = {
        'Injection_Temperature': [inj_temp], 'Injection_Pressure': [inj_press],
        'Cycle_Time': [cycle_time], 'Cooling_Time': [cool_time],
        'Material_Viscosity': [mat_visc], 'Ambient_Temperature': [amb_temp],
        'Machine_Age': [mach_age], 'Operator_Experience': [op_exp],
        'Maintenance_Hours': [maint_hours], 'Temperature_Pressure_Ratio': [temp_press_ratio],
        'Total_Cycle_Time': [total_cycle_time], 'Efficiency_Score': [efficiency_score],
        'Machine_Utilization': [machine_util],
        'Shift': [shift], 'Machine_Type': [machine_type],
        'Material_Grade': [material_grade], 'Day_of_Week': [day_of_week]
    }
    input_df = pd.DataFrame(input_data)

    # Reorder columns to match the order used during training
    try:
        input_df = input_df[training_cols_order]
    except KeyError as e:
        st.error(f"Column mismatch error: {e}. Ensure all input features are provided and names match.")
        st.stop()
    except Exception as e:
        st.error(f"Error reordering columns: {e}")
        st.stop()


    # Make prediction using the loaded pipeline
    try:
        prediction = pipeline.predict(input_df)
        predicted_value = prediction[0]

        st.subheader("Prediction Result")
        st.metric(label="Predicted Parts Per Hour", value=f"{predicted_value:.2f}")

        # Provide context based on typical ranges from EDA (Mean ~29.3)
        avg_output = 29.30
        if predicted_value > avg_output * 1.2: # Wider margin for success
             st.success("Output prediction is significantly above average.")
        elif predicted_value < avg_output * 0.8: # Wider margin for warning
             st.warning("Output prediction is significantly below average.")
        else:
             st.info("Output prediction is around the average range.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please check input values and model compatibility.")

# --- Footer/Info ---
st.markdown("---")
st.caption(f"Model: Tuned LassoCV. Loaded from: {MODEL_PATH}")