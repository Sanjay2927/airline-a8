import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("aircrash_model.pkl", "rb"))

st.set_page_config(page_title="Air Crash Fatality Prediction", layout="centered")
st.title("✈️ Air Crash Fatality Prediction App")
st.write("This app predicts whether an airplane crash will likely result in fatalities based on incident details.")

# User inputs
col1, col2 = st.columns(2)

with col1:
    aboard = st.number_input("Number of People Aboard", min_value=1, value=100)
    ground = st.number_input("Number of Ground Fatalities", min_value=0, value=0)
    year = st.number_input("Year of Crash", min_value=1900, max_value=2025, value=2000)
    month = st.slider("Month", 1, 12, 6)

with col2:
    time_hr = st.slider("Hour of Crash (24-hr format)", 0, 23, 12)
    engine_type = st.selectbox("Engine Type", ['Jet', 'Turboprop', 'Piston', 'Unknown'])
    purpose = st.selectbox("Purpose of Flight", ['Commercial', 'Training', 'Cargo', 'Private', 'Other'])

# Encode categorical variables
def encode_inputs(engine_type, purpose):
    data = {
        'Aboard': aboard,
        'Ground': ground,
        'Year': year,
        'Month': month,
        'Hour': time_hr,
        'Engine_Type_Jet': 0,
        'Engine_Type_Piston': 0,
        'Engine_Type_Turboprop': 0,
        'Engine_Type_Unknown': 0,
        'Purpose_Cargo': 0,
        'Purpose_Commercial': 0,
        'Purpose_Other': 0,
        'Purpose_Private': 0,
        'Purpose_Training': 0,
    }

    engine_col = f'Engine_Type_{engine_type}'
    purpose_col = f'Purpose_{purpose}'

    if engine_col in data:
        data[engine_col] = 1
    if purpose_col in data:
        data[purpose_col] = 1

    return pd.DataFrame([data])

# Prepare input DataFrame
input_df = encode_inputs(engine_type, purpose)

# Prediction
if st.button("Predict Fatality Risk"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ High Risk: This crash is likely to result in **fatalities**.")
    else:
        st.success("✅ Low Risk: This crash is **unlikely** to result in fatalities.")

# Optional: Show input data
if st.checkbox("Show input data"):
    st.write(input_df)
