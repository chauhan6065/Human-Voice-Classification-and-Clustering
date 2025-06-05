import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the scaler, model, and expected column names
scaler = joblib.load('scaler.pkl')
model = joblib.load('voice_gender_classifier.pkl')
expected_columns = joblib.load('feature_columns.pkl')

st.title("Human Voice Gender Classification")

st.write("""
Upload a CSV file containing extracted voice features or input features manually for gender prediction.
""")

# Upload CSV
# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    # Drop label column if present
    if 'label' in data.columns:
        data = data.drop(columns=['label'])

    # Scale data
    data_scaled = scaler.transform(data)

    # Predict
    preds = model.predict(data_scaled)

    # Map prediction to label
    pred_labels = ['Female' if p == 0 else 'Male' for p in preds]

    st.write("Predictions:")
    st.write(pred_labels)


else:
    st.write("Or enter feature values manually:")

    # List of features you have to input (example: mean_pitch, rms_energy,...)
    features = ['mean_spectral_centroid', 'std_spectral_centroid', 'mean_spectral_bandwidth',
                'std_spectral_bandwidth', 'mean_spectral_contrast', 'mean_spectral_flatness',
                'mean_spectral_rolloff', 'zero_crossing_rate', 'rms_energy', 'mean_pitch',
                'min_pitch', 'max_pitch', 'std_pitch', 'spectral_skew', 'spectral_kurtosis',
                'energy_entropy', 'log_energy']

    user_input = []
    for feature in features:
        val = st.number_input(f"{feature}", format="%.6f")
        user_input.append(val)

    if st.button("Predict Gender"):
        input_array = np.array(user_input).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        gender = 'Female' if prediction == 0 else 'Male'
        st.success(f"Predicted Gender: {gender}")
