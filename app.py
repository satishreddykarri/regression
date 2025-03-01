import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Interactive Model Deployment")

MODEL_PATH = "models/best_regression_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    feature_names = [f"Feature {i}" for i in range(model.n_features_in_)]

st.subheader("Enter Feature Values")
user_inputs = {}

for feature in feature_names:
    user_inputs[feature] = st.text_input(f"Enter {feature}", "")

if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_inputs]).astype(float)
        prediction = model.predict(input_df)
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
