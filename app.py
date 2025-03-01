import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading models
import sklearn  # Ensure scikit-learn is available

st.title("Interactive Model Deployment (Without PyCaret)")

# Upload a model file
uploaded_model = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])

if uploaded_model:
    # Save and load the uploaded model
    with open("model.pkl", "wb") as f:
        f.write(uploaded_model.getbuffer())

    model = joblib.load("model.pkl")  # Load model using joblib
    st.success("Model uploaded and loaded successfully!")

    # Check if model has feature names
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = [f"Feature {i}" for i in range(model.n_features_in_)]

    st.subheader("Enter Feature Values")
    user_inputs = {}

    for feature in feature_names:
        user_inputs[feature] = st.text_input(f"Enter {feature}", "")

    # Convert inputs to DataFrame
    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([user_inputs])

            # Convert input values to correct data types
            input_df = input_df.astype(float)

            # Predict using the model
            prediction = model.predict(input_df)

            # Display prediction
            st.success(f"Prediction: {prediction[0]}")

        except Exception as e:
            st.error(f"Error: {e}")
