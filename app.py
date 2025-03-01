import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier  # Example classifier
from sklearn.linear_model import LinearRegression  # Example regressor

st.title("Interactive Model Deployment Without PyCaret")

# Upload a model file
uploaded_model = st.file_uploader("Upload your scikit-learn model (.pkl)", type=["pkl"])

if uploaded_model:
    # Create model directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save uploaded model
    model_path = "models/uploaded_model.pkl"
    with open(model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())

    # Load the model with pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    st.success("Model uploaded and loaded successfully!")

    # Ask user to specify whether the model is for classification or regression
    model_type = st.radio("Select Model Type", ["Classification", "Regression"])

    # Dynamically ask for feature inputs
    st.subheader("Enter Feature Values")

    # Extract feature names using model's `feature_names_in_` attribute (for classifiers)
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)  # For scikit-learn models
    else:
        # If model doesn't have feature names, manually define or ask the user
        feature_names = st.text_area("Enter feature names (comma-separated)").split(",")

    user_inputs = {}

    for feature in feature_names:
        user_inputs[feature] = st.text_input(f"Enter {feature}", "")

    # Convert input to DataFrame and handle conversion to correct data types
    if st.button("Predict"):
        try:
            # Convert user inputs to numeric values
            input_data = {k: [float(v)] for k, v in user_inputs.items()}
            input_df = pd.DataFrame(input_data)

            # Make predictions using the uploaded model
            if model_type == "Classification":
                prediction = model.predict(input_df)
                st.success(f"Prediction (Class): {prediction[0]}")
            elif model_type == "Regression":
                prediction = model.predict(input_df)
                st.success(f"Prediction (Value): {prediction[0]}")
            else:
                st.warning("Invalid model type selected.")

        except Exception as e:
            st.error(f"Error in making prediction: {e}")
