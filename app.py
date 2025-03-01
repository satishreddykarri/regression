import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier  # Example classifier
from sklearn.linear_model import LinearRegression  # Example regressor

st.title("Interactive Model Deployment Without PyCaret")

# Upload a model file
uploaded_model = st.file_uploader("Upload your scikit-learn model (.pkl)", type=["pkl"])

if uploaded_model:
    # Save and load the uploaded model
    with open("temp_model.pkl", "wb") as f:
        f.write(uploaded_model.getbuffer())

    # Load the model with pickle
    with open("temp_model.pkl", "rb") as f:
        model = pickle.load(f)

    st.success("Model uploaded and loaded successfully!")

    # Dynamically ask for feature inputs
    st.subheader("Enter Feature Values")

    # Extract feature names using model's `feature_names_in_` attribute (for classifiers)
    # or you can manually define feature names if they are not available.
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)  # For models like RandomForest
    else:
        # If the model does not have `feature_names_in_`, you can manually define them
        # For example, for a random forest model trained on [feature1, feature2, feature3]
        feature_names = ['feature1', 'feature2', 'feature3']  # Replace with actual feature names

    user_inputs = {}

    for feature in feature_names:
        user_inputs[feature] = st.text_input(f"Enter {feature}", "")

    # Convert input to DataFrame and handle conversion to correct data types
    if st.button("Predict"):
        try:
            # Convert user inputs to appropriate data types
            input_data = {k: [v] for k, v in user_inputs.items()}
            input_df = pd.DataFrame(input_data)

            # Make predictions using the uploaded model
            if isinstance(model, RandomForestClassifier):
                prediction = model.predict(input_df)
                st.success(f"Prediction (Class): {prediction[0]}")
            elif isinstance(model, LinearRegression):
                prediction = model.predict(input_df)
                st.success(f"Prediction (Value): {prediction[0]}")
            else:
                st.warning("Model type not recognized. Make sure it is a classifier or regressor.")

        except Exception as e:
            st.error(f"Error in making prediction: {e}")
