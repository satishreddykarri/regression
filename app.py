import streamlit as st
import pandas as pd
import joblib
import os

st.title("Machine Learning Model Deployment")

# 📌 Step 1: Upload Model File
st.sidebar.header("Upload Model")
uploaded_model = st.sidebar.file_uploader("Upload a trained model (.pkl)", type=["pkl"])

if uploaded_model:
    # 📌 Step 2: Save Uploaded Model to 'models' Folder
    model_path = os.path.join("models", "uploaded_model.pkl")
    os.makedirs("models", exist_ok=True)  # Create 'models' folder if it doesn't exist

    with open(model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())

    # 📌 Step 3: Load the Model
    try:
        model = joblib.load(model_path)
        st.sidebar.success("Model uploaded and loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.stop()
else:
    st.sidebar.warning("Please upload a model first.")
    st.stop()

# 📌 Step 4: Extract Feature Names (if available)
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    feature_names = [f"Feature {i+1}" for i in range(model.n_features_in_)]

# 📌 Step 5: Accept User Inputs for Features
st.subheader("Enter Feature Values")
user_inputs = {}

for feature in feature_names:
    user_inputs[feature] = st.text_input(f"{feature}", "")

# 📌 Step 6: Make Predictions
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_inputs]).astype(float)  # Convert inputs to float
        prediction = model.predict(input_df)  # Make Prediction
        st.success(f"Prediction: {prediction[0]}")  # Display Prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
