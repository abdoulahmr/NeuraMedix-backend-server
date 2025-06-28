# heart_prediction_utils.py

import numpy as np
import joblib
import os

# Define the path to the models directory relative to this script
# Assuming 'models' directory is at the same level as app.py and this script
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, "heart-disease-model.pkl")

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope):
    """
    Loads a pre-trained heart disease prediction model and makes a prediction
    based on the provided input features.

    Args:
        age (int): Age in years.
        sex (int): Sex (0 = female, 1 = male).
        cp (int): Chest pain type (0-3).
        trestbps (int): Resting blood pressure (mm Hg).
        chol (int): Serum cholestoral in mg/dl.
        fbs (int): Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
        restecg (int): Resting electrocardiographic results (0-2).
        thalach (int): Maximum heart rate achieved.
        exang (int): Exercise induced angina (1 = yes; 0 = no).
        oldpeak (float): ST depression induced by exercise relative to rest.
        slope (int): The slope of the peak exercise ST segment (0-2).

    Returns:
        tuple: A tuple containing:
            - str: A message indicating if disease is detected or not.
            - float: Risk of heart disease in percentage (0-100).
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please ensure 'models/heart-disease-model.pkl' exists.")

    # Load model
    model = joblib.load(MODEL_PATH)

    # Prepare input data as a NumPy array
    # Ensure the order of features matches the model's training data
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])

    # Predict
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)
    risk = proba[0][1] * 100

    prediction_message = "✅ Disease Detected" if prediction[0] == 1 else "❌ No Heart Disease"

    return prediction_message, risk