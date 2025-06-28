import joblib
import numpy as np
import os

# Load model and scaler once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'models/ensemble_model_soft_voting.joblib'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models/scaler.joblib'))

feature_order = [
    'AGE', 'GENDER', 'SMOKING', 'FINGER_DISCOLORATION', 'MENTAL_STRESS',
    'EXPOSURE_TO_POLLUTION', 'LONG_TERM_ILLNESS', 'ENERGY_LEVEL', 'IMMUNE_WEAKNESS',
    'BREATHING_ISSUE', 'ALCOHOL_CONSUMPTION', 'THROAT_DISCOMFORT', 'OXYGEN_SATURATION',
    'CHEST_TIGHTNESS', 'FAMILY_HISTORY', 'SMOKING_FAMILY_HISTORY', 'STRESS_IMMUNE'
]

def predict_pulmonary_disease(data_dict, threshold=0.4):
    # Ensure all features are present
    if not all(f in data_dict for f in feature_order):
        raise ValueError("Missing one or more required input features.")
    
    # Format input
    input_values = np.array([float(data_dict[f]) for f in feature_order]).reshape(1, -1)
    input_scaled = scaler.transform(input_values)
    
    # Predict probability
    probability = model.predict_proba(input_scaled)[0][1]
    label = "At Risk" if probability >= threshold else "Low"
    
    return {
        "prediction": label,
        "probability": round(probability * 100, 2)
    }
