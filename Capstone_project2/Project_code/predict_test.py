#!/usr/bin/env python3
"""
Simple test script to validate heart disease prediction
"""
from model_utils import predict_heart_disease

# Sample features (replace with actual feature names/order from dataset)
# Assuming features are: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
sample_features = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]  # Example patient data

if __name__ == "__main__":
    try:
        result = predict_heart_disease(sample_features)
        print("Test prediction successful:", result)
    except Exception as e:
        print("Error during prediction:", str(e))