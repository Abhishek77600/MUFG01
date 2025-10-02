import requests

# Test the heart disease classification API
url = "http://localhost:8002/predict"

# Sample features (same as in predict_test.py)
data = {
    "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
}

try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print("API Test Successful!")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f}")
    else:
        print(f"API Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Request failed: {e}")