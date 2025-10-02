from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any

# Load the model
try:
    with open('manufacturing_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        r2_score = model_data['r2_score']
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Please run the notebook first to train and save the model.")
    model = None
    scaler = None
    features = None

# Create FastAPI app
app = FastAPI(
    title="Manufacturing Output Prediction API",
    description="API for predicting manufacturing equipment output using linear regression",
    version="1.0.0"
)

# Define input data model
class ManufacturingInput(BaseModel):
    Injection_Temperature: float
    Injection_Pressure: float
    Cycle_Time: float
    Cooling_Time: float
    Material_Viscosity: float
    Ambient_Temperature: float
    Machine_Age: float
    Operator_Experience: float
    Maintenance_Hours: float

# Define response model
class PredictionResponse(BaseModel):
    predicted_output: float
    model_info: Dict[str, Any]
    input_features: Dict[str, float]

@app.get("/")
async def root():
    return {
        "message": "Manufacturing Output Prediction API",
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_r2_score": r2_score if r2_score else None
    }

@app.get("/model-info")
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": "LinearRegression",
        "features": features,
        "r2_score": r2_score,
        "coefficients": model.coef_.tolist() if hasattr(model, 'coef_') else None,
        "intercept": model.intercept_ if hasattr(model, 'intercept_') else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_output(input_data: ManufacturingInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])

        # Ensure correct feature order
        input_df = input_df[features]

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        # Ensure prediction is not negative
        prediction = max(0, prediction)

        return PredictionResponse(
            predicted_output=round(prediction, 2),
            model_info={
                "type": "LinearRegression",
                "r2_score": round(r2_score, 4)
            },
            input_features=input_dict
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(inputs: list[ManufacturingInput]):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert inputs to DataFrame
        input_dicts = [inp.dict() for inp in inputs]
        input_df = pd.DataFrame(input_dicts)

        # Ensure correct feature order
        input_df = input_df[features]

        # Scale the inputs
        input_scaled = scaler.transform(input_df)

        # Make predictions
        predictions = model.predict(input_scaled)

        # Ensure predictions are not negative
        predictions = np.maximum(0, predictions)

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "prediction_id": i + 1,
                "predicted_output": round(pred, 2),
                "input_features": input_dicts[i]
            })

        return {
            "predictions": results,
            "model_info": {
                "type": "LinearRegression",
                "r2_score": round(r2_score, 4)
            },
            "total_predictions": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)