# Manufacturing Equipment Output Prediction

A complete data science project for predicting manufacturing equipment output using linear regression, featuring end-to-end deployment with FastAPI and Docker.

## Project Overview

This project demonstrates a comprehensive approach to building a predictive model for manufacturing equipment output. The system predicts the hourly production rate (parts per hour) based on various machine operating parameters.

## Dataset

The dataset contains 1,000+ records of hourly machine performance data with the following features:
- **Injection_Temperature**: Molten plastic temperature (°C)
- **Injection_Pressure**: Hydraulic pressure (bar)
- **Cycle_Time**: Time per part cycle (seconds)
- **Cooling_Time**: Part cooling duration (seconds)
- **Material_Viscosity**: Plastic material flow resistance (Pa·s)
- **Ambient_Temperature**: Factory floor temperature (°C)
- **Machine_Age**: Equipment age in years
- **Operator_Experience**: Operator experience level (months)
- **Maintenance_Hours**: Hours since last maintenance
- **Parts_Per_Hour**: Target variable (production rate)

## Project Structure

```
├── Manufacturing_Output_Prediction.ipynb  # Main analysis notebook
├── fastapi_app.py                         # FastAPI backend
├── app.py                                 # Streamlit frontend (existing)
├── train.py                               # Training script (existing)
├── requirements.txt                       # Python dependencies
├── Dockerfile                             # Docker configuration
├── docker-compose.yml                     # Multi-service deployment
├── .dockerignore                          # Docker ignore file
├── manufacturing_model.pkl                # Trained model
├── manufacturing_dataset_1000_samples.csv # Dataset
└── README.md                              # This file
```

## Quick Start

### 1. Run the Jupyter Notebook
```bash
jupyter notebook Manufacturing_Output_Prediction.ipynb
```

### 2. Run FastAPI Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn fastapi_app:app --reload
```

### 3. Run with Docker
```bash
# Build and run
docker-compose up --build

# Or run with development services
docker-compose --profile dev up --build
```

## API Endpoints

### FastAPI Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /model-info` - Model metadata
- `POST /predict` - Single prediction
- `POST /batch-predict` - Batch predictions

### Example API Usage

```python
import requests

# Single prediction
data = {
    "Injection_Temperature": 220.0,
    "Injection_Pressure": 120.0,
    "Cycle_Time": 25.0,
    "Cooling_Time": 12.0,
    "Material_Viscosity": 300.0,
    "Ambient_Temperature": 25.0,
    "Machine_Age": 5.0,
    "Operator_Experience": 60.0,
    "Maintenance_Hours": 50.0
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Model Performance

- **Algorithm**: Linear Regression
- **R² Score**: ~0.75 (varies with data)
- **Key Insights**:
  - Cycle time has the strongest negative correlation with output
  - Temperature and pressure optimization can improve efficiency
  - Regular maintenance prevents performance degradation

## Business Impact

### Optimization Recommendations:
1. **Minimize Cycle Time**: Reduce cycle time to increase hourly output
2. **Optimize Temperature**: Maintain injection temperature in optimal range
3. **Pressure Management**: Balance injection pressure for efficiency
4. **Maintenance Schedule**: Implement preventive maintenance
5. **Operator Training**: Invest in experienced operators

### Potential Cost Savings:
- 10-20% improvement in production efficiency
- Reduced downtime through predictive maintenance
- Better resource utilization

## Development

### Prerequisites
- Python 3.8+
- Docker (optional)
- Jupyter Notebook

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook

# Run API
uvicorn fastapi_app:app --reload
```

### Docker Development
```bash
# Run API only
docker-compose up manufacturing-api

# Run with Jupyter for development
docker-compose --profile dev up
```

## Deployment

### Production Deployment
```bash
# Build production image
docker build -t manufacturing-api .

# Run container
docker run -p 8000:8000 manufacturing-api
```

### Using Docker Compose
```bash
# Production deployment
docker-compose up -d

# With monitoring
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

## API Documentation

Once the FastAPI server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset generated using realistic manufacturing principles
- Built with scikit-learn, FastAPI, and Docker
- Inspired by industrial IoT and manufacturing analytics use cases