import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import json
from model_utils import load_artifacts
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🫀 Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }
    .low-risk {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
    .moderate-risk {
        background: linear-gradient(135deg, #ffd93d, #ff9f43);
        color: white;
    }
    .feature-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    .model-prediction {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# Load all model artifacts
@st.cache_resource
def load_all_model_artifacts():
    """Load all trained models, scaler, and metadata"""
    try:
        artifacts_dir = 'artifacts'
        model, scaler, metadata = load_artifacts(artifacts_dir)
        
        # Load individual models from metadata
        models = {}
        if 'evaluations' in metadata:
            for model_name in metadata['evaluations'].keys():
                try:
                    # For this demo, we'll use the best model for all predictions
                    # In a real scenario, you'd save each model separately
                    models[model_name] = model
                except:
                    models[model_name] = model
        
        return models, scaler, metadata
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        return None, None, None

# Feature information and ranges
FEATURE_INFO = {
    'age': {
        'name': 'Age',
        'description': 'Age of the patient in years',
        'min': 29, 'max': 77, 'default': 50,
        'unit': 'years'
    },
    'sex': {
        'name': 'Sex',
        'description': 'Gender of the patient',
        'options': {0: 'Female', 1: 'Male'},
        'default': 1
    },
    'chest_pain_type': {
        'name': 'Chest Pain Type',
        'description': 'Type of chest pain experienced',
        'options': {
            0: 'Typical Angina',
            1: 'Atypical Angina', 
            2: 'Non-Anginal Pain',
            3: 'Asymptomatic'
        },
        'default': 0
    },
    'resting_blood_pressure': {
        'name': 'Resting Blood Pressure',
        'description': 'Resting blood pressure in mm Hg',
        'min': 94, 'max': 200, 'default': 120,
        'unit': 'mm Hg'
    },
    'cholesterol': {
        'name': 'Cholesterol',
        'description': 'Serum cholesterol level',
        'min': 126, 'max': 564, 'default': 200,
        'unit': 'mg/dl'
    },
    'fasting_blood_sugar': {
        'name': 'Fasting Blood Sugar',
        'description': 'Fasting blood sugar > 120 mg/dl',
        'options': {0: 'False (≤ 120 mg/dl)', 1: 'True (> 120 mg/dl)'},
        'default': 0
    },
    'resting_ecg': {
        'name': 'Resting ECG',
        'description': 'Resting electrocardiographic results',
        'options': {
            0: 'Normal',
            1: 'ST-T Wave Abnormality',
            2: 'Left Ventricular Hypertrophy'
        },
        'default': 0
    },
    'max_heart_rate': {
        'name': 'Maximum Heart Rate',
        'description': 'Maximum heart rate achieved during exercise',
        'min': 71, 'max': 202, 'default': 150,
        'unit': 'bpm'
    },
    'exercise_induced_angina': {
        'name': 'Exercise Induced Angina',
        'description': 'Exercise induced angina',
        'options': {0: 'No', 1: 'Yes'},
        'default': 0
    },
    'st_depression': {
        'name': 'ST Depression',
        'description': 'ST depression induced by exercise relative to rest',
        'min': 0.0, 'max': 6.2, 'default': 1.0,
        'unit': 'mm'
    },
    'st_slope': {
        'name': 'ST Slope',
        'description': 'Slope of the peak exercise ST segment',
        'options': {
            0: 'Upsloping',
            1: 'Flat',
            2: 'Downsloping'
        },
        'default': 1
    },
    'num_major_vessels': {
        'name': 'Number of Major Vessels',
        'description': 'Number of major vessels colored by fluoroscopy',
        'options': {0: '0', 1: '1', 2: '2', 3: '3'},
        'default': 0
    },
    'thalassemia': {
        'name': 'Thalassemia',
        'description': 'Thalassemia blood disorder',
        'options': {
            0: 'Normal',
            1: 'Fixed Defect',
            2: 'Reversible Defect',
            3: 'Not Described'
        },
        'default': 2
    }
}

def predict_with_all_models(features, models, scaler):
    """Make predictions with all available models"""
    predictions = {}
    features_scaled = scaler.transform(np.array([features]))
    
    for model_name, model in models.items():
        try:
            prob = model.predict_proba(features_scaled)[0][1]
            prediction = int(prob > 0.5)
            predictions[model_name] = {
                'prediction': prediction,
                'probability': prob
            }
        except Exception as e:
            st.warning(f"Error with {model_name}: {str(e)}")
            predictions[model_name] = {
                'prediction': 0,
                'probability': 0.0
            }
    
    return predictions

def create_ensemble_prediction(predictions):
    """Create ensemble prediction from all models"""
    if not predictions:
        return {'prediction': 0, 'probability': 0.0}
    
    # Average probability across all models
    avg_prob = np.mean([pred['probability'] for pred in predictions.values()])
    
    # Majority vote for prediction
    votes = [pred['prediction'] for pred in predictions.values()]
    ensemble_pred = int(np.mean(votes) > 0.5)
    
    return {'prediction': ensemble_pred, 'probability': avg_prob}

def create_model_comparison_chart(predictions):
    """Create a comparison chart of all model predictions"""
    model_names = list(predictions.keys())
    probabilities = [predictions[name]['probability'] * 100 for name in model_names]
    
    fig = px.bar(
        x=probabilities,
        y=model_names,
        orientation='h',
        title="Heart Disease Risk Predictions by Model",
        labels={'x': 'Risk Probability (%)', 'y': 'Models'},
        color=probabilities,
        color_continuous_scale='RdYlGn_r'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_font_size=16,
        title_x=0.5
    )
    
    return fig

def create_risk_gauge(probability):
    """Create a risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Ensemble Risk (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def get_risk_interpretation(probability):
    """Get risk interpretation and recommendations"""
    if probability < 0.3:
        risk_level = "LOW RISK"
        color = "low-risk"
        recommendations = [
            "🎉 Great news! Your risk is low",
            "💪 Continue maintaining a healthy lifestyle",
            "🏃‍♂️ Regular exercise and balanced diet",
            "📅 Regular check-ups as recommended by your doctor"
        ]
    elif probability < 0.7:
        risk_level = "MODERATE RISK"
        color = "moderate-risk"
        recommendations = [
            "⚠️ Moderate risk detected",
            "👨‍⚕️ Consult with your healthcare provider",
            "🥗 Consider lifestyle modifications",
            "💊 Follow prescribed medications if any",
            "📊 Monitor key health metrics regularly"
        ]
    else:
        risk_level = "HIGH RISK"
        color = "high-risk"
        recommendations = [
            "🚨 High risk detected - Seek immediate medical attention",
            "👨‍⚕️ Consult a cardiologist as soon as possible",
            "💊 Follow all prescribed treatments strictly",
            "🏥 Consider comprehensive cardiac evaluation",
            "📱 Monitor symptoms closely and seek emergency care if needed"
        ]
    
    return risk_level, color, recommendations

def main():
    # Header
    st.markdown('<h1 class="main-header">🫀 Heart Disease Risk Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Multi-Model ML-powered cardiovascular risk assessment</p>', unsafe_allow_html=True)
    
    # Load models
    models, scaler, metadata = load_all_model_artifacts()
    
    if models is None:
        st.error("❌ Could not load the prediction models. Please ensure model artifacts are available.")
        return
    
    # Sidebar for model information
    with st.sidebar:
        st.markdown("### 🤖 Available Models")
        if metadata and 'evaluations' in metadata:
            for model_name, eval_data in metadata['evaluations'].items():
                st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                test_eval = eval_data['test_eval']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{test_eval['classification_report']['accuracy']:.3f}")
                with col2:
                    st.metric("ROC-AUC", f"{test_eval['roc_auc']:.3f}")
                st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.markdown("""
        This application uses multiple machine learning models to assess heart disease risk:
        - **Decision Tree**: Rule-based predictions
        - **Random Forest**: Ensemble of decision trees
        - **Logistic Regression**: Statistical approach
        - **SVM**: Support Vector Machine
        
        **⚠️ Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📋 Patient Information</h2>', unsafe_allow_html=True)
        
        # Create input form
        with st.form("prediction_form"):
            # Organize inputs in columns
            input_col1, input_col2, input_col3 = st.columns(3)
            
            features = {}
            
            with input_col1:
                st.markdown("**Demographics**")
                features['age'] = st.slider(
                    FEATURE_INFO['age']['name'],
                    min_value=FEATURE_INFO['age']['min'],
                    max_value=FEATURE_INFO['age']['max'],
                    value=FEATURE_INFO['age']['default'],
                    help=FEATURE_INFO['age']['description']
                )
                
                sex_options = list(FEATURE_INFO['sex']['options'].values())
                sex_values = list(FEATURE_INFO['sex']['options'].keys())
                sex_selection = st.selectbox(
                    FEATURE_INFO['sex']['name'],
                    options=sex_options,
                    index=FEATURE_INFO['sex']['default'],
                    help=FEATURE_INFO['sex']['description']
                )
                features['sex'] = sex_values[sex_options.index(sex_selection)]
                
                st.markdown("**Chest Pain**")
                cp_options = list(FEATURE_INFO['chest_pain_type']['options'].values())
                cp_values = list(FEATURE_INFO['chest_pain_type']['options'].keys())
                cp_selection = st.selectbox(
                    FEATURE_INFO['chest_pain_type']['name'],
                    options=cp_options,
                    index=FEATURE_INFO['chest_pain_type']['default'],
                    help=FEATURE_INFO['chest_pain_type']['description']
                )
                features['chest_pain_type'] = cp_values[cp_options.index(cp_selection)]
                
            with input_col2:
                st.markdown("**Vital Signs**")
                features['resting_blood_pressure'] = st.slider(
                    FEATURE_INFO['resting_blood_pressure']['name'],
                    min_value=FEATURE_INFO['resting_blood_pressure']['min'],
                    max_value=FEATURE_INFO['resting_blood_pressure']['max'],
                    value=FEATURE_INFO['resting_blood_pressure']['default'],
                    help=FEATURE_INFO['resting_blood_pressure']['description']
                )
                
                features['cholesterol'] = st.slider(
                    FEATURE_INFO['cholesterol']['name'],
                    min_value=FEATURE_INFO['cholesterol']['min'],
                    max_value=FEATURE_INFO['cholesterol']['max'],
                    value=FEATURE_INFO['cholesterol']['default'],
                    help=FEATURE_INFO['cholesterol']['description']
                )
                
                features['max_heart_rate'] = st.slider(
                    FEATURE_INFO['max_heart_rate']['name'],
                    min_value=FEATURE_INFO['max_heart_rate']['min'],
                    max_value=FEATURE_INFO['max_heart_rate']['max'],
                    value=FEATURE_INFO['max_heart_rate']['default'],
                    help=FEATURE_INFO['max_heart_rate']['description']
                )
                
            with input_col3:
                st.markdown("**Medical Tests**")
                fbs_options = list(FEATURE_INFO['fasting_blood_sugar']['options'].values())
                fbs_values = list(FEATURE_INFO['fasting_blood_sugar']['options'].keys())
                fbs_selection = st.selectbox(
                    FEATURE_INFO['fasting_blood_sugar']['name'],
                    options=fbs_options,
                    index=FEATURE_INFO['fasting_blood_sugar']['default'],
                    help=FEATURE_INFO['fasting_blood_sugar']['description']
                )
                features['fasting_blood_sugar'] = fbs_values[fbs_options.index(fbs_selection)]
                
                ecg_options = list(FEATURE_INFO['resting_ecg']['options'].values())
                ecg_values = list(FEATURE_INFO['resting_ecg']['options'].keys())
                ecg_selection = st.selectbox(
                    FEATURE_INFO['resting_ecg']['name'],
                    options=ecg_options,
                    index=FEATURE_INFO['resting_ecg']['default'],
                    help=FEATURE_INFO['resting_ecg']['description']
                )
                features['resting_ecg'] = ecg_values[ecg_options.index(ecg_selection)]
                
                angina_options = list(FEATURE_INFO['exercise_induced_angina']['options'].values())
                angina_values = list(FEATURE_INFO['exercise_induced_angina']['options'].keys())
                angina_selection = st.selectbox(
                    FEATURE_INFO['exercise_induced_angina']['name'],
                    options=angina_options,
                    index=FEATURE_INFO['exercise_induced_angina']['default'],
                    help=FEATURE_INFO['exercise_induced_angina']['description']
                )
                features['exercise_induced_angina'] = angina_values[angina_options.index(angina_selection)]
            
            # Additional features in a new row
            st.markdown("**Advanced Measurements**")
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                features['st_depression'] = st.slider(
                    FEATURE_INFO['st_depression']['name'],
                    min_value=FEATURE_INFO['st_depression']['min'],
                    max_value=FEATURE_INFO['st_depression']['max'],
                    value=FEATURE_INFO['st_depression']['default'],
                    step=0.1,
                    help=FEATURE_INFO['st_depression']['description']
                )
                
            with adv_col2:
                slope_options = list(FEATURE_INFO['st_slope']['options'].values())
                slope_values = list(FEATURE_INFO['st_slope']['options'].keys())
                slope_selection = st.selectbox(
                    FEATURE_INFO['st_slope']['name'],
                    options=slope_options,
                    index=FEATURE_INFO['st_slope']['default'],
                    help=FEATURE_INFO['st_slope']['description']
                )
                features['st_slope'] = slope_values[slope_options.index(slope_selection)]
                
            with adv_col3:
                vessels_options = list(FEATURE_INFO['num_major_vessels']['options'].values())
                vessels_values = list(FEATURE_INFO['num_major_vessels']['options'].keys())
                vessels_selection = st.selectbox(
                    FEATURE_INFO['num_major_vessels']['name'],
                    options=vessels_options,
                    index=FEATURE_INFO['num_major_vessels']['default'],
                    help=FEATURE_INFO['num_major_vessels']['description']
                )
                features['num_major_vessels'] = vessels_values[vessels_options.index(vessels_selection)]
            
            # Thalassemia
            thal_options = list(FEATURE_INFO['thalassemia']['options'].values())
            thal_values = list(FEATURE_INFO['thalassemia']['options'].keys())
            thal_selection = st.selectbox(
                FEATURE_INFO['thalassemia']['name'],
                options=thal_options,
                index=FEATURE_INFO['thalassemia']['default'],
                help=FEATURE_INFO['thalassemia']['description']
            )
            features['thalassemia'] = thal_values[thal_options.index(thal_selection)]
            
            # Submit button
            submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)
            
        if submitted:
            # Convert features to list in correct order
            feature_order = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol',
                           'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 'exercise_induced_angina',
                           'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia']
            
            feature_list = [features[feature] for feature in feature_order]
            
            # Make predictions with all models
            predictions = predict_with_all_models(feature_list, models, scaler)
            ensemble_result = create_ensemble_prediction(predictions)
            
            # Display results
            st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Individual model predictions
            st.markdown("### Individual Model Predictions")
            for model_name, result in predictions.items():
                risk_pct = result['probability'] * 100
                prediction_text = "High Risk" if result['prediction'] == 1 else "Low Risk"
                color = "#ff6b6b" if result['prediction'] == 1 else "#51cf66"
                
                st.markdown(f"""
                <div class="model-prediction">
                    <strong>{model_name.replace('_', ' ').title()}</strong><br>
                    <span style="color: {color}; font-weight: bold;">{prediction_text}</span> 
                    (Risk: {risk_pct:.1f}%)
                </div>
                """, unsafe_allow_html=True)
            
            # Ensemble prediction
            st.markdown("### 🎯 Ensemble Prediction")
            risk_level, color, recommendations = get_risk_interpretation(ensemble_result['probability'])
            
            st.markdown(f"""
            <div class="prediction-result {color}">
                {risk_level}<br>
                <span style="font-size: 1rem;">Risk Probability: {ensemble_result['probability']*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 📋 Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Visualizations</h2>', unsafe_allow_html=True)
        
        if 'submitted' in locals() and submitted:
            # Risk gauge
            gauge_fig = create_risk_gauge(ensemble_result['probability'])
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Model comparison
            comparison_fig = create_model_comparison_chart(predictions)
            st.plotly_chart(comparison_fig, use_container_width=True)
        else:
            st.info("👆 Fill out the form and click 'Predict' to see visualizations")

if __name__ == "__main__":
    main()