import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Manufacturing Output Predictor",
    page_icon="🏭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    /* Global Styles */
    body {
        background: linear-gradient(135deg, #fef7ff 0%, #f3e8ff 50%, #e0f2fe 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Title */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 25%, #2563eb 50%, #0d9488 75%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #7c3aed 50%, #334155 100%);
        border-right: 2px solid #a855f7;
        box-shadow: 2px 0 10px rgba(124, 58, 237, 0.3);
    }
    .css-1d391kg .css-1lcbmhc {
        color: #f1f5f9;
        font-weight: 600;
        border-radius: 12px;
        margin: 4px;
        transition: all 0.3s ease;
    }
    .css-1d391kg .css-1lcbmhc:hover {
        background: linear-gradient(135deg, #a855f7 0%, #7c3aed 100%);
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
    }

    /* Metric Boxes */
    .metric-box {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 50%, #0d9488 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    .metric-box:hover::before {
        left: 100%;
    }
    .metric-box:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 35px rgba(37, 99, 235, 0.4);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }

    /* Prediction Result */
    .prediction-result {
        background: linear-gradient(135deg, #059669 0%, #0d9488 30%, #7c3aed 70%, #a855f7 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(16, 185, 129, 0.4);
        position: relative;
        overflow: hidden;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 15px 35px rgba(16, 185, 129, 0.4); }
        50% { box-shadow: 0 15px 35px rgba(168, 85, 247, 0.6); }
        100% { box-shadow: 0 15px 35px rgba(16, 185, 129, 0.4); }
    }
    .prediction-result::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="3" fill="rgba(255,255,255,0.1)"/><circle cx="20" cy="80" r="2" fill="rgba(255,255,255,0.05)"/><circle cx="80" cy="20" r="2" fill="rgba(255,255,255,0.05)"/></svg>');
        opacity: 0.1;
    }
    .prediction-value {
        font-size: 4rem;
        font-weight: 800;
        margin: 1rem 0;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    /* Section Titles */
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #7c3aed;
        display: inline-block;
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Cards */
    .card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.1);
        border: 1px solid #e9d5ff;
        position: relative;
    }
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #7c3aed 0%, #a855f7 50%, #2563eb 100%);
        border-radius: 16px 16px 0 0;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
        position: relative;
        overflow: hidden;
    }
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    .stButton>button:hover::before {
        left: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4);
    }

    /* Sliders */
    .stSlider .st-bs {
        background: linear-gradient(90deg, #7c3aed 0%, #a855f7 50%, #2563eb 100%);
        height: 6px;
        border-radius: 3px;
    }
    .stSlider .st-bq {
        background: linear-gradient(135deg, #f3e8ff 0%, #e0f2fe 100%);
    }

    /* Tables */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.1);
        border: 1px solid #e9d5ff;
    }
    .dataframe th {
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
        font-weight: 600;
        color: #374151;
    }
    .dataframe tbody tr:nth-child(even) {
        background: #fef7ff;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #f8fafc 0%, #f3e8ff 100%);
        padding: 0.5rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        border: 2px solid #e9d5ff;
        color: #64748b;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #a855f7;
        color: #7c3aed;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        color: white;
        border-color: #7c3aed;
    }

    /* Success/Warning/Info messages */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    .stSuccess {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 4px solid #16a34a;
    }
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #d97706;
    }
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #2563eb;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0;
        border-top: 2px solid #e9d5ff;
        margin-top: 3rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px 12px 0 0;
    }
    .footer h3 {
        color: #7c3aed;
        margin-bottom: 0.5rem;
    }

    /* Custom colored sections */
    .insights-card {
        background: linear-gradient(135deg, #fef7ff 0%, #f3e8ff 100%);
        border-left: 4px solid #a855f7;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid #f59e0b;
    }
    .impact-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid #16a34a;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('manufacturing_dataset_1000_samples.csv')

@st.cache_resource
def load_model():
    try:
        with open('manufacturing_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Load resources
df = load_data()
model_data = load_model()

# Title
st.markdown('<div class="main-title">🏭 Manufacturing Output Predictor</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Data Analysis", "Model Performance", "Make Prediction", "Insights"])

if page == "Overview":
    st.markdown('<div class="section-title">📊 Project Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">{:.1f}</div>
            <div class="metric-label">Avg Output (parts/hr)</div>
        </div>
        """.format(df['Parts_Per_Hour'].mean()), unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">{:.1f}</div>
            <div class="metric-label">Max Output (parts/hr)</div>
        </div>
        """.format(df['Parts_Per_Hour'].max()), unsafe_allow_html=True)

    with col4:
        if model_data:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-value">{:.1%}</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            """.format(model_data['r2_score']), unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🎯 Problem Statement</h3>
        <p>This application predicts manufacturing equipment output using machine learning to help optimize production efficiency. The system forecasts hourly production rates based on key manufacturing parameters.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>📋 Dataset Features</h3>
    </div>
    """, unsafe_allow_html=True)

    features_info = {
        "Injection_Temperature": "Molten plastic temperature (°C)",
        "Injection_Pressure": "Hydraulic pressure (bar)",
        "Cycle_Time": "Time per part cycle (seconds)",
        "Cooling_Time": "Part cooling duration (seconds)",
        "Material_Viscosity": "Plastic flow resistance (Pa·s)",
        "Ambient_Temperature": "Factory temperature (°C)",
        "Machine_Age": "Equipment age in years",
        "Operator_Experience": "Experience level (months)",
        "Maintenance_Hours": "Hours since last maintenance"
    }

    cols = st.columns(3)
    colors = ['#fef7ff', '#f3e8ff', '#e0f2fe', '#ecfdf5', '#fef3c7', '#fffbeb']
    borders = ['#a855f7', '#7c3aed', '#0284c7', '#059669', '#d97706', '#f59e0b']
    for i, (feature, desc) in enumerate(features_info.items()):
        color_idx = i % len(colors)
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {colors[color_idx]} 0%, white 100%); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid {borders[color_idx]};">
                <strong style="color: #1e293b;">{feature}</strong><br>
                <small style="color: #64748b;">{desc}</small>
            </div>
            """, unsafe_allow_html=True)

elif page == "Data Analysis":
    st.markdown('<div class="section-title">📊 Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Relationships"])

    with tab1:
        st.subheader("Feature Distributions")

        features = ['Parts_Per_Hour', 'Cycle_Time', 'Injection_Temperature', 'Injection_Pressure']

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()

        for i, feature in enumerate(features):
            axes[i].hist(df[feature], bins=30, edgecolor='black', alpha=0.7, color='#1f77b4')
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("Feature Correlations")

        numerical_features = [
            'Injection_Temperature', 'Injection_Pressure', 'Cycle_Time',
            'Cooling_Time', 'Material_Viscosity', 'Ambient_Temperature',
            'Machine_Age', 'Operator_Experience', 'Maintenance_Hours', 'Parts_Per_Hour'
        ]

        corr_matrix = df[numerical_features].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix')
        st.pyplot(plt)

        # Top correlations with target
        target_corr = corr_matrix['Parts_Per_Hour'].sort_values(ascending=False)
        st.subheader("Top Correlations with Output")
        st.dataframe(target_corr.head(6))

    with tab3:
        st.subheader("Key Relationships")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Cycle Time vs Output
        axes[0].scatter(df['Cycle_Time'], df['Parts_Per_Hour'], alpha=0.6, color='#1f77b4')
        axes[0].set_xlabel('Cycle Time (seconds)')
        axes[0].set_ylabel('Parts Per Hour')
        axes[0].set_title('Cycle Time vs Output')
        axes[0].grid(True, alpha=0.3)

        # Temperature vs Output
        axes[1].scatter(df['Injection_Temperature'], df['Parts_Per_Hour'], alpha=0.6, color='#ff7f0e')
        axes[1].set_xlabel('Injection Temperature (°C)')
        axes[1].set_ylabel('Parts Per Hour')
        axes[1].set_title('Temperature vs Output')
        axes[1].grid(True, alpha=0.3)

        # Operator Experience vs Output
        axes[2].scatter(df['Operator_Experience'], df['Parts_Per_Hour'], alpha=0.6, color='#2ca02c')
        axes[2].set_xlabel('Operator Experience (months)')
        axes[2].set_ylabel('Parts Per Hour')
        axes[2].set_title('Experience vs Output')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

elif page == "Model Performance":
    st.markdown('<div class="section-title">🤖 Model Performance</div>', unsafe_allow_html=True)

    if model_data is None:
        st.error("Model not loaded. Please run the analysis first.")
    else:
        # Model evaluation
        st.subheader("Model Evaluation")

        features = model_data['features']
        df_model = df[features + ['Parts_Per_Hour']].dropna()
        X = df_model[features]
        y = df_model['Parts_Per_Hour']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = model_data['scaler'].transform(X_test)
        y_pred = model_data['model'].predict(X_test_scaled)

        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("R² Score", f"{model_data['r2_score']:.4f}")
        with col2:
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        with col3:
            st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        with col4:
            threshold = y_test.mean()
            y_test_class = (y_test > threshold).astype(int)
            y_pred_class = (y_pred > threshold).astype(int)
            accuracy = accuracy_score(y_test_class, y_pred_class)
            st.metric("Accuracy", f"{accuracy:.1%}")

        # Additional metrics
        col5, col6 = st.columns(2)
        with col5:
            precision = precision_score(y_test_class, y_pred_class, zero_division=0)
            st.metric("Precision", f"{precision:.4f}")
        with col6:
            f1 = f1_score(y_test_class, y_pred_class, zero_division=0)
            st.metric("F1 Score", f"{f1:.4f}")

        # Predictions vs Actual
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color='#1f77b4', s=50)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        ax.set_xlabel('Actual Parts Per Hour')
        ax.set_ylabel('Predicted Parts Per Hour')
        ax.set_title('Predictions vs Actual Values')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Feature importance
        st.subheader("Feature Importance")

        feature_importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': model_data['model'].coef_,
            'Absolute': np.abs(model_data['model'].coef_)
        }).sort_values('Absolute', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_importance['Feature'], feature_importance['Absolute'], color='#1f77b4')
        ax.set_xlabel('Absolute Coefficient')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Model Coefficients")
        st.dataframe(feature_importance[['Feature', 'Coefficient']].round(4))

elif page == "Make Prediction":
    st.markdown('<div class="section-title">🔮 Make Prediction</div>', unsafe_allow_html=True)

    if model_data is None:
        st.error("Model not loaded. Please run the analysis first.")
    else:
        st.write("Enter manufacturing parameters to predict hourly output:")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                temp = st.slider("Injection Temperature (°C)", 180.0, 250.0, 220.0, 0.1)
                pressure = st.slider("Injection Pressure (bar)", 80.0, 150.0, 120.0, 0.1)
                cycle_time = st.slider("Cycle Time (seconds)", 15.0, 45.0, 25.0, 0.1)

            with col2:
                cooling_time = st.slider("Cooling Time (seconds)", 8.0, 20.0, 12.0, 0.1)
                viscosity = st.slider("Material Viscosity (Pa·s)", 100.0, 400.0, 300.0, 1.0)
                ambient_temp = st.slider("Ambient Temperature (°C)", 18.0, 28.0, 25.0, 0.1)

            with col3:
                machine_age = st.slider("Machine Age (years)", 1.0, 15.0, 5.0, 0.1)
                operator_exp = st.slider("Operator Experience (months)", 1.0, 120.0, 60.0, 1.0)
                maintenance = st.slider("Maintenance Hours", 0.0, 200.0, 50.0, 1.0)

            submitted = st.form_submit_button("🔮 Predict Output")

            if submitted:
                # Prepare input
                input_data = np.array([[temp, pressure, cycle_time, cooling_time,
                                      viscosity, ambient_temp, machine_age,
                                      operator_exp, maintenance]])

                # Scale and predict
                input_scaled = model_data['scaler'].transform(input_data)
                prediction = model_data['model'].predict(input_scaled)[0]
                prediction = max(0, prediction)

                # Display result
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>Prediction Result</h2>
                    <div class="prediction-value">{prediction:.1f}</div>
                    <p>Parts Per Hour</p>
                    <p>Model R² Score: {model_data['r2_score']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)

                # Interpretation
                avg_output = df['Parts_Per_Hour'].mean()
                if prediction > avg_output * 1.1:
                    st.success("🎉 Excellent performance! Output above average.")
                elif prediction > avg_output * 0.9:
                    st.info("✅ Good performance! Output within normal range.")
                else:
                    st.warning("⚠️ Below average performance. Consider optimization.")

                # Optimal feature values
                st.markdown("""
                <div class="card">
                    <h3>💡 Optimal Feature Values Suggested</h3>
                    <p>These are the recommended values for maximizing manufacturing output:</p>
                </div>
                """, unsafe_allow_html=True)

                optimal_values = {
                    "Injection Temperature (°C)": "210-230",
                    "Injection Pressure (bar)": "100-130 (optimized per material)",
                    "Cycle Time (seconds)": "<30",
                    "Cooling Time (seconds)": "8-12",
                    "Material Viscosity (Pa·s)": "200-300",
                    "Ambient Temperature (°C)": "20-25",
                    "Machine Age (years)": "<5",
                    "Operator Experience (months)": ">60",
                    "Maintenance Hours": "<50"
                }

                cols = st.columns(3)
                opt_colors = ['#fef7ff', '#f3e8ff', '#e0f2fe', '#ecfdf5', '#fef3c7', '#fffbeb', '#f0f9ff', '#dbeafe', '#f0fdf4']
                opt_borders = ['#a855f7', '#7c3aed', '#0284c7', '#059669', '#d97706', '#f59e0b', '#2563eb', '#1d4ed8', '#16a34a']
                for i, (feature, value) in enumerate(optimal_values.items()):
                    color_idx = i % len(opt_colors)
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {opt_colors[color_idx]} 0%, white 100%); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {opt_borders[color_idx]}; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                            <strong style="color: #1e293b;">{feature}</strong><br>
                            <span style="color: {opt_borders[color_idx]}; font-weight: 500;">{value}</span>
                        </div>
                        """, unsafe_allow_html=True)

elif page == "Insights":
    st.markdown('<div class="section-title">💡 Insights & Recommendations</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🎯 Key Findings</h3>
    </div>
    """, unsafe_allow_html=True)

    insights = [
        "🔄 **Cycle Time** has the strongest impact on output (coefficient: -9.71)",
        "👥 **Operator Experience** positively affects productivity (+3.91)",
        "🌡️ **Injection Temperature** optimization improves efficiency (+2.02)",
        "🤖 Model achieves high accuracy in predicting manufacturing output"
    ]

    for insight in insights:
        st.markdown(f"""
        <div class="insights-card" style="padding: 1rem; border-radius: 8px; margin: 0.5rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            {insight}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🏭 Optimization Recommendations</h3>
    </div>
    """, unsafe_allow_html=True)

    recommendations = [
        ("Cycle Time", "Reduce cycle time below 30 seconds for optimal output"),
        ("Temperature Control", "Maintain injection temperature in 210-230°C range"),
        ("Pressure Management", "Balance injection pressure with material properties"),
        ("Maintenance", "Implement preventive maintenance schedules"),
        ("Operator Training", "Invest in experienced operators (>60 months)")
    ]

    cols = st.columns(2)
    for i, (rec_type, rec_desc) in enumerate(recommendations):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="recommendation-card" style="padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <strong style="color: #92400e;">{rec_type}</strong><br>
                <span style="color: #78350f;">{rec_desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>📈 Business Impact</h3>
    </div>
    """, unsafe_allow_html=True)

    impacts = [
        "10-20% improvement in production efficiency",
        "Reduced downtime through predictive maintenance",
        "Better resource utilization and planning",
        "Data-driven decision making for optimization"
    ]

    for impact in impacts:
        st.markdown(f"""
        <div class="impact-card" style="padding: 0.75rem; border-radius: 6px; margin: 0.25rem 0;">
            <span style="color: #065f46;">✓ {impact}</span>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <h3>🏭 Manufacturing Equipment Output Prediction System</h3>
    <p>Data Science Capstone Project - Built with ❤️ using Streamlit & Scikit-learn</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">© 2024 - Optimizing Manufacturing Efficiency Through AI</p>
</div>
""", unsafe_allow_html=True)