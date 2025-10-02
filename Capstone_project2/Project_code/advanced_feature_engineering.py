import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(X):
    """
    Advanced feature engineering techniques for heart disease prediction
    """
    X = X.copy()
    
    # 1. MEDICAL DOMAIN FEATURES
    # Risk stratification features
    X['age_risk_category'] = pd.cut(X['age'], 
                                   bins=[0, 45, 55, 65, 100], 
                                   labels=[0, 1, 2, 3])
    
    # Blood pressure categories (AHA guidelines)
    X['bp_category'] = pd.cut(X['resting_blood_pressure'],
                             bins=[0, 120, 130, 140, 180, 300],
                             labels=[0, 1, 2, 3, 4])
    
    # Cholesterol risk levels
    X['chol_risk'] = pd.cut(X['cholesterol'],
                           bins=[0, 200, 240, 300, 600],
                           labels=[0, 1, 2, 3])
    
    # 2. CARDIOVASCULAR FITNESS INDICATORS
    # Age-adjusted max heart rate (220 - age is theoretical max)
    X['hr_reserve'] = X['max_heart_rate'] - (220 - X['age'])
    X['hr_percentage'] = X['max_heart_rate'] / (220 - X['age'])
    
    # Exercise capacity indicator
    X['exercise_tolerance'] = X['max_heart_rate'] * (1 - X['exercise_induced_angina'])
    
    # 3. COMPOSITE RISK SCORES
    # Framingham-like risk score components
    X['metabolic_risk'] = (X['cholesterol'] / 200) * (X['fasting_blood_sugar'] + 1)
    X['cardiac_stress'] = X['st_depression'] * (X['st_slope'] + 1)
    X['vessel_disease_severity'] = X['num_major_vessels'] * (X['thalassemia'] + 1)
    
    # 4. INTERACTION FEATURES (Enhanced)
    # Age interactions
    X['age_sex_interaction'] = X['age'] * X['sex']
    X['age_chest_pain'] = X['age'] * X['chest_pain_type']
    X['age_bp_interaction'] = X['age'] * X['resting_blood_pressure']
    
    # Cardiovascular interactions
    X['bp_chol_age'] = X['resting_blood_pressure'] * X['cholesterol'] * X['age']
    X['exercise_age_hr'] = X['exercise_induced_angina'] * X['age'] * X['max_heart_rate']
    
    # ECG and exercise interactions
    X['ecg_exercise'] = X['resting_ecg'] * X['exercise_induced_angina']
    X['st_vessel_interaction'] = X['st_depression'] * X['num_major_vessels']
    
    # 5. RATIO FEATURES
    # Physiological ratios
    X['chol_age_ratio'] = X['cholesterol'] / (X['age'] + 1)
    X['bp_age_ratio'] = X['resting_blood_pressure'] / (X['age'] + 1)
    X['hr_age_ratio'] = X['max_heart_rate'] / (X['age'] + 1)
    
    # 6. POLYNOMIAL FEATURES (Selective)
    # Only for most important continuous features
    key_features = ['age', 'max_heart_rate', 'st_depression', 'cholesterol']
    for feature in key_features:
        if feature in X.columns:
            X[f'{feature}_squared'] = X[feature] ** 2
            X[f'{feature}_sqrt'] = np.sqrt(X[feature] + 1)  # +1 to handle zeros
    
    return X

def create_clustering_features(X, n_clusters=5):
    """
    Create clustering-based features for patient stratification
    """
    X_scaled = StandardScaler().fit_transform(X)
    
    # K-means clustering for patient groups
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Distance to cluster centers
    distances = kmeans.transform(X_scaled)
    
    # Add cluster features
    X_clustered = X.copy()
    X_clustered['patient_cluster'] = cluster_labels
    
    for i in range(n_clusters):
        X_clustered[f'distance_to_cluster_{i}'] = distances[:, i]
    
    return X_clustered

def create_pca_features(X, n_components=5):
    """
    Create PCA features for dimensionality reduction
    """
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=n_components, random_state=42)
    pca_features = pca.fit_transform(X_scaled)
    
    # Create PCA dataframe
    pca_df = pd.DataFrame(
        pca_features,
        columns=[f'pca_component_{i+1}' for i in range(n_components)],
        index=X.index
    )
    
    return pd.concat([X, pca_df], axis=1), pca

def advanced_feature_selection(X, y, method='combined'):
    """
    Advanced feature selection combining multiple methods
    """
    selected_features = []
    
    if method in ['statistical', 'combined']:
        # Statistical selection
        selector_stat = SelectKBest(score_func=f_classif, k=15)
        selector_stat.fit(X, y)
        stat_features = X.columns[selector_stat.get_support()].tolist()
        selected_features.extend(stat_features)
    
    if method in ['model_based', 'combined']:
        # Model-based selection
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        selector_model = SelectFromModel(rf, threshold='median')
        selector_model.fit(X, y)
        model_features = X.columns[selector_model.get_support()].tolist()
        selected_features.extend(model_features)
    
    if method in ['recursive', 'combined']:
        # Recursive feature elimination
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        selector_rfe = RFE(rf, n_features_to_select=12)
        selector_rfe.fit(X, y)
        rfe_features = X.columns[selector_rfe.get_support()].tolist()
        selected_features.extend(rfe_features)
    
    # Remove duplicates and return
    return list(set(selected_features))

def create_time_series_features(X):
    """
    Create features that capture temporal patterns (if applicable)
    """
    X = X.copy()
    
    # Age-based temporal features
    X['age_decade'] = (X['age'] // 10) * 10
    X['age_group'] = pd.cut(X['age'], bins=[0, 30, 40, 50, 60, 70, 100], labels=[0,1,2,3,4,5])
    
    # Risk progression features
    X['cumulative_risk'] = (
        X['age'] * 0.1 + 
        X['cholesterol'] * 0.001 + 
        X['resting_blood_pressure'] * 0.01 +
        X['num_major_vessels'] * 0.5
    )
    
    return X

# Example usage function
def enhanced_preprocessing(df, target_col='heart_disease'):
    """
    Complete enhanced preprocessing pipeline
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print("Original features:", X.shape[1])
    
    # 1. Advanced feature engineering
    X_advanced = create_advanced_features(X)
    print("After advanced features:", X_advanced.shape[1])
    
    # 2. Clustering features
    X_clustered = create_clustering_features(X_advanced)
    print("After clustering features:", X_clustered.shape[1])
    
    # 3. PCA features
    X_pca, pca_model = create_pca_features(X_clustered)
    print("After PCA features:", X_pca.shape[1])
    
    # 4. Time series features
    X_temporal = create_time_series_features(X_pca)
    print("After temporal features:", X_temporal.shape[1])
    
    # 5. Feature selection
    selected_features = advanced_feature_selection(X_temporal, y, method='combined')
    X_selected = X_temporal[selected_features]
    print("After feature selection:", X_selected.shape[1])
    print("Selected features:", selected_features[:10], "...")
    
    return X_selected, y, selected_features

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('heart_disease_dataset.csv')
    X_enhanced, y, features = enhanced_preprocessing(df)
    print(f"\nFinal dataset shape: {X_enhanced.shape}")
    print(f"Feature engineering complete!")