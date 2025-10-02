import os
import json
import joblib
from typing import Dict

import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Data loading & preprocessing
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, target_col: str):
    """Split into features, target and apply scaling"""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# -----------------------------
# Baseline models
# -----------------------------
def train_baselines(X, y) -> Dict:
    """Train a few quick baseline models without tuning"""
    baselines = {
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(),
        "logistic_regression": LogisticRegression(max_iter=500),
        "svm": SVC(probability=True)
    }

    for name, model in baselines.items():
        model.fit(X, y)

    return baselines


# -----------------------------
# Grid Search optimization
# -----------------------------
def run_grid_search(X, y, random_state=42, cv=5) -> Dict:
    dt_param_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    }

    rf_param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }

    lr_param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [1000, 2000]
    }

    svm_param_grid = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1]
    }

    models_and_grids = [
        (DecisionTreeClassifier(random_state=random_state), dt_param_grid, "decision_tree"),
        (RandomForestClassifier(random_state=random_state), rf_param_grid, "random_forest"),
        (LogisticRegression(random_state=random_state, max_iter=2000), lr_param_grid, "logistic_regression"),
        (SVC(probability=True, random_state=random_state), svm_param_grid, "svm")
    ]

    best_models = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for estimator, grid, name in models_and_grids:
        print(f"🔍 Grid searching: {name}")
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring="roc_auc",
            cv=skf,
            n_jobs=-1,
            verbose=1
        )
        gs.fit(X, y)
        best_models[name] = {
            "best_estimator": gs.best_estimator_,
            "best_params": gs.best_params_,
            "best_score": float(gs.best_score_)
        }

    return best_models


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(estimator, X_test, y_test) -> Dict:
    y_pred = estimator.predict(X_test)
    try:
        y_prob = estimator.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = estimator.decision_function(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)

    return {
        "classification_report": report,
        "roc_auc": float(roc_auc)
    }


# -----------------------------
# Load artifacts
# -----------------------------
def load_artifacts(artifacts_dir: str = "artifacts"):
    """Load model, scaler, and metadata from artifacts directory"""
    model_path = os.path.join(artifacts_dir, "best_model.pkl")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    metadata_path = os.path.join(artifacts_dir, "metadata.json")

    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return model, scaler, metadata


# -----------------------------
# Save artifacts
# -----------------------------
def save_artifacts(model, scaler, metadata: Dict, artifacts_dir: str = "artifacts") -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(model, os.path.join(artifacts_dir, "best_model.pkl"))
    if scaler:
        joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
    with open(os.path.join(artifacts_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
