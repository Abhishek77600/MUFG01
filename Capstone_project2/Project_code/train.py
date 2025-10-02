#!/usr/bin/env python3
"""
Script to train models and save artifacts
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from model_utils import (
    load_data,
    preprocess,
    train_baselines,
    run_grid_search,
    evaluate_model,
    save_artifacts
)

# Paths
DATA_PATH = "heart_disease_dataset.csv"
ARTIFACTS_DIR = "artifacts"
TEST_SIZE = 0.2
RANDOM_STATE = 42

print("📂 Loading data...")
df = load_data(DATA_PATH)

print("⚙️ Preprocessing...")
X, y, scaler = preprocess(df, target_col="heart_disease")

print(f"✅ Data shape after preprocessing: {X.shape}, Target shape: {y.shape}")

print("✂️ Splitting into train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

print("🏗️ Training baseline models...")
baselines = train_baselines(X_train, y_train)
print("✅ Baseline models trained")

print("🔍 Running grid search for hyperparameter optimization...")
best_models = run_grid_search(X_train, y_train)
print("✅ Grid search complete")

print("📊 Evaluating optimized models on test set...")
evaluations = {}

for name, info in best_models.items():
    estimator = info['best_estimator']
    eval_res = evaluate_model(estimator, X_test, y_test)
    evaluations[name] = {
        "best_params": info["best_params"],
        "cv_score": info["best_score"],
        "test_eval": eval_res
    }

print("✅ Evaluation complete")

# Choose best model by test ROC-AUC
best_name = max(evaluations.keys(), key=lambda n: evaluations[n]["test_eval"]["roc_auc"])
chosen_estimator = best_models[best_name]["best_estimator"]

metadata = {
    "chosen_model": best_name,
    "evaluations": evaluations
}

save_artifacts(chosen_estimator, scaler, metadata, artifacts_dir=ARTIFACTS_DIR)
print(f"✅ Saved best model: {best_name} to {ARTIFACTS_DIR}")

print("📊 Summary of evaluations (ROC-AUC on test set):")
print(json.dumps(
    {k: {"roc_auc": v["test_eval"]["roc_auc"]} for k, v in evaluations.items()},
    indent=2
))