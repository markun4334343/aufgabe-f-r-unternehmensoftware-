"""
Script: Train XGBoost Model
Location: src/scripts/05_modeling/train_model.py
Description: Loads processed data, trains an XGBoost classifier, and evaluates performance.
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import yaml
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
base_dir = Path(__file__).parent.parent.parent
params_path = base_dir / "conf" / "params.yaml"

print(f"Loading config from: {params_path}")
params = yaml.safe_load(open(params_path))
DATA_PATH = Path(params['DATA_ACQUISITON']['DATA_PATH'])
PROCESSED_DIR = DATA_PATH / "Processed"
MODELS_DIR = base_dir / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 1. LOAD DATA ---
print("-" * 50)
print("📂 Loading Processed Data...")

try:
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_val   = np.load(PROCESSED_DIR / "X_val.npy")
    y_val   = np.load(PROCESSED_DIR / "y_val.npy")
    # We don't touch X_test yet. We keep it for the FINAL evaluation only.

    print(f"✅ Data Loaded.")
    print(f"   Train shape: {X_train.shape}")
    print(f"   Val shape:   {X_val.shape}")
except FileNotFoundError:
    print("❌ Error: Could not find .npy files. Run post_split_processing.py first.")
    exit(1)

# --- 2. PREPARE LABELS ---
# XGBoost expects labels to be 0, 1, 2.
# Currently we have -1 (Down), 0 (Flat), 1 (Up).
# We simply add 1 to everything: -1->0, 0->1, 1->2

print("\n🔄 Adjusting Labels for XGBoost (-1,0,1 -> 0,1,2)...")
y_train_encoded = y_train + 1
y_val_encoded   = y_val + 1

# Check if it worked
unique_classes = np.unique(y_train_encoded)
print(f"   Classes found: {unique_classes} (0=Down, 1=Flat, 2=Up)")

# --- 3. INITIALIZE & TRAIN MODEL ---
print("\n🚀 Starting Training (XGBoost)...")

# Hyperparameters (Baseline)
# These are safe starting points. We can tune them later.
model = xgb.XGBClassifier(
    n_estimators=500,        # Max number of trees
    learning_rate=0.05,      # Step size
    max_depth=6,             # Depth of trees (prevent overfitting)
    subsample=0.8,           # Use 80% of data per tree
    colsample_bytree=0.8,    # Use 80% of features per tree
    objective='multi:softprob', # Multi-class classification
    num_class=3,             # 3 Classes (Down, Flat, Up)
    random_state=42,
    n_jobs=-1,               # Use all CPU cores
    early_stopping_rounds=20 # Stop if validation score doesn't improve for 20 rounds
)

# Fit the model
model.fit(
    X_train, y_train_encoded,
    eval_set=[(X_train, y_train_encoded), (X_val, y_val_encoded)],
    verbose=True  # Prints progress
)

# --- 4. EVALUATION ---
print("\n" + "="*50)
print("📊 EVALUATION ON VALIDATION SET")
print("="*50)

# Make predictions
y_pred_encoded = model.predict(X_val)
# Convert back to original labels (-1, 0, 1) for readability
y_pred = y_pred_encoded - 1

# Metrics
acc = accuracy_score(y_val, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['Down (-1)', 'Flat (0)', 'Up (1)']))

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(cm)

# --- 5. FEATURE IMPORTANCE ---
# This is crucial for your findings section
print("\n--- Feature Importance ---")
feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])] # Placeholder names if we don't have list
# Try to load scaler to get real names if possible, otherwise just use indices
try:
    scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")
    if hasattr(scaler, 'feature_names_in_'):
        feature_names = scaler.feature_names_in_
except:
    pass

# Get importance
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

print("Top 10 Most Important Features:")
for i in range(10):
    print(f"   {i+1}. {feature_names[indices[i]]}: {importance[indices[i]]:.4f}")

# --- 6. SAVE MODEL ---
model_path = MODELS_DIR / "xgb_model_baseline.json"
model.save_model(model_path)
print(f"\n✅ Model saved to: {model_path}")