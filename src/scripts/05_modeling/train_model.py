"""
Script: Train XGBoost for Breakout Validation
Location: src/scripts/05_modeling/train_model.py
Description: Trains a model to predict if a Breakout will be PROFITABLE.
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, precision_score
import joblib
import yaml
from pathlib import Path
import os

# --- CONFIGURATION ---
base_dir = Path(__file__).parent.parent.parent
params_path = base_dir / "conf" / "params.yaml"
params = yaml.safe_load(open(params_path))
DATA_PATH = Path(params['DATA_ACQUISITON']['DATA_PATH'])
PROCESSED_DIR = DATA_PATH / "Processed"
MODELS_DIR = base_dir / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 1. LOAD DATA ---
print("📂 Loading Processed Data...")
X_train = np.load(PROCESSED_DIR / "X_train.npy")
y_train = np.load(PROCESSED_DIR / "y_train.npy")
X_val   = np.load(PROCESSED_DIR / "X_val.npy")
y_val   = np.load(PROCESSED_DIR / "y_val.npy")

# --- 2. RELABELING FOR BREAKOUTS ---
# Original Labels: -1 (Down), 0 (Flat), 1 (Up)
# New Goal: We want to catch BIG moves (Trends).
# So, we treat -1 (Crash) and 1 (Pump) as "Target" (Class 1).
# We treat 0 (Flat/Chop) as "Ignore" (Class 0).

print("🔄 Relabeling for Trend Detection...")
# If y is -1 or 1, set to 1 (Trend). If 0, set to 0 (No Trend).
y_train_binary = np.where(y_train != 0, 1, 0)
y_val_binary = np.where(y_val != 0, 1, 0)

print(f"   Train Class Balance: {np.bincount(y_train_binary)}")

# --- 3. TRAIN MODEL ---
print("🚀 Training Trend-Confirmation Model...")

# We use a Binary Logistic model now
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic', # Predict Probability of Trend
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20
)

model.fit(
    X_train, y_train_binary,
    eval_set=[(X_val, y_val_binary)],
    verbose=False
)

# --- 4. EVALUATION ---
y_pred = model.predict(X_val)
acc = accuracy_score(y_val_binary, y_pred)
prec = precision_score(y_val_binary, y_pred)

print("\n" + "="*40)
print(f"📊 TREND MODEL RESULTS")
print(f"   Accuracy:  {acc:.2%}")
print(f"   Precision: {prec:.2%} (How often is the trend real?)")
print("="*40)

# --- 5. SAVE ---
model_path = MODELS_DIR / "xgb_model_baseline.json"
model.save_model(model_path)
print(f"✅ Model saved to: {model_path}")