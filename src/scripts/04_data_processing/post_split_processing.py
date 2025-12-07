"""
Script: Data Processing (Post-Split Splitting & Scaling)
Location: src/scripts/04_data_processing/post_split_processing.py
Description:
    1. Loads 'ETHUSDT_preprocessed.parquet'.
    2. Splits data Chronologically (Train 70% / Val 15% / Test 15%).
    3. Scales Input Features (StandardScaler) fitting ONLY on Training data.
    4. Saves final .npy arrays for Model Training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import yaml
import joblib
import sys
import os

# --- CONFIGURATION ---
# Adjust path to step back out of 'src/scripts/04_data_processing'
base_dir = Path(__file__).parent.parent.parent
params_path = base_dir / "conf" / "params.yaml"

print(f"Loading config from: {params_path}")
try:
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
except FileNotFoundError:
    print(f"❌ ERROR: Configuration file not found at {params_path}")
    exit(1)

DATA_PATH = Path(params['DATA_ACQUISITON']['DATA_PATH'])

# INPUT: Your specific file name
INPUT_FILE = DATA_PATH / "ETHUSDT_preprocessed.parquet"

# OUTPUT: New folder for binary arrays
OUTPUT_DIR = DATA_PATH / "Processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- SETTINGS ---
# We must pick ONE target to train the model on for now.
# You can change this to 'target_15m' or 'target_240m' later.
SELECTED_TARGET = 'target_60m'

# --- 1. LOAD DATA ---
print("-" * 50)
print(f"📂 Loading: {INPUT_FILE}")
if not INPUT_FILE.exists():
    print(f"❌ ERROR: {INPUT_FILE} not found. Run your data_preparation.py first.")
    exit(1)

df = pd.read_parquet(INPUT_FILE)

# Ensure strictly sorted by time
df = df.sort_index()

# --- 2. DEFINE FEATURES & TARGET ---

# Identify all target columns (so we can exclude them from features)
all_targets = [c for c in df.columns if 'target_' in c]
print(f"ℹ️  Found targets in file: {all_targets}")

if SELECTED_TARGET not in df.columns:
    print(f"❌ ERROR: Selected target '{SELECTED_TARGET}' not found in dataframe.")
    exit(1)

# Features = Everything that is NOT a target
feature_cols = [c for c in df.columns if c not in all_targets]
print(f"ℹ️  Input Features ({len(feature_cols)}): {feature_cols}")

# --- 3. CHRONOLOGICAL SPLIT ---
# 70% Train, 15% Val, 15% Test
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Split the dataframe into three chunks
train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

print(f"\n--- ✂️ Split Sizes ---")
print(f"Train: {len(train_df)} rows")
print(f"Val:   {len(val_df)} rows")
print(f"Test:  {len(test_df)} rows")

# --- 4. SCALING (NO LEAKAGE) ---
print("\n--- ⚖️ Scaling Features ---")
scaler = StandardScaler()

# CRITICAL: Fit scaler ONLY on Training features
scaler.fit(train_df[feature_cols])

# Transform inputs (X)
X_train = scaler.transform(train_df[feature_cols])
X_val   = scaler.transform(val_df[feature_cols])
X_test  = scaler.transform(test_df[feature_cols])

# Extract targets (y) - No scaling needed for classification labels
y_train = train_df[SELECTED_TARGET].values
y_val   = val_df[SELECTED_TARGET].values
y_test  = test_df[SELECTED_TARGET].values

# --- 5. SAVE ARRAYS ---
print("\n--- 💾 Saving .npy Arrays ---")

# Save X (Inputs)
np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "X_val.npy", X_val)
np.save(OUTPUT_DIR / "X_test.npy", X_test)

# Save y (Targets)
np.save(OUTPUT_DIR / "y_train.npy", y_train)
np.save(OUTPUT_DIR / "y_val.npy", y_val)
np.save(OUTPUT_DIR / "y_test.npy", y_test)

# Save the scaler for future live deployment
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

print(f"✅ Success! Processed data saved to: {OUTPUT_DIR}")
print(f"   Target used: {SELECTED_TARGET}")