"""
Script: Data Preparation (Feature Engineering and Target Creation)
Location: src/scripts/03_data_preparation/data_preparation.py
Description: Loads raw data, generates technical indicators (MA, Volatility, VWAP),
             and creates the multi-class prediction target variables.
"""

import pandas as pd
import yaml
import os
from pathlib import Path
import numpy as np

# --- CONFIGURATION & PATH SETUP ---
# Path logic adjusted for: .../src/scripts/03_data_preparation/data_preparation.py
base_dir = Path(__file__).parent.parent.parent
params_path = base_dir / "conf" / "params.yaml"

print(f"Loading config from: {params_path}")
try:
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
except FileNotFoundError:
    print(f"❌ ERROR: Configuration file not found at {params_path}")
    exit(1)

# Extract Parameters and Paths
MODEL_CONFIG = params['MODEL_CONFIG']
LOOKBACK_WINDOWS = MODEL_CONFIG['LOOKBACK_WINDOWS']
HORIZONS = MODEL_CONFIG['PREDICTION_HORIZONS']
VOLATILITY_THRESHOLD = MODEL_CONFIG['VOLATILITY_THRESHOLD']
FEATURE_SETTINGS = MODEL_CONFIG['FEATURES']

DATA_PATH = Path(params['DATA_ACQUISITON']['DATA_PATH'])
INPUT_FILE_PATH = DATA_PATH / "Bars_1m_crypto" / "ETHUSDT.parquet"
OUTPUT_FILE_PATH = DATA_PATH / "ETHUSDT_preprocessed.parquet"

# --- 1. DATA LOADING ---
print("-" * 50)
print(f"Attempting to load data from: {INPUT_FILE_PATH}")
try:
    df = pd.read_parquet(INPUT_FILE_PATH)
    # Ensure timestamp is the index for time-series operations
    df = df.set_index('timestamp').sort_index()
    print(f"✅ Data loaded successfully. Total records: {len(df)}")
except FileNotFoundError:
    print(f"❌ ERROR: Data file not found at {INPUT_FILE_PATH}. Run data_loader.py first.")
    exit(1)

# Handle potential missing data by simple forward fill (common in financial data)
df = df.ffill()

# --- 2. FEATURE ENGINEERING FUNCTIONS ---

def calculate_ma(data, window):
    """Calculates Simple Moving Average (SMA)."""
    return data['Close'].rolling(window=window).mean()

def calculate_volatility(data, window):
    """Calculates rolling standard deviation of returns."""
    # We use log returns for better stationarity properties
    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    return log_returns.rolling(window=window).std()

def calculate_vwap(data):
    """Calculates Volume Weighted Average Price (VWAP) over a rolling 30-min window."""
    window = 30
    price_volume = data['Close'] * data['Volume']
    return price_volume.rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()

def create_target_label(df, horizon, volatility_threshold):
    """
    Creates a multi-class classification target (1=Up, -1=Down, 0=Flat).
    The threshold is based on the rolling volatility of raw returns.
    """
    # 1. Calculate future close price (C_t+h)
    future_close = df['Close'].shift(-horizon)

    # 2. Calculate future return (R_h)
    future_return = (future_close / df['Close']) - 1

    # 3. Calculate a dynamic volatility threshold (using a fixed 720-min window for stability)
    # We use a 720-min (12-hour) lookback for the standard deviation of returns
    # The return used here is the *actual* raw return, not the future one.
    raw_returns = df['Close'].pct_change()
    volatility = raw_returns.rolling(window=720).std()

    # 4. Define the absolute movement threshold (T)
    movement_threshold = volatility * volatility_threshold

    # 5. Create the classification label
    label = pd.Series(0, index=df.index, dtype=np.int8)

    # Label 1 (Up): Future return is greater than the positive threshold
    label[future_return > movement_threshold] = 1

    # Label -1 (Down): Future return is less than the negative threshold
    label[future_return < -movement_threshold] = -1

    return label.rename(f'target_{horizon}m')


# --- 3. APPLY FEATURE ENGINEERING ---

print("\n--- Generating Features ---")

# A. Lookback Window Features (MA and Volatility)
for t in LOOKBACK_WINDOWS:
    df[f'MA_{t}'] = calculate_ma(df, window=t)
    df[f'Vol_{t}'] = calculate_volatility(df, window=t)
    print(f"   Generated MA_{t} and Vol_{t}")

# B. VWAP Feature
if FEATURE_SETTINGS['USE_VWAP']:
    df['VWAP'] = calculate_vwap(df)
    print("   Generated VWAP")

# C. External Features (Skipped for now)
# NOTE: The USE_GAS_FEES and USE_FUNDING_RATE flags are TRUE in params.yaml,
# but since the corresponding data was not loaded by the data_loader script,
# we skip these features here. They would be implemented by merging external data.
if FEATURE_SETTINGS['USE_GAS_FEES'] or FEATURE_SETTINGS['USE_FUNDING_RATE']:
    print("   ⚠️ Skipped Gas Fees/Funding Rate: External data for these features was not included.")


# --- 4. TARGET VARIABLE CREATION ---

print("\n--- Generating Target Variables ---")
for h in HORIZONS:
    target_series = create_target_label(df, horizon=h, volatility_threshold=VOLATILITY_THRESHOLD)
    df = pd.concat([df, target_series], axis=1)
    print(f"   Generated target_{h}m (Prediction Horizon: {h} minutes)")


# --- 5. FINAL CLEAN-UP AND SAVING ---

# Remove initial rows that contain NaN values due to the lookback windows (max window: 1200)
# and remove final rows with NaN targets due to the future shift
max_window = max(LOOKBACK_WINDOWS)
df_cleaned = df.dropna()

print("\n--- Final Data Status ---")
print(f"Total rows before cleanup: {len(df)}")
print(f"Initial NaNs dropped (due to lookback window {max_window} and target shift): {len(df) - len(df_cleaned)}")
print(f"Total rows prepared for modeling: {len(df_cleaned)}")
print(f"Total columns (Features + Targets): {len(df_cleaned.columns)}")

# Save the final preprocessed data
df_cleaned.to_parquet(OUTPUT_FILE_PATH)
print("-" * 50)
print(f"✅ SUCCESS! Preprocessed data saved to: {OUTPUT_FILE_PATH}")