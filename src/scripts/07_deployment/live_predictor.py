"""
Script: LIVE PREDICTOR (Simulated Deployment) - NO KEYS VERSION
Location: src/scripts/07_deployment/live_predictor.py
Description:
    Loads the optimized XGBoost model and scaler. Connects to the Binance PUBLIC API
    (no keys required) to fetch raw data, computes features, scales, and generates
    a high-conviction BUY/SHORT signal based on the 0.55 confidence threshold.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import yaml
import time
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & SETUP ---
# Correctly navigate 4 levels up to find 'conf'
base_dir = Path(__file__).parent.parent.parent.parent
params_path = base_dir / "src" / "conf" / "params.yaml"

# Check if params exists
if not params_path.exists():
    # Fallback for the other folder structure mentioned
    params_path = base_dir / "conf" / "params.yaml"

# Import Binance Client
try:
    from binance.client import Client
except ImportError:
    print("❌ ERROR: python-binance library not found. Install with: pip install python-binance")
    sys.exit(1)

# Load Parameters
try:
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
except FileNotFoundError:
    print(f"❌ Error: Configuration file not found at {params_path}")
    sys.exit(1)

# Paths and Parameters
DATA_PATH = Path(params['DATA_ACQUISITON']['DATA_PATH'])
PROCESSED_DIR = DATA_PATH / "Processed"
# Adjust models path if it's in src/models or root/models
MODELS_DIR = base_dir / "src" / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = base_dir / "models"

MODEL_CONFIG = params['MODEL_CONFIG']

# --- WINNING STRATEGY PARAMETERS ---
CONFIDENCE_THRESHOLD = 0.55
SYMBOL = 'ETHUSDT'
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
REQUIRED_HISTORY = 1300

# ✅ PUBLIC CLIENT (No Keys Needed)
client = Client()

# --- 2. CORE UTILITY FUNCTIONS ---

def load_components():
    """Loads saved model and scaler needed for live prediction."""
    print("   Loading Model and Scaler...")

    model_path = MODELS_DIR / "xgb_model_baseline.json"
    scaler_path = PROCESSED_DIR / "scaler.pkl"

    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        sys.exit(1)
    if not scaler_path.exists():
        print(f"❌ Error: Scaler not found at {scaler_path}")
        sys.exit(1)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    scaler = joblib.load(scaler_path)
    feature_cols = list(scaler.feature_names_in_)

    return model, scaler, feature_cols


def get_live_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetches the required historical klines from Binance Public API."""
    try:
        # We fetch slightly more to ensure we have enough valid data after calculations
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
    except Exception as e:
        print(f"   ❌ Binance API Error: {e}")
        return pd.DataFrame()

    # Binance returns strings, we need floats
    # Columns: Open_time, Open, High, Low, Close, Volume, ...
    df = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    # Convert numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Number of Trades']
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df[numeric_cols].copy()


def calculate_live_features(df_raw: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Mirrors the data_preparation.py logic."""
    df = df_raw.copy()

    # 1. Log Returns
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

    # 2. VWAP (Rolling 30)
    # Handle division by zero if volume is 0
    v = df['Volume'].replace(0, np.nan)
    df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=30).sum() / v.rolling(window=30).sum()

    # 3. Features
    lookback_windows = MODEL_CONFIG['LOOKBACK_WINDOWS']
    for t in lookback_windows:
        df[f'MA_{t}'] = df['Close'].rolling(window=t).mean()
        df[f'Vol_{t}'] = df['log_ret'].rolling(window=t).std()

    # 4. Fill NaNs created by rolling windows
    df = df.fillna(method='ffill').fillna(0)

    # 5. Extract ONLY the last row for prediction
    # We must ensure we have the columns in the EXACT order expected by the scaler
    try:
        df_features = df.iloc[[-1]][feature_cols]
    except KeyError as e:
        print(f"❌ Feature Mismatch: {e}")
        print("Model expects features that are not in the dataframe.")
        return pd.DataFrame()

    return df_features

# --- 3. MAIN PREDICTOR LOOP ---

def run_live_prediction(model: xgb.XGBClassifier, scaler: joblib, feature_cols: List[str]):
    print("\n--- 🟢 Starting Live Predictor Cycle (Public Data) ---")

    # 3.1. Fetch Raw Data
    print("   📡 Fetching live data from Binance...")
    df_raw = get_live_klines(SYMBOL, INTERVAL, REQUIRED_HISTORY)

    if df_raw.empty:
        print("   🔴 Failed to fetch data.")
        return

    # 3.2. Feature Engineering
    X_live_df = calculate_live_features(df_raw, feature_cols)
    if X_live_df.empty:
        return

    # 3.3. Scaling
    X_live_scaled = scaler.transform(X_live_df[feature_cols].values)

    # 3.4. Prediction
    probabilities = model.predict_proba(X_live_scaled)[0]

    prob_down = probabilities[0]
    prob_flat = probabilities[1]
    prob_up   = probabilities[2]

    decision = "HOLD/FLAT"
    max_prob = np.max(probabilities)

    if prob_up > CONFIDENCE_THRESHOLD and prob_up == max_prob:
        decision = "BUY (LONG)"
    elif prob_down > CONFIDENCE_THRESHOLD and prob_down == max_prob:
        decision = "SELL (SHORT)"

    print(f"   🕒 Time: {X_live_df.index[0]}")
    print(f"   📊 Price: ${df_raw['Close'].iloc[-1]:.2f}")
    print(f"   🔮 Prediction: P(Down): {prob_down*100:.1f}% | P(Flat): {prob_flat*100:.1f}% | P(Up): {prob_up*100:.1f}%")

    if decision != "HOLD/FLAT":
        print(f"   🚀 **SIGNAL:** {decision} (Confidence: {max_prob*100:.1f}%)")
    else:
        print(f"   💤 Action: {decision} (Confidence too low)")

# --- EXECUTION ---
if __name__ == "__main__":
    model, scaler, feature_cols = load_components()
    run_live_prediction(model, scaler, feature_cols)