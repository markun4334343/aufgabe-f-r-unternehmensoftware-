"""
Script: Refined Backtest with Confidence Thresholds
Location: src/scripts/06_backtesting/backtest_refined.py
Description:
    Prevents over-trading by only taking signals with high probability.
    Reduces "flickering" and fee burn.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
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
IMAGES_DIR = base_dir / "images"

# --- ⚙️ NEW SETTINGS ---
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001   # 0.1% fee

# CRITICAL FIX: Only trade if confidence is > 50% (Baseline is 33%)
CONFIDENCE_THRESHOLD = 0.50

# --- 1. LOAD DATA ---
print("-" * 50)
print("📂 Loading Data...")
model = xgb.XGBClassifier()
model.load_model(MODELS_DIR / "xgb_model_baseline.json")

X_test = np.load(PROCESSED_DIR / "X_test.npy")
# Load Prices
df_raw = pd.read_parquet(DATA_PATH / "ETHUSDT_preprocessed.parquet").sort_index()
val_end = int(len(df_raw) * 0.85)
test_df = df_raw.iloc[val_end:].copy()

# --- 2. GENERATE SMART SIGNALS ---
print(f"\n🔮 Generating Signals (Threshold: {CONFIDENCE_THRESHOLD*100}%)...")

# Get probabilities instead of hard labels
# Shape: [rows, 3] -> [Prob_Down, Prob_Flat, Prob_Up]
probs = model.predict_proba(X_test)

# Initialize Signal as 0 (Hold/Flat)
test_df['Signal'] = 0

# Apply Threshold Logic
# Class 0 in XGB = Down (-1)
# Class 2 in XGB = Up (+1)

# If probability of Down > Threshold -> Short (-1)
down_mask = probs[:, 0] > CONFIDENCE_THRESHOLD
test_df.loc[down_mask, 'Signal'] = -1

# If probability of Up > Threshold -> Long (1)
up_mask = probs[:, 2] > CONFIDENCE_THRESHOLD
test_df.loc[up_mask, 'Signal'] = 1

# (Everything else stays 0/Flat)

# Shift Signal (Trade at Open of next candle based on Close of previous)
test_df['Position'] = test_df['Signal'].shift(1)

# --- 3. CALCULATE RETURNS ---
print("\n💰 Calculating PnL...")

test_df['Market_Return'] = test_df['Close'].pct_change()
test_df['Strategy_Raw_Return'] = test_df['Position'] * test_df['Market_Return']

# Calculate Trades: Any change in position counts as a trade
test_df['Trade_Count'] = test_df['Position'].diff().abs()

# Fee logic:
# Going 0 to 1 = 1 unit of fee
# Going 1 to -1 = 2 units of fee (Sell long, Enter short)
test_df['Fee_Cost'] = test_df['Trade_Count'] * TRANSACTION_COST

test_df['Strategy_Net_Return'] = test_df['Strategy_Raw_Return'] - test_df['Fee_Cost']
test_df.fillna(0, inplace=True)

# Equity Curve
test_df['Equity_Curve'] = INITIAL_CAPITAL * (1 + test_df['Strategy_Net_Return']).cumprod()
test_df['Buy_Hold_Curve'] = INITIAL_CAPITAL * (1 + test_df['Market_Return']).cumprod()

# --- 4. METRICS ---
final_equity = test_df['Equity_Curve'].iloc[-1]
total_return = (final_equity / INITIAL_CAPITAL) - 1
total_trades = test_df['Trade_Count'].sum()
avg_trades_per_day = total_trades / (len(test_df) / 1440) # 1440 mins in a day

print("\n" + "="*40)
print("📊 REFINED BACKTEST RESULTS")
print("="*40)
print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD*100}%")
print(f"Final Equity:     ${final_equity:,.2f}")
print(f"Net Profit:       {total_return*100:.2f}%")
print(f"Total Trades:     {total_trades:.0f} (Was ~36,000)")
print(f"Trades Per Day:   {avg_trades_per_day:.1f}")
print("-" * 40)

# --- 5. PLOT ---
plt.figure(figsize=(12, 6))
plt.plot(test_df.index, test_df['Equity_Curve'], label='AI Strategy (Smart)', color='green')
plt.plot(test_df.index, test_df['Buy_Hold_Curve'], label='Buy & Hold', color='gray', alpha=0.5, linestyle='--')
plt.title(f'Refined Strategy (Threshold {CONFIDENCE_THRESHOLD}) vs Buy & Hold')
plt.ylabel('Equity ($)')
plt.legend()
plt.grid(True, alpha=0.3)
save_path = IMAGES_DIR / "backtest_refined.png"
plt.savefig(save_path)
print(f"📈 Plot saved to: {save_path}")