"""
Script: Backtest XGBoost Model
Location: src/scripts/06_backtesting/backtest_model.py
Description:
    1. Loads the Test Data (Prices & Returns).
    2. Loads the Trained Model.
    3. Simulates a trading strategy (Long/Short) with transaction costs.
    4. Calculates Sharpe Ratio, Max Drawdown, and Cumulative Return.
    5. Plots the Equity Curve.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
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
os.makedirs(IMAGES_DIR, exist_ok=True)

# TRADING PARAMETERS
INITIAL_CAPITAL = 10000.0  # Start with $10,000
TRANSACTION_COST = 0.001   # 0.1% fee per trade (Taker fee)

# --- 1. LOAD DATA ---
print("-" * 50)
print("📂 Loading Data for Backtest...")

# Load Model
model = xgb.XGBClassifier()
model.load_model(MODELS_DIR / "xgb_model_baseline.json")

# Load Processed Features (for prediction)
X_test = np.load(PROCESSED_DIR / "X_test.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")

# Load Original Price Data (for PnL calculation)
# We need to slice it exactly like we did in the split step
df_raw = pd.read_parquet(DATA_PATH / "ETHUSDT_preprocessed.parquet")
df_raw = df_raw.sort_index()

# Re-create the Test Split (Last 15%)
n = len(df_raw)
val_end = int(n * 0.85)
test_df = df_raw.iloc[val_end:].copy()

# Double check alignment
if len(test_df) != len(X_test):
    print("❌ Critical Error: Test Set length mismatch!")
    print(f"   X_test: {len(X_test)}, test_df: {len(test_df)}")
    exit(1)

print(f"✅ Data Aligned. Backtesting on {len(test_df)} minutes of unseen data.")

# --- 2. GENERATE SIGNALS ---
print("\n🔮 Generating Predictions...")
pred_probs = model.predict(X_test)

# Map XGBoost classes (0, 1, 2) to Trading Positions (-1, 0, 1)
# 0 (Down) -> -1 (Short)
# 1 (Flat) ->  0 (Cash)
# 2 (Up)   ->  1 (Long)
test_df['Signal'] = 0
test_df.loc[pred_probs == 0, 'Signal'] = -1
test_df.loc[pred_probs == 1, 'Signal'] = 0
test_df.loc[pred_probs == 2, 'Signal'] = 1

# Shift Signal by 1: We trade at the CLOSE of t based on signal at t.
# So our Position for return at t+1 is determined by Signal at t.
test_df['Position'] = test_df['Signal'].shift(1)

# --- 3. CALCULATE RETURNS ---
print("\n💰 Calculating PnL...")

# Raw Market Return (Buy & Hold Strategy)
test_df['Market_Return'] = test_df['Close'].pct_change()

# Strategy Return = Position * Market Return
test_df['Strategy_Raw_Return'] = test_df['Position'] * test_df['Market_Return']

# Calculate Transaction Costs
# We pay a fee whenever we CHANGE position (e.g., Long -> Short is 2 units of change)
test_df['Trade_Count'] = test_df['Position'].diff().abs()
test_df['Fee_Cost'] = test_df['Trade_Count'] * TRANSACTION_COST

# Net Strategy Return
test_df['Strategy_Net_Return'] = test_df['Strategy_Raw_Return'] - test_df['Fee_Cost']

# Fill NaNs (first row)
test_df.fillna(0, inplace=True)

# Cumulative Returns (Equity Curve)
test_df['Equity_Curve'] = INITIAL_CAPITAL * (1 + test_df['Strategy_Net_Return']).cumprod()
test_df['Buy_Hold_Curve'] = INITIAL_CAPITAL * (1 + test_df['Market_Return']).cumprod()

# --- 4. PERFORMANCE METRICS ---
total_return = (test_df['Equity_Curve'].iloc[-1] / INITIAL_CAPITAL) - 1
buy_hold_return = (test_df['Buy_Hold_Curve'].iloc[-1] / INITIAL_CAPITAL) - 1

# Sharpe Ratio (assuming minute data, annualized by sqrt(minutes in year))
# 365 days * 24 hours * 60 minutes = 525,600
risk_free_rate = 0.0
sharpe_ratio = (test_df['Strategy_Net_Return'].mean() * np.sqrt(525600)) / (test_df['Strategy_Net_Return'].std() + 1e-9)

# Max Drawdown
rolling_max = test_df['Equity_Curve'].cummax()
drawdown = (test_df['Equity_Curve'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print("\n" + "="*40)
print("📊 BACKTEST RESULTS")
print("="*40)
print(f"Start Date:       {test_df.index[0]}")
print(f"End Date:         {test_df.index[-1]}")
print(f"Initial Capital:  ${INITIAL_CAPITAL:,.2f}")
print(f"Final Equity:     ${test_df['Equity_Curve'].iloc[-1]:,.2f}")
print(f"Net Profit:       {total_return*100:.2f}%")
print(f"Buy & Hold:       {buy_hold_return*100:.2f}%")
print("-" * 40)
print(f"Sharpe Ratio:     {sharpe_ratio:.2f}")
print(f"Max Drawdown:     {max_drawdown*100:.2f}%")
print(f"Total Trades:     {test_df['Trade_Count'].sum():.0f}")

# --- 5. PLOTTING ---
plt.figure(figsize=(12, 6))
plt.plot(test_df.index, test_df['Equity_Curve'], label='AI Strategy (Net)', color='blue')
plt.plot(test_df.index, test_df['Buy_Hold_Curve'], label='Buy & Hold ETH', color='gray', alpha=0.6, linestyle='--')
plt.title(f'Equity Curve: XGBoost Strategy vs Buy & Hold (Fee: {TRANSACTION_COST*100}%)')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True, alpha=0.3)

save_path = IMAGES_DIR / "backtest_equity_curve.png"
plt.savefig(save_path)
print(f"\n📈 Equity Curve saved to: {save_path}")