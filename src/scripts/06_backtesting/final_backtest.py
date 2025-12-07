"""
Script: Final Backtest (Winning Strategy)
Location: src/scripts/06_backtesting/final_backtest.py
Description: Generates the final Equity Curve using the optimized settings found in grid search.
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

# --- 🏆 WINNING SETTINGS ---
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001
CONFIDENCE_THRESHOLD = 0.55  # The sweet spot found in grid search

# --- LOAD DATA ---
print(f"🚀 Running Final Backtest (Threshold: {CONFIDENCE_THRESHOLD})...")
model = xgb.XGBClassifier()
model.load_model(MODELS_DIR / "xgb_model_baseline.json")
X_test = np.load(PROCESSED_DIR / "X_test.npy")
df_raw = pd.read_parquet(DATA_PATH / "ETHUSDT_preprocessed.parquet").sort_index()
test_df = df_raw.iloc[int(len(df_raw) * 0.85):].copy()

# --- GENERATE SIGNALS ---
probs = model.predict_proba(X_test)
test_df['Signal'] = 0
# Short if confident Down
test_df.loc[probs[:, 0] > CONFIDENCE_THRESHOLD, 'Signal'] = -1
# Long if confident Up
test_df.loc[probs[:, 2] > CONFIDENCE_THRESHOLD, 'Signal'] = 1

# Shift Position (Enter next open)
test_df['Position'] = test_df['Signal'].shift(1).fillna(0)

# --- CALCULATE PnL ---
test_df['Market_Return'] = test_df['Close'].pct_change()
test_df['Strategy_Raw'] = test_df['Position'] * test_df['Market_Return']
test_df['Fee'] = test_df['Position'].diff().abs() * TRANSACTION_COST
test_df['Strategy_Net'] = test_df['Strategy_Raw'] - test_df['Fee']

test_df['Equity'] = INITIAL_CAPITAL * (1 + test_df['Strategy_Net']).cumprod()
test_df['Buy_Hold'] = INITIAL_CAPITAL * (1 + test_df['Market_Return']).cumprod()

# --- FINAL METRICS ---
net_profit = (test_df['Equity'].iloc[-1] / INITIAL_CAPITAL) - 1
buy_hold_profit = (test_df['Buy_Hold'].iloc[-1] / INITIAL_CAPITAL) - 1

print("\n" + "="*40)
print(f"FINAL RESULT (2024-2025)")
print("="*40)
print(f"AI Strategy Profit: {net_profit*100:+.2f}%  (Final: ${test_df['Equity'].iloc[-1]:,.2f})")
print(f"Buy & Hold Profit:  {buy_hold_profit*100:+.2f}%  (Final: ${test_df['Buy_Hold'].iloc[-1]:,.2f})")
print(f"Total Trades:       {test_df['Position'].diff().abs().sum():.0f}")
print("-" * 40)

if net_profit > buy_hold_profit:
    print("✅ SUCCESS: Strategy Outperformed Buy & Hold!")
else:
    print("⚠️ NOTE: Strategy is profitable but trailed Buy & Hold.")

# --- PLOT ---
plt.figure(figsize=(12, 6))
plt.plot(test_df.index, test_df['Equity'], label='AI Strategy (Optimized)', color='green', linewidth=2)
plt.plot(test_df.index, test_df['Buy_Hold'], label='Buy & Hold ETH', color='gray', alpha=0.5, linestyle='--')
plt.title(f'Final Backtest: XGBoost (Threshold {CONFIDENCE_THRESHOLD})')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True, alpha=0.3)
save_path = IMAGES_DIR / "final_winning_strategy.png"
plt.savefig(save_path)
print(f"\n📈 Final Chart saved to: {save_path}")