"""
Script: Strategy Optimizer (Grid Search)
Location: src/scripts/06_backtesting/optimize_strategy.py
Description:
    Loops through multiple Confidence Thresholds and Stop-Loss settings
    to find the profitable 'Sweet Spot' for the strategy.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
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

# SETTINGS TO TEST
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
STOP_LOSSES = [None, 0.01, 0.02, 0.03] # None, 1%, 2%, 3%

INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001

# --- 1. LOAD DATA (ONCE) ---
print("-" * 60)
print("📂 Loading Data & Model...")
model = xgb.XGBClassifier()
model.load_model(MODELS_DIR / "xgb_model_baseline.json")

X_test = np.load(PROCESSED_DIR / "X_test.npy")
# We need raw prices for PnL
df_raw = pd.read_parquet(DATA_PATH / "ETHUSDT_preprocessed.parquet").sort_index()
val_end = int(len(df_raw) * 0.85)
test_df_base = df_raw.iloc[val_end:].copy()

# Pre-calculate probabilities to save time
print("🔮 Generating Probabilities...")
probs = model.predict_proba(X_test)
max_conf = np.max(probs, axis=1)
print(f"   Max Confidence seen in Test Data: {np.max(max_conf)*100:.2f}%")
print("-" * 60)

# --- 2. THE BACKTEST FUNCTION ---
def run_backtest(threshold, stop_loss_pct):
    df = test_df_base.copy()

    # 1. Generate Signals
    df['Signal'] = 0

    # Down (-1)
    down_mask = probs[:, 0] > threshold
    df.loc[down_mask, 'Signal'] = -1

    # Up (1)
    up_mask = probs[:, 2] > threshold
    df.loc[up_mask, 'Signal'] = 1

    # 2. Apply Position Logic
    # We trade at the OPEN of t+1 based on Signal at t
    df['Target_Position'] = df['Signal'].shift(1).fillna(0)

    # 3. Simulate Stop Loss (Simplified Vectorized Approach)
    # If we have a Stop Loss, we check if the Low/High hit it
    # Note: Vectorized Stop Loss is complex, this is a simplified 'End of Candle' check
    # for speed. Real-time would be intra-candle.

    # Calculate simple returns first
    df['Market_Return'] = df['Close'].pct_change()

    # If Stop Loss is active
    if stop_loss_pct is not None:
        # Long Stop: If Low < Entry * (1 - SL)
        # Short Stop: If High > Entry * (1 + SL)
        # Since we are vectorized, we treat a big adverse move as a "Stop Out"
        # and cap the loss at -stop_loss_pct

        # Calculate adverse excursion
        # Long: (Low / Open) - 1
        # Short: 1 - (High / Open)

        # This is an approximation for speed
        df['Intra_Bar_Return'] = 0.0

        # Longs
        longs = df['Target_Position'] == 1
        df.loc[longs, 'Intra_Bar_Return'] = (df.loc[longs, 'Low'] / df.loc[longs, 'Open']) - 1

        # Shorts
        shorts = df['Target_Position'] == -1
        df.loc[shorts, 'Intra_Bar_Return'] = 1 - (df.loc[shorts, 'High'] / df.loc[shorts, 'Open'])

        # Apply Stop
        hit_stop = df['Intra_Bar_Return'] < -stop_loss_pct

        # If stop hit, our return for that bar is fixed at -stop_loss
        # (ignoring slippage for now)
        # We also exit the position (Position becomes 0 for next bar - handled by signal logic next step?
        # No, simpler to just penalize the return)

        # Override the Market Return with Stop Loss limit
        # Strategy Return = Position * Market_Return
        # But if Stop Hit, Strategy Return = -stop_loss_pct - fee

        # We need to construct the Strategy Return column carefully
        df['Strategy_Raw_Return'] = df['Target_Position'] * df['Market_Return']

        # Overwrite with Stop Loss
        # If Position is 1 and Low dropped -> Loss is capped
        df.loc[hit_stop, 'Strategy_Raw_Return'] = -stop_loss_pct

    else:
        # No Stop Loss
        df['Strategy_Raw_Return'] = df['Target_Position'] * df['Market_Return']

    # 4. Calculate Fees
    df['Trade_Count'] = df['Target_Position'].diff().abs()
    df['Fee_Cost'] = df['Trade_Count'] * TRANSACTION_COST

    # 5. Net Return
    df['Strategy_Net_Return'] = df['Strategy_Raw_Return'] - df['Fee_Cost']
    df.fillna(0, inplace=True)

    # 6. Metrics
    total_return = (1 + df['Strategy_Net_Return']).cumprod().iloc[-1] - 1
    num_trades = df['Trade_Count'].sum()

    return total_return, num_trades

# --- 3. RUN THE GRID SEARCH ---
print(f"{'CONFIDENCE':<12} | {'STOP LOSS':<10} | {'NET PROFIT':<12} | {'TRADES':<8}")
print("-" * 60)

results = []

for thresh in THRESHOLDS:
    for sl in STOP_LOSSES:

        # Skip invalid high thresholds if model never reached them
        if thresh > np.max(max_conf):
            continue

        profit, trades = run_backtest(thresh, sl)

        sl_str = "None" if sl is None else f"{sl*100}%"
        print(f"{thresh*100:>9.0f}%   | {sl_str:>10} | {profit*100:>11.2f}% | {trades:>8.0f}")

        results.append({
            'Threshold': thresh,
            'StopLoss': sl,
            'Profit': profit,
            'Trades': trades
        })

print("-" * 60)
# Find Best
if len(results) > 0:
    best_result = max(results, key=lambda x: x['Profit'])
    print("\n🏆 BEST SETTING FOUND:")
    print(f"   Threshold: {best_result['Threshold']*100}%")
    print(f"   Stop Loss: {best_result['StopLoss']}")
    print(f"   Profit:    {best_result['Profit']*100:.2f}%")
else:
    print("No valid results found (Thresholds might be too high).")