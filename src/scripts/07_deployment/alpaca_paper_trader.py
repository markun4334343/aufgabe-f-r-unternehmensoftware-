"""
Script: ALPACA RSI SCALPER (Fixed Data Columns)
Location: src/scripts/07_deployment/alpaca_paper_trader.py
Description:
    - Checks market every 30 seconds.
    - BUYS when AI is Bullish + RSI < 35 (Oversold/Dip).
    - SELLS when AI is Bearish or RSI > 65 or Profit Target Hit.
    - Fixes the 'Quote Asset Volume' error by fetching full data.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import yaml
import time
import sys
from pathlib import Path
from binance.client import Client as BinanceClient
import alpaca_trade_api as tradeapi
from datetime import datetime

# --- 1. SETUP & CONFIG ---
base_dir = Path(__file__).parent.parent.parent.parent
params_path = base_dir / "src" / "conf" / "params.yaml"
keys_path = base_dir / "src" / "conf" / "keys.yaml"

# Load Config
try:
    params = yaml.safe_load(open(params_path))
    keys = yaml.safe_load(open(keys_path))
except:
    # Fallback paths
    keys_path = base_dir / "conf" / "keys.yaml"
    params_path = base_dir / "conf" / "params.yaml"
    params = yaml.safe_load(open(params_path))
    keys = yaml.safe_load(open(keys_path))

# Connect APIs
binance_client = BinanceClient() # Public Data
alpaca = tradeapi.REST(
    keys['ALPACA']['API_KEY'],
    keys['ALPACA']['API_SECRET'],
    keys['ALPACA']['BASE_URL'],
    api_version='v2'
)

# --- ⚡ SCALPING SETTINGS ---
BINANCE_SYMBOL = 'ETHUSDT'
ALPACA_SYMBOL = 'ETH/USD'
QTY_TO_TRADE = 0.5
RSI_PERIOD = 14
RSI_OVERSOLD = 35   # Buy Zone
RSI_OVERBOUGHT = 65 # Sell Zone

# Load Brains
DATA_PATH = Path(params['DATA_ACQUISITON']['DATA_PATH'])
PROCESSED_DIR = DATA_PATH / "Processed"
MODELS_DIR = base_dir / "src" / "models"
if not MODELS_DIR.exists(): MODELS_DIR = base_dir / "models"

print("🧠 Loading Model & Scaler...")
model = xgb.XGBClassifier()
model.load_model(MODELS_DIR / "xgb_model_baseline.json")
scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")
feature_cols = list(scaler.feature_names_in_)

# --- 2. INDICATOR FUNCTIONS ---

def calculate_rsi(series, period=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_market_data():
    """Fetches FULL data to satisfy the Scaler requirements."""
    # Fetch 100 candles (enough for RSI + Lookback)
    klines = binance_client.get_historical_klines(BINANCE_SYMBOL, BinanceClient.KLINE_INTERVAL_1MINUTE, limit=100)

    # ✅ FIX: Define ALL columns so 'Quote Asset Volume' exists
    cols = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base', 'Taker Buy Quote', 'Ignore']

    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    # Convert numeric columns to float
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Number of Trades']
    df[numeric_cols] = df[numeric_cols].astype(float)

    # ⚡ Calculate RSI Live
    df['RSI'] = calculate_rsi(df['Close'], RSI_PERIOD)
    return df

def generate_features(df_raw):
    """Matches the model's training features."""
    df = df_raw.copy()

    # 1. Log Returns
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

    # 2. VWAP
    v = df['Volume'].replace(0, np.nan)
    df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=30).sum() / v.rolling(window=30).sum()

    # 3. Moving Averages & Volatility
    for t in params['MODEL_CONFIG']['LOOKBACK_WINDOWS']:
        df[f'MA_{t}'] = df['Close'].rolling(window=t).mean()
        df[f'Vol_{t}'] = df['log_ret'].rolling(window=t).std()

    df = df.ffill().fillna(0)

    # Return exactly what the scaler expects
    return df.iloc[[-1]][feature_cols]

# --- 3. TRADING LOGIC ---
def run_scalper():
    timestamp = datetime.now().strftime('%H:%M:%S')

    # 1. Get Data & Indicators
    df_raw = get_market_data()
    current_rsi = df_raw['RSI'].iloc[-1]
    current_price = df_raw['Close'].iloc[-1]

    # 2. Get AI Prediction
    X_live = generate_features(df_raw)
    X_scaled = scaler.transform(X_live.values)
    probs = model.predict_proba(X_scaled)[0]
    p_up = probs[2]
    p_down = probs[0]

    # 3. Decision Logic (AI + RSI Filter)
    print(f"\n--- ⏱️ {timestamp} | Price: ${current_price:.2f} | RSI: {current_rsi:.1f} ---")
    print(f"   🤖 AI Confidence: Up {p_up:.2f} vs Down {p_down:.2f}")

    # CHECK POSITION
    try:
        pos = alpaca.get_position(ALPACA_SYMBOL)
        qty = float(pos.qty)
        avg_entry = float(pos.avg_entry_price)
        pl_pct = (current_price - avg_entry) / avg_entry * 100
        print(f"   💼 Holding: {qty} ETH | P/L: {pl_pct:.2f}%")
    except:
        qty = 0
        avg_entry = 0
        print("   💼 Holding: 0 ETH")

    # BUY LOGIC (Dip Sniper)
    if qty == 0:
        if p_up > p_down and current_rsi < RSI_OVERSOLD:
            print("   🚀 SIGNAL: BUY (Bullish AI + Dip)")
            alpaca.submit_order(ALPACA_SYMBOL, QTY_TO_TRADE, 'buy', 'market', 'gtc')
        else:
            if current_rsi >= RSI_OVERSOLD:
                print(f"   ✋ Waiting... RSI {current_rsi:.1f} is too high (Need < {RSI_OVERSOLD})")
            else:
                print("   ✋ Waiting... AI is bearish.")

    # SELL LOGIC (Profit Taker)
    elif qty > 0:
        take_profit = (current_price > avg_entry * 1.003) # +0.3% Profit
        stop_loss = (current_price < avg_entry * 0.995)   # -0.5% Loss (Safety)

        if p_down > p_up or current_rsi > RSI_OVERBOUGHT or take_profit or stop_loss:
            reason = "AI Bearish" if p_down > p_up else "RSI Overbought"
            if take_profit: reason = "Take Profit (+0.3%)"
            if stop_loss: reason = "Stop Loss (-0.5%)"

            print(f"   📉 SIGNAL: SELL ({reason})")
            alpaca.close_position(ALPACA_SYMBOL)
        else:
            print("   🧘 HODL... Letting profits run.")

if __name__ == "__main__":
    print("🤖 RSI SMART SCALPER STARTED (Ctrl+C to stop)")
    while True:
        try:
            run_scalper()
            print("   ⏳ Checking again in 30s...")
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n🛑 Stopped.")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(10)