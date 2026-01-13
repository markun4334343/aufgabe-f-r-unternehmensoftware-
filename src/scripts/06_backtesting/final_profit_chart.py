import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# --- 1. CONFIGURATION ---
START_DATE = '2024-01-01'
END_DATE = '2025-01-01'
INITIAL_CAPITAL = 100000
CONFIDENCE_THRESHOLD = 0.55
FEE = 0.001  # 0.1%

# --- 2. PATHS ---
DATA_PATH = r"C:\Users\rakun\Desktop\projket fur unternehmenssofware\etherium projekt\src\Data\Bars_1m_crypto\ETHUSDT.parquet"
SCALER_PATH = r"C:\Users\rakun\Desktop\projket fur unternehmenssofware\etherium projekt\src\Data\Processed\scaler.pkl"
MODEL_PATH = r"C:\Users\rakun\Desktop\projket fur unternehmenssofware\etherium projekt\src\models\xgb_model_baseline.json"

# Fallback for model path
if not Path(MODEL_PATH).exists():
    MODEL_PATH = r"C:\Users\rakun\Desktop\projket fur unternehmenssofware\etherium projekt\models\xgb_model_baseline.json"

print("⏳ Loading Data...")
df = pd.read_parquet(DATA_PATH)

# Ensure Datetime Index
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

# --- 3. FEATURE ENGINEERING (The Missing Step) ---
print("⚙️  Calculating Features (Moving Averages, VWAP, etc.)...")

# 1. Log Returns
df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

# 2. VWAP (Rolling 30)
v = df['Volume'].replace(0, np.nan)
df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=30).sum() / v.rolling(window=30).sum()

# 3. MAs and Volatility
windows = [5, 10, 15, 20, 30, 60, 120, 240, 360, 480, 720, 960, 1200]
for t in windows:
    df[f'MA_{t}'] = df['Close'].rolling(window=t).mean()
    df[f'Vol_{t}'] = df['log_ret'].rolling(window=t).std()

# 4. Cleanup
df = df.fillna(method='ffill').fillna(0)

# --- 4. FILTER DATE RANGE ---
# We do this AFTER features to ensure MAs have enough data to calculate correctly
df = df.loc[START_DATE:END_DATE].copy()
print(f"✅ Data Ready: {len(df)} rows ({START_DATE} to {END_DATE})")

# --- 5. PREDICT & SIMULATE ---
print("🧠 Loading AI & Predicting...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Select only the features the scaler expects
feature_cols = list(scaler.feature_names_in_)
X_scaled = scaler.transform(df[feature_cols])
probs = model.predict_proba(X_scaled)

print("💰 Simulating Trades...")
cash = INITIAL_CAPITAL
eth_balance = 0
position = "FLAT"
equity_curve = []

for i in range(len(df)):
    current_price = df['Close'].iloc[i]
    p_down = probs[i][0]
    p_up = probs[i][2]

    # BUY LOGIC
    if position == "FLAT" and p_up > CONFIDENCE_THRESHOLD:
        spend = cash * 0.99
        fees = spend * FEE
        eth_balance = (spend - fees) / current_price
        cash -= spend
        position = "LONG"

    # SELL LOGIC
    elif position == "LONG" and (p_up <= CONFIDENCE_THRESHOLD or p_down > CONFIDENCE_THRESHOLD):
        revenue = eth_balance * current_price
        fees = revenue * FEE
        cash += (revenue - fees)
        eth_balance = 0
        position = "FLAT"

    # Calc Value
    total_val = cash + (eth_balance * current_price)
    equity_curve.append(total_val)

df['My_Portfolio'] = equity_curve

# --- 6. PLOT ---
print("📸 Plotting...")
df_plot = df[['My_Portfolio']].resample('4h').last().dropna()

plt.figure(figsize=(12, 6))
plt.plot(df_plot.index, df_plot['My_Portfolio'], color='#00C805', linewidth=2.5, label='My AI Strategy')
plt.fill_between(df_plot.index, df_plot['My_Portfolio'], INITIAL_CAPITAL * 0.95, color='#E6F9E6', alpha=0.5)

plt.title(f"AI Bot Performance: $100k Start ({START_DATE} to {END_DATE})", fontsize=14, fontweight='bold')
plt.ylabel("Portfolio Value ($)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gcf().autofmt_xdate()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.savefig("my_actual_profit_chart.png", dpi=300)
print("✅ DONE. Chart saved as: my_actual_profit_chart.png")
plt.show()