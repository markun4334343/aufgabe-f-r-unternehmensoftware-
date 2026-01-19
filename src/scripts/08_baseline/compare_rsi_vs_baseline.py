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
START_DATE = '2022-01-01'
END_DATE = '2026-02-01'

INITIAL_CAPITAL = 100000
FEE = 0.001  # 0.1% Fee

# STRATEGY SETTINGS
LOOKBACK_PERIOD = 192   # 48 Hours (Breakout Window)
TRAILING_STOP_PCT = 0.02 # 2% Trailing Stop
AI_CONFIDENCE = 0.60    # AI must be 60% sure it's a trend

# --- 2. LOAD DATA ---
base_dir = Path(__file__).parent.parent.parent.parent
data_path = base_dir / "src" / "Data" / "Bars_1m_crypto" / "ETHUSDT.parquet"
model_path = base_dir / "src" / "models" / "xgb_model_baseline.json"
scaler_path = base_dir / "src" / "Data" / "Processed" / "scaler.pkl"

if not data_path.exists():
    data_path = Path(r"C:\Users\rakun\Desktop\projket fur unternehmenssofware\etherium projekt\src\Data\Bars_1m_crypto\ETHUSDT.parquet")

print(f"⏳ Loading Data...")
df_raw = pd.read_parquet(data_path)

df_raw = df_raw.reset_index()
time_col = next((col for col in df_raw.columns if col.lower() in ['timestamp', 'date', 'datetime']), None)
if not time_col: exit()
df_raw[time_col] = pd.to_datetime(df_raw[time_col])
df_raw = df_raw.set_index(time_col).sort_index()

print("🔄 Resampling to 15-Minute Candles...")
agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
if 'Quote Asset Volume' in df_raw.columns: agg_dict['Quote Asset Volume'] = 'sum'
if 'Number of Trades' in df_raw.columns: agg_dict['Number of Trades'] = 'sum'

df = df_raw.resample('15min').agg(agg_dict).dropna()

try:
    df = df.loc[START_DATE:END_DATE].copy()
except KeyError:
    df = df.loc[START_DATE:].copy()

print(f"✅ Data Ready: {len(df)} candles.")

# --- 3. INDICATORS ---

# 1. Donchian Channels (The Strategy)
df['HIGH_channel'] = df['High'].rolling(window=LOOKBACK_PERIOD).max().shift(1)
df['LOW_channel'] = df['Low'].rolling(window=LOOKBACK_PERIOD).min().shift(1)

# 2. Features for AI (Must match training!)
df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
v = df['Volume'].replace(0, np.nan)
df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=30).sum() / v.rolling(window=30).sum()
for t in [5, 10, 15, 20, 30, 60, 120, 240, 360, 480, 720, 960, 1200]:
    df[f'MA_{t}'] = df['Close'].rolling(window=t).mean()
    df[f'Vol_{t}'] = df['log_ret'].rolling(window=t).std()
df = df.fillna(0)

# --- 4. AI PREDICTION ---
print("🧠 AI Predicting Trend Probability...")
model = xgb.XGBClassifier()
model.load_model(model_path)
scaler = joblib.load(scaler_path)

required_features = list(scaler.feature_names_in_)
for feature in required_features:
    if feature not in df.columns: df[feature] = 0.0

X_scaled = scaler.transform(df[required_features])

# Predict Probability of Class 1 (Trend)
probs = model.predict_proba(X_scaled)
df['trend_prob'] = probs[:, 1]  # Column 1 is "Trend" probability

# --- 5. SIMULATION LOOP (AI + BREAKOUT) ---
print("⚔️ Running AI-Filtered Breakout Strategy...")

cash = INITIAL_CAPITAL
eth_amt = 0
position = "FLAT"
entry_price = 0.0
equity_curve = []
stop_price = 0.0

closes = df['Close'].values
high_channels = df['HIGH_channel'].values
low_channels = df['LOW_channel'].values
ai_probs = df['trend_prob'].values

for i in range(len(df)):
    price = closes[i]
    breakout_high = high_channels[i]
    breakout_low = low_channels[i]
    ai_confidence = ai_probs[i]

    # 1. MARK TO MARKET
    if position == "LONG":
        current_equity = cash + (eth_amt * price)
    elif position == "SHORT":
        pnl = (entry_price - price) * abs(eth_amt)
        current_equity = cash + pnl
    else:
        current_equity = cash

    equity_curve.append(current_equity)

    if np.isnan(breakout_high): continue

    # 2. TRADING LOGIC
    if position == "FLAT":

        # 🟢 LONG ENTRY (Breakout + AI)
        if price > breakout_high:
            # AI FILTER: Only enter if AI is 60% sure it's a trend
            if ai_confidence > AI_CONFIDENCE:
                spend = cash * 0.99
                fees = spend * FEE
                eth_amt = (spend - fees) / price
                cash -= spend
                position = "LONG"
                entry_price = price
                stop_price = price * (1 - TRAILING_STOP_PCT)

        # 🔴 SHORT ENTRY (Breakdown + AI)
        elif price < breakout_low:
            # AI FILTER: Only enter if AI is 60% sure it's a trend
            if ai_confidence > AI_CONFIDENCE:
                collateral = cash * 0.99
                fees = collateral * FEE
                eth_amt = -(collateral - fees) / price
                position = "SHORT"
                entry_price = price
                stop_price = price * (1 + TRAILING_STOP_PCT)

    elif position == "LONG":
        # Trailing Stop
        new_stop = price * (1 - TRAILING_STOP_PCT)
        if new_stop > stop_price: stop_price = new_stop

        # Exit
        if price < stop_price:
            revenue = eth_amt * price
            fees = revenue * FEE
            cash += (revenue - fees)
            eth_amt = 0
            position = "FLAT"
            entry_price = 0

    elif position == "SHORT":
        # Trailing Stop
        new_stop = price * (1 + TRAILING_STOP_PCT)
        if new_stop < stop_price: stop_price = new_stop

        # Exit
        if price > stop_price:
            pnl = (entry_price - price) * abs(eth_amt)
            exit_value = price * abs(eth_amt)
            fees = exit_value * FEE
            cash = cash + pnl - fees
            eth_amt = 0
            position = "FLAT"
            entry_price = 0

# --- 6. REPORT ---
df['Equity'] = equity_curve
df['Benchmark'] = INITIAL_CAPITAL * (df['Close'] / df['Close'].iloc[0])

final_val = equity_curve[-1]
profit = final_val - INITIAL_CAPITAL
last_date = df.index[-1].strftime('%Y-%m-%d')

print("\n" + "="*40)
print(f"💰 FINAL BALANCE: ${final_val:,.2f}")
print(f"📈 NET PROFIT:     ${profit:,.2f}")
print("="*40)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Benchmark'], color='gray', alpha=0.3, label='Buy & Hold')
plt.plot(df.index, df['Equity'], color='#00C805', linewidth=2, label='AI-Enhanced Breakout Sniper')
plt.title(f"Final Thesis Strategy (AI + Price Action) ({START_DATE} - {last_date})", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gcf().autofmt_xdate()
plt.savefig("final_thesis_result.png", dpi=300)
print("✅ Chart saved: final_thesis_result.png")
plt.show()