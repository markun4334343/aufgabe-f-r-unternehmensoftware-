"""
Script: Binance Data Loader (Public/Free Mode)
Location: src/scripts/01_data_acquisition/data_loader.py
Description: Downloads 1m ETHUSDT data using public endpoints.
             NO API KEYS or DEPOSIT required.
"""

from binance.client import Client
import pandas as pd
from datetime import datetime
import yaml
import os
import time
from pathlib import Path

# --- CONFIGURATION (FIXED PATH) ---
# Your script location: .../src/scripts/01_data_acquisition/data_loader.py
# Target config location: .../src/conf/params.yaml

# 1. Path(__file__).parent.parent: -> .../src/scripts/
# 2. .parent.parent.parent: -> .../src/ (This gets us to the correct root)
base_dir = Path(__file__).parent.parent.parent
params_path = base_dir / "conf" / "params.yaml"

print(f"Loading config from: {params_path}")
try:
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
except FileNotFoundError:
    print(f"❌ ERROR: Configuration file not found at {params_path}")
    print("Please check the location of your 'params.yaml' file.")
    exit(1)


# Extract Data Parameters
PATH_BARS = Path(params['DATA_ACQUISITON']['DATA_PATH'])
START_DATE_STR = params['DATA_ACQUISITON']['START_DATE']
END_DATE_STR = params['DATA_ACQUISITON']['END_DATE']

# Ensure output directory exists
output_dir = PATH_BARS / "Bars_1m_crypto"
os.makedirs(output_dir, exist_ok=True)

# --- BINANCE CLIENT (PUBLIC MODE) ---
client = Client()

symbol = "ETHUSDT"
interval = Client.KLINE_INTERVAL_1MINUTE

print(f"🚀 Fetching 1m bars for {symbol} (Public Mode)")
print(f"📅 Range: {START_DATE_STR} to {END_DATE_STR}")

# Convert strings to timestamps (ms)
start_ts = int(datetime.strptime(START_DATE_STR, "%Y-%m-%d").timestamp() * 1000)
end_ts = int(datetime.strptime(END_DATE_STR, "%Y-%m-%d").timestamp() * 1000)

klines_data = []
current_start = start_ts

# --- FETCH LOOP ---
while current_start < end_ts:
    try:
        candles = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=current_start,
            endTime=end_ts,
            limit=1000
        )

        if not candles:
            break

        klines_data.extend(candles)

        # Move start time
        last_close_time = candles[-1][6]
        current_start = last_close_time + 1

        # Progress Log
        last_date = datetime.fromtimestamp(candles[-1][0]/1000).strftime('%Y-%m-%d %H:%M:%S')
        print(f"   Fetched up to {last_date} | Total Rows: {len(klines_data)}")

        # Sleep to respect public rate limits
        time.sleep(0.1)

    except Exception as e:
        print(f"⚠️ Error during fetch: {e}")
        time.sleep(2)

# --- PROCESSING ---
print("-" * 30)
print(f"Processing {len(klines_data)} records into DataFrame...")

columns = [
    'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Close Time', 'Quote Asset Volume', 'Number of Trades',
    'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
]

df = pd.DataFrame(klines_data, columns=columns)

# Convert Time
df['timestamp'] = pd.to_datetime(df['Open Time'], unit='ms')

# Convert Numbers
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Number of Trades']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Clean Columns
df.drop(columns=['Open Time', 'Close Time', 'Ignore', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume'], inplace=True)

# Final Date Filter
start_dt = pd.to_datetime(START_DATE_STR)
end_dt = pd.to_datetime(END_DATE_STR)
mask = (df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)
df = df.loc[mask]

# --- SAVING ---
save_path = output_dir / f"{symbol}.parquet"
df.to_parquet(save_path, index=False)

print("-" * 30)
print(f"✅ SUCCESS! Final DataFrame size: {df.shape}")
print(f"Data saved to: {save_path}")