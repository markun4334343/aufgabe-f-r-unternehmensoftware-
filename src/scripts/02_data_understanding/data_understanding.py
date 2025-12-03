"""
Script: Data Understanding and Exploration
Location: src/scripts/02_data_understanding/data_understanding.py
Description: Loads the acquired ETHUSDT data, calculates descriptive statistics,
             and generates relevant plots to understand the market history.
"""

import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION & PATH SETUP ---

# Assuming this script is at: .../src/scripts/02_data_understanding/data_understanding.py
# Use Pathlib to navigate up to the project 'src' folder (3 levels up)
base_dir = Path(__file__).parent.parent.parent
params_path = base_dir / "conf" / "params.yaml"

# --- TARGET IMAGE OUTPUT DIRECTORY ---
# The target is: C:\Users\rakun\Desktop\projket fur unternehmenssofware\etherium projekt\src\images
PLOTS_DIR = base_dir / "images"
os.makedirs(PLOTS_DIR, exist_ok=True) # Creates the directory if it doesn't exist

print(f"Loading config from: {params_path}")
try:
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
except FileNotFoundError:
    print(f"❌ ERROR: Configuration file not found at {params_path}")
    exit(1)

# Extract Data Paths
DATA_PATH = Path(params['DATA_ACQUISITON']['DATA_PATH'])
DATA_FILE_PATH = DATA_PATH / "Bars_1m_crypto" / "ETHUSDT.parquet"


# --- 1. DATA LOADING ---
print("-" * 50)
print(f"Attempting to load data from: {DATA_FILE_PATH}")
try:
    df = pd.read_parquet(DATA_FILE_PATH)
    print(f"✅ Data loaded successfully. Total records: {len(df)}")
except FileNotFoundError:
    print(f"❌ ERROR: Data file not found at {DATA_FILE_PATH}. Run data_loader.py first.")
    exit(1)

# Inspect the data structure
print("\n--- DataFrame Information ---")
df.info()


# --- 2. EXPLAIN RELEVANT DATA COLUMNS ---
print("\n--- Relevant Data Columns Explained ---")
print(
    """
    - timestamp: The start time of the 1-minute candlestick interval (UTC).
    - Open: The price at the beginning of the interval.
    - High: The highest price reached during the interval.
    - Low: The lowest price reached during the interval.
    - Close: The price at the end of the interval (the price used for trading decisions).
    - Volume: The amount of base asset (ETH) traded during the interval.
    - Quote Asset Volume: The amount of quote asset (USDT) traded during the interval.
    - Number of Trades: The count of individual transactions that occurred in the interval.
    """
)


# --- 3. DESCRIPTIVE STATISTICS ---
print("\n--- Descriptive Statistics for Price and Volume ---")
stats_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Number of Trades']
print(df[stats_cols].describe().to_markdown())


# --- 4. RELEVANT PLOTS OF VARIABLES (SAVING TO NEW PLOTS_DIR) ---

# Set plotting style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 6))

# Plot 1: Close Price Over Time (Time Series)
plt.plot(df['timestamp'], df['Close'], label='Close Price (USDT)', color='skyblue', linewidth=0.5)
plt.title(f'ETHUSDT Close Price Over Time ({df["timestamp"].min().date()} to {df["timestamp"].max().date()})')
plt.xlabel('Date')
plt.ylabel('Price (USDT)')
plt.legend()
plt.tight_layout()
price_plot_path = PLOTS_DIR / "close_price_time_series.png"
plt.savefig(price_plot_path)
plt.close()
print(f"📊 Close Price Time Series plot saved to: {price_plot_path}")

# Plot 2: Volume Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['Volume'], bins=100, kde=True, color='purple')
plt.title('Distribution of Trading Volume (ETH)')
plt.xlabel('Volume (ETH)')
plt.ylabel('Frequency (1-min bars)')
plt.tight_layout()
volume_plot_path = PLOTS_DIR / "volume_histogram.png"
plt.savefig(volume_plot_path)
plt.close()
print(f"📊 Volume Histogram plot saved to: {volume_plot_path}")

# --- 5. PRESENT FINDINGS (SUMMARY) ---
print("-" * 50)
print("\n--- Findings from Data Understanding ---")

# Price Range Finding
min_price = df['Low'].min()
max_price = df['High'].max()
print(f"1. **Price Range**: Over the observation period, ETH traded between a low of ${min_price:,.2f} and a high of ${max_price:,.2f}.")

# Volume Finding (using 99th percentile to characterize typical activity vs spikes)
volume_median = df['Volume'].median()
volume_p99 = df['Volume'].quantile(0.99)
print(f"2. **Typical Volume**: The median 1-minute trading volume is {volume_median:,.2f} ETH. However, the top 1% of trading bars have a volume exceeding {volume_p99:,.2f} ETH, indicating significant volume spikes.")

# Trade Count Finding
trades_mean = df['Number of Trades'].mean()
print(f"3. **Trade Frequency**: An average of {trades_mean:,.0f} trades occurred every minute.")

print("\nThese statistics and plots provide a foundational view of the ETHUSDT market necessary for building a robust trading model.")
print("-" * 50)