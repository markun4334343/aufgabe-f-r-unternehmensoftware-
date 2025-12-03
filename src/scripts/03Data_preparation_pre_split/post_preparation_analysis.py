"""
Script: Post-Preparation Analysis and Visualization
Location: src/scripts/03_data_preparation/post_preparation_analysis.py (Example location)
Description: Loads the preprocessed data, plots sample features, and analyzes target distribution.
"""

import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Suppress harmless Matplotlib warning
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION & PATH SETUP ---
# Path logic adjusted for navigating from a subfolder inside 'scripts' to 'src'
base_dir = Path(__file__).parent.parent.parent
params_path = base_dir / "conf" / "params.yaml"

# Target Directories
DATA_PATH = Path(yaml.safe_load(open(params_path))['DATA_ACQUISITON']['DATA_PATH'])
INPUT_FILE_PATH = DATA_PATH / "ETHUSDT_preprocessed.parquet"
PLOTS_DIR = base_dir / "images"
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- 1. DATA LOADING ---
print("-" * 50)
print(f"Attempting to load preprocessed data from: {INPUT_FILE_PATH}")
try:
    df = pd.read_parquet(INPUT_FILE_PATH)
    print(f"✅ Data loaded successfully. Total records: {len(df)}")
except FileNotFoundError:
    print(f"❌ ERROR: Preprocessed file not found. Run data_preparation.py first.")
    exit(1)


# --- 2. VISUALIZATION OF FEATURES ---

# Focus on a short, relevant time window for clarity (e.g., the last 30 days)
SAMPLE_WINDOW = '30D'
df_sample = df.last(SAMPLE_WINDOW)

print("\n--- Generating Feature Plots ---")
sns.set_style("whitegrid")

# Plot 1: Price vs. Moving Averages
plt.figure(figsize=(15, 7))
plt.plot(df_sample.index, df_sample['Close'], label='Close Price', color='black', linewidth=1)
plt.plot(df_sample.index, df_sample['MA_60'], label='MA 60 min', color='blue', linestyle='--', linewidth=1)
plt.plot(df_sample.index, df_sample['MA_240'], label='MA 240 min', color='red', linestyle='--', linewidth=1)
plt.title(f'Close Price vs. Moving Averages (Sampled {SAMPLE_WINDOW})')
plt.xlabel('Timestamp')
plt.ylabel('Price (USDT)')
plt.legend()
plt.tight_layout()
feature_plot_path = PLOTS_DIR / "feature_price_vs_ma.png"
plt.savefig(feature_plot_path)
plt.close()
print(f"📊 1. Price vs MA Plot saved to: {feature_plot_path}")


# Plot 2: Volatility Features
plt.figure(figsize=(15, 5))
plt.plot(df_sample.index, df_sample['Vol_60'], label='Volatility 60 min', color='green', linewidth=1)
plt.plot(df_sample.index, df_sample['Vol_720'], label='Volatility 720 min', color='orange', linewidth=1)
plt.title(f'Rolling Volatility Features (Sampled {SAMPLE_WINDOW})')
plt.xlabel('Timestamp')
plt.ylabel('Standard Deviation of Log Returns')
plt.legend()
plt.tight_layout()
volatility_plot_path = PLOTS_DIR / "feature_volatility.png"
plt.savefig(volatility_plot_path)
plt.close()
print(f"📊 2. Volatility Plot saved to: {volatility_plot_path}")


# --- 3. TARGET VARIABLE ANALYSIS ---

# Analyze the distribution of the target variables (we'll focus on the 60-minute target)
target_col = 'target_60m'

# Map the numerical targets to descriptive labels
target_map = {1: 'Up (+1)', -1: 'Down (-1)', 0: 'Flat (0)'}
df[target_col + '_label'] = df[target_col].map(target_map)

print("\n--- Generating Target Distribution Plot ---")

# Plot 3: Target Distribution (Class Imbalance Check)
plt.figure(figsize=(8, 6))
sns.countplot(x=target_col + '_label', data=df, palette='viridis')
plt.title(f'Distribution of Target Variable: {target_col} (60 min horizon)')
plt.xlabel('Movement Class')
plt.ylabel('Count of 1-minute Bars')
plt.tight_layout()
target_plot_path = PLOTS_DIR / "target_distribution_60m.png"
plt.savefig(target_plot_path)
plt.close()
print(f"📊 3. Target Distribution Plot saved to: {target_plot_path}")


# Print quantitative distribution
target_counts = df[target_col].value_counts(normalize=True).mul(100).round(2)
print("\n--- Quantitative Target Distribution (Percentage) ---")
print(target_counts.rename(index=target_map).to_markdown())

print("-" * 50)
print("✅ Post-preparation analysis complete. Check the 'src/images' folder for plots.")