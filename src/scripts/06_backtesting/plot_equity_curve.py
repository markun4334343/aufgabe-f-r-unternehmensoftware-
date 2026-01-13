import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from pandas.plotting import register_matplotlib_converters

# Register converters
register_matplotlib_converters()

# --- CONFIG ---
DATA_PATH_STR = r"C:\Users\rakun\Desktop\projket fur unternehmenssofware\etherium projekt\src\Data\Bars_1m_crypto\ETHUSDT.parquet"
START_DATE = '2024-01-01'

# --- LOAD DATA ---
data_path = Path(DATA_PATH_STR)
print(f"📂 Loading data from: {data_path}")

if not data_path.exists():
    print("❌ ERROR: File not found!")
    exit()

df = pd.read_parquet(data_path)

# --- 🛠️ CRITICAL FIX: ENSURE DATETIME INDEX ---
# If 'timestamp' is a column, move it to the index
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
# If the index is already datetime but lost its type, force it
elif not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

# Filter for 2024 onwards
try:
    df = df.loc[START_DATE:]
except Exception as e:
    print(f"⚠️ Warning filtering date: {e}")

print(f"✅ Data Ready: {len(df)} rows")

# --- DOWNSAMPLE ---
# Now this will work because the index is definitely Datetime
df_plot = df.resample('4h').last().dropna()
print(f"📉 Downsampled to {len(df_plot)} points")

# --- SIMULATE EQUITY ---
initial_capital = 100000
df_plot['Equity'] = initial_capital * (df_plot['Close'] / df_plot['Close'].iloc[0])

# --- PLOT ---
plt.figure(figsize=(12, 6))
plt.plot(df_plot.index, df_plot['Equity'], color='#F0B90B', linewidth=2, label='Portfolio Value')
plt.fill_between(df_plot.index, df_plot['Equity'], initial_capital * 0.9, color='#FEF6D8', alpha=0.5)

plt.title(f"Simulated Portfolio Performance ({START_DATE} - Present)", fontsize=14, fontweight='bold', loc='left')
plt.ylabel("Portfolio Value ($)", fontsize=12)
plt.grid(True, which='major', axis='y', linestyle='--', alpha=0.3)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gcf().autofmt_xdate()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.legend(loc='upper left')
plt.tight_layout()

# Save
output_file = "my_portfolio_chart.png"
plt.savefig(output_file, dpi=300)
print(f"📸 Chart saved as: {output_file}")
plt.show()