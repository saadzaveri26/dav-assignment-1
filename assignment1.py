# downloading the required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")
 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# loading the dataset
df = pd.read_csv("HEROMOTOCO.csv")

print(df.head())

print("\nColumn names:", df.columns.tolist())

# started with the data preprocessing

df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
 

df = df.sort_values("date").reset_index(drop=True)
 

print("\nMissing values before handling:")
print(df.isnull().sum())
 

df["close"] = df["close"].fillna(method="ffill")
 

df.dropna(subset=["close"], inplace=True)
 
print("\nMissing values after handling:")
print(df.isnull().sum())
 
print(f"\nData spans from {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total trading days: {len(df)}")
 
 

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["date"], df["close"], color="#1f77b4", linewidth=1.5)
ax.set_title("HEROMOTOCO — Daily Closing Price (Past 1 Year)", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price (INR)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output_1_closing_price_trend.png", dpi=150)
plt.show()
print("Plot saved: output_1_closing_price_trend.png")