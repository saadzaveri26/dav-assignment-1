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
# started with the data preprocessing
df.columns = df.columns.str.strip()  # remove hidden spaces first
print("\nActual column names:", df.columns.tolist())  # SEE EXACT NAMES

# Now rename based on actual NSE column names
df.rename(columns={
    "Date": "date",
    "DATE": "date", 
    " Date": "date",
    "CLOSE": "close",
    "Close": "close",
    "close price": "close"
}, inplace=True)

print("Columns after rename:", df.columns.tolist())  # confirm it worked

df["close"] = df["close"].astype(str).str.replace(",", "").astype(float)
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

# ARIMA model implementation

close_series = df.set_index("date")["close"]
 
 
# (a) ADF Test — Check for stationarity
print("\n--- ADF Test (Original Series) ---")
adf_result = adfuller(close_series)
print(f"ADF Statistic : {adf_result[0]:.4f}")
print(f"p-value       : {adf_result[1]:.4f}")
print(f"Critical Values:")
for key, val in adf_result[4].items():
    print(f"   {key}: {val:.4f}")
 
if adf_result[1] < 0.05:
    print(">> Series is STATIONARY (p < 0.05). No differencing needed.")
    d_value = 0
else:
    print(">> Series is NON-STATIONARY (p >= 0.05). Differencing required.")
    d_value = 1
 
    # Check again after first-order differencing
    diff_series = close_series.diff().dropna()
    adf_diff = adfuller(diff_series)
    print("\n--- ADF Test (After 1st-order Differencing) ---")
    print(f"ADF Statistic : {adf_diff[0]:.4f}")
    print(f"p-value       : {adf_diff[1]:.4f}")
    if adf_diff[1] < 0.05:
        print(">> Differenced series IS stationary. Using d=1.")
    else:
        print(">> Still non-stationary. May need d=2 (rare).")
 
 
# (b) ACF and PACF plots to determine p and q
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
 
plot_acf(close_series.diff().dropna(), lags=30, ax=axes[0], color="#e74c3c")
axes[0].set_title("ACF Plot (1st Difference)", fontweight="bold")
 
plot_pacf(close_series.diff().dropna(), lags=30, ax=axes[1], color="#2ecc71")
axes[1].set_title("PACF Plot (1st Difference)", fontweight="bold")
 
plt.tight_layout()
plt.savefig("output_2_acf_pacf.png", dpi=150)
plt.show()
print("Plot saved: output_2_acf_pacf.png")
 
# Based on typical ACF/PACF behaviour for stock data,
# p=1, d=1, q=1 is a common starting point (ARIMA(1,1,1))
# You can tune this further after looking at your plots.
p, d, q = 1, d_value, 1
print(f"\nUsing ARIMA({p},{d},{q}) — adjust based on your ACF/PACF plots if needed.")
 
 
# (c) Fit ARIMA model
# Split into train and test (last 30 days = test)
train = close_series[:-30]
test = close_series[-30:]
 
print(f"\nTraining on {len(train)} data points...")
model = ARIMA(train, order=(p, d, q))
fitted_model = model.fit()
 
print(fitted_model.summary())
 
# In-sample predictions on the training set
train_pred = fitted_model.fittedvalues
 
# Evaluate on test set using rolling forecast
predictions = []
history = list(train)
 
for t in range(len(test)):
    temp_model = ARIMA(history, order=(p, d, q))
    temp_fitted = temp_model.fit()
    yhat = temp_fitted.forecast(steps=1)[0]
    predictions.append(yhat)
    history.append(test.iloc[t])
 
predictions = pd.Series(predictions, index=test.index)
 
# Performance metrics
rmse = np.sqrt(mean_squared_error(test, predictions))
mae = mean_absolute_error(test, predictions)
mape = np.mean(np.abs((test.values - predictions.values) / test.values)) * 100
 
print(f"\n--- Model Performance on Test Set (Last 30 Days) ---")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"MAPE : {mape:.2f}%")

# Future Price prediction for next 30 days

# (a) Retrain on full data and forecast 30 days ahead
final_model = ARIMA(close_series, order=(p, d, q))
final_fitted = final_model.fit()
 
forecast_result = final_fitted.get_forecast(steps=30)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()
 
# Create future date index (only weekdays — markets closed on weekends)
last_date = close_series.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)
forecast_mean.index = future_dates
forecast_ci.index = future_dates
 
print("\nForecasted closing prices for next 30 trading days:")
print(forecast_mean.round(2).to_string())
 
 
# (b) Plot historical + forecasted prices
fig, ax = plt.subplots(figsize=(14, 6))
 
# Plot last 90 days of historical data for clarity
ax.plot(close_series[-90:], label="Historical Price", color="#1f77b4", linewidth=1.8)
 
# Plot forecast
ax.plot(forecast_mean, label="Forecasted Price (Next 30 Days)", color="#e74c3c",
        linewidth=2, linestyle="--", marker="o", markersize=4)
 
# Confidence interval shading
ax.fill_between(future_dates,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color="#e74c3c", alpha=0.15, label="95% Confidence Interval")
 
ax.axvline(x=close_series.index[-1], color="gray", linestyle=":", linewidth=1.5,
           label="Forecast Start")
 
ax.set_title("HEROMOTOCO — Historical vs Forecasted Closing Price", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price (INR)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output_3_forecast.png", dpi=150)
plt.show()
print("Plot saved: output_3_forecast.png")

#future prediction for next 30 days

final_model = ARIMA(close_series, order=(p, d, q))
final_fitted = final_model.fit()
 
forecast_result = final_fitted.get_forecast(steps=30)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()
 
# Create future date index (only weekdays — markets closed on weekends)
last_date = close_series.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)
forecast_mean.index = future_dates
forecast_ci.index = future_dates
 
print("\nForecasted closing prices for next 30 trading days:")
print(forecast_mean.round(2).to_string())
 
 
# (b) Plot historical + forecasted prices
fig, ax = plt.subplots(figsize=(14, 6))
 
# Plot last 90 days of historical data for clarity
ax.plot(close_series[-90:], label="Historical Price", color="#1f77b4", linewidth=1.8)
 
# Plot forecast
ax.plot(forecast_mean, label="Forecasted Price (Next 30 Days)", color="#e74c3c",
        linewidth=2, linestyle="--", marker="o", markersize=4)
 
# Confidence interval shading
ax.fill_between(future_dates,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color="#e74c3c", alpha=0.15, label="95% Confidence Interval")
 
ax.axvline(x=close_series.index[-1], color="gray", linestyle=":", linewidth=1.5,
           label="Forecast Start")
 
ax.set_title("HEROMOTOCO — Historical vs Forecasted Closing Price", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price (INR)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output_3_forecast.png", dpi=150)
plt.show()
print("Plot saved: output_3_forecast.png")