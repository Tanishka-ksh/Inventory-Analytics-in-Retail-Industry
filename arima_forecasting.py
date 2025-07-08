import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Load data
print("📦 Loading dataset...")
df = pd.read_csv("retail_store_inventory.csv", parse_dates=["Date"])
df = df.sort_values("Date")

# Group by date and sum total demand
df_daily = df.groupby("Date")["Units Sold"].sum().reset_index()

# Normalize the data for fair comparison with LSTM
scaler = MinMaxScaler()
df_daily["Scaled_Demand"] = scaler.fit_transform(df_daily[["Units Sold"]])

# Train/test split
train_size = int(0.8 * len(df_daily))
train, test = df_daily["Scaled_Demand"][:train_size], df_daily["Scaled_Demand"][train_size:]

# Fit ARIMA model
print("🔁 Fitting ARIMA model...")
model = ARIMA(train, order=(5, 1, 0))  # You can tune this order
model_fit = model.fit()

# Forecast
print("📈 Forecasting future demand...")
forecast = model_fit.forecast(steps=len(test))
forecast = np.array(forecast)

# Inverse scale
y_test_rescaled = scaler.inverse_transform(test.values.reshape(-1, 1))
forecast_rescaled = scaler.inverse_transform(forecast.reshape(-1, 1))

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test_rescaled, forecast_rescaled))
print(f"✅ ARIMA Test RMSE: {rmse:.2f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test_rescaled, label='Actual Demand')
plt.plot(forecast_rescaled, label='ARIMA Forecast')
plt.legend()
plt.title("ARIMA Demand Forecasting")
plt.xlabel("Time")
plt.ylabel("Units Sold")
plt.show()
