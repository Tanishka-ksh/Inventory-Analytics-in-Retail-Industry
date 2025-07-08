import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("retail_store_inventory.csv", parse_dates=["Date"])
df = df.sort_values("Date")

# Filter data
category_filter = "All"  # Replace with your choice of category
product_filter = "All"  # Replace with your choice of product
region_filter = "All"  # Replace with your choice of region

filtered_df = df.copy()
if category_filter != "All":
    filtered_df = filtered_df[filtered_df["Category"] == category_filter]
if product_filter != "All":
    filtered_df = filtered_df[filtered_df["Product ID"] == product_filter]
if region_filter != "All":
    filtered_df = filtered_df[filtered_df["Region"] == region_filter]

# Group data by date
df_daily = filtered_df.groupby("Date")["Units Sold"].sum().reset_index()

if len(df_daily) < 60:
    print("Not enough data for time series forecasting after applying filters.")
    exit()

# --- LSTM Forecasting ---
full_demand = df_daily["Units Sold"].values
train_size = int(0.8 * len(full_demand))
train_data = full_demand[:train_size]
test_data = full_demand[train_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
test_scaled = scaler.transform(test_data.reshape(-1, 1))

# Sequence creator
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 30
X_train, y_train = create_sequences(train_scaled, sequence_length)
X_test, y_test = create_sequences(test_scaled, sequence_length)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# LSTM Model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(32))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

# Predictions
lstm_predictions = model.predict(X_test)
lstm_predictions_rescaled = scaler.inverse_transform(lstm_predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# --- ARIMA Forecasting ---
df_arima = df_daily.set_index("Date")["Units Sold"]
train_arima = df_arima.iloc[:-len(y_test)]
test_arima = df_arima.iloc[-len(y_test):]
model_arima = ARIMA(train_arima, order=(5, 1, 0))
model_arima_fit = model_arima.fit()
arima_predictions = model_arima_fit.forecast(steps=len(test_arima))

# --- Evaluation ---
lstm_rmse = np.sqrt(mean_squared_error(y_test_rescaled, lstm_predictions_rescaled))
arima_rmse = np.sqrt(mean_squared_error(test_arima, arima_predictions))

# Print the evaluation metrics to terminal
print(f"ARIMA RMSE: {arima_rmse:.2f}")
print(f"LSTM RMSE: {lstm_rmse:.2f}")

# --- Inventory Optimization ---
def compute_inventory_status(row):
    avg_daily_demand = row['Units Sold'] / 30
    reorder_point = avg_daily_demand * 5 + 20
    overstock_threshold = 500 * 0.8
    if row['Inventory Level'] < reorder_point:
        return 'Stockout Risk'
    elif row['Inventory Level'] > overstock_threshold:
        return 'Overstock'
    else:
        return 'Optimal'

latest_inventory = filtered_df.sort_values("Date", ascending=False).drop_duplicates(["Product ID", "Region"])
latest_inventory['Inventory Status'] = latest_inventory.apply(compute_inventory_status, axis=1)

# Print Inventory Status to terminal
print("\nInventory Status (Top Products):")
print(latest_inventory[["Product ID", "Region", "Category", "Inventory Level", "Units Sold", "Inventory Status"]].head())

# Calculate Inventory Metrics
stockout_incidents = latest_inventory[latest_inventory['Inventory Status'] == 'Stockout Risk'].shape[0]
overstock_rate = latest_inventory[latest_inventory['Inventory Status'] == 'Overstock'].shape[0] / len(latest_inventory) * 100
inventory_turnover = latest_inventory['Units Sold'].sum() / latest_inventory['Inventory Level'].replace(0, np.nan).mean()

# Print inventory metrics to terminal
print(f"\nInventory Metrics:")
print(f"Stockout Incidents per Month: {stockout_incidents}")
print(f"Overstock Rate: {overstock_rate:.2f}%")
print(f"Inventory Turnover Ratio: {inventory_turnover:.2f}")
