import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from tensorflow import keras

print("📦 Loading data...")
# Load data
df = pd.read_csv("retail_store_inventory.csv", parse_dates=["Date"])

# Sort data by date
df = df.sort_values("Date")

# Group by date and sum total demand across all stores/products
df_daily = df.groupby("Date")["Units Sold"].sum().reset_index()

print("🔄 Scaling and preparing data...")
# Normalize data
scaler = MinMaxScaler()
df_daily["Scaled_Demand"] = scaler.fit_transform(df_daily[["Units Sold"]])

# Prepare sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 30
data = df_daily["Scaled_Demand"].values
X, y = create_sequences(data, sequence_length)

# Reshape for LSTM [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("🧠 Building and training LSTM model...")
# Build LSTM model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

print("📈 Predicting and evaluating...")
# Predict
predictions = model.predict(X_test)

# Inverse scale
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
predictions_rescaled = scaler.inverse_transform(predictions)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
print(f"✅ Test RMSE: {rmse:.2f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test_rescaled, label='Actual Demand')
plt.plot(predictions_rescaled, label='Predicted Demand')
plt.legend()
plt.title("📊 LSTM Demand Forecasting")
plt.xlabel("Time")
plt.ylabel("Units Sold")
plt.show()
