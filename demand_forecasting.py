import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Smart Retail Forecasting Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("retail_store_inventory.csv", parse_dates=["Date"])
    return df

df = load_data()

st.title("📈 Smart Retail Demand Forecasting")
st.markdown("Select filters to forecast demand using LSTM & ARIMA.")

# Sidebar Filters
col1, col2 = st.columns(2)
with col1:
    selected_category = st.selectbox("Select Category:", ["All"] + sorted(df["Category"].dropna().unique().tolist()))
with col2:
    selected_product = st.selectbox("Select Product ID:", ["All"] + sorted(df["Product ID"].dropna().unique().tolist()))

# Filter Data Based on Selection
filtered_df = df.copy()
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]
if selected_product != "All":
    filtered_df = filtered_df[filtered_df["Product ID"] == selected_product]

if filtered_df.empty:
    st.warning("No data found for the selected filters.")
else:
    # Aggregate demand per day
    df_daily = filtered_df.groupby("Date")["Units Sold"].sum().reset_index().sort_values("Date")

    if len(df_daily) < 50:
        st.warning("Not enough data points for forecasting (need at least 50 days).")
    else:
        # Preprocessing
        scaler = MinMaxScaler()
        df_daily["Scaled_Demand"] = scaler.fit_transform(df_daily[["Units Sold"]])
        
        # Prepare LSTM sequences
        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i+seq_len])
                y.append(data[i+seq_len])
            return np.array(X), np.array(y)

        sequence_length = 30
        series = df_daily["Scaled_Demand"].values
        X, y = create_sequences(series, sequence_length)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # LSTM Model
        model = keras.models.Sequential([
            keras.layers.LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

        # Predictions
        lstm_predictions = model.predict(X_test)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        lstm_pred_rescaled = scaler.inverse_transform(lstm_predictions)
        lstm_rmse = np.sqrt(mean_squared_error(y_test_rescaled, lstm_pred_rescaled))

        # ARIMA
        demand_series = df_daily.set_index("Date")["Units Sold"]
        arima_model = ARIMA(demand_series, order=(5, 1, 0)).fit()
        arima_forecast = arima_model.forecast(steps=len(y_test))
        arima_rmse = np.sqrt(mean_squared_error(y_test_rescaled, arima_forecast))

        # Plot
        st.subheader("📊 Demand Forecast Comparison")
        fig, ax = plt.subplots(figsize=(12, 5))
        test_dates = df_daily["Date"][len(df_daily) - len(y_test):]

        ax.plot(test_dates, y_test_rescaled, label="Actual", color="black")
        ax.plot(test_dates, lstm_pred_rescaled, label=f"LSTM (RMSE: {lstm_rmse:.2f})", linestyle="--", color="blue")
        ax.plot(test_dates, arima_forecast, label=f"ARIMA (RMSE: {arima_rmse:.2f})", linestyle=":", color="green")

        ax.set_xlabel("Date")
        ax.set_ylabel("Units Sold")
        ax.set_title("Demand Forecasting - LSTM vs ARIMA")
        ax.legend()
        st.pyplot(fig)

        st.success("Forecasting completed successfully!")
