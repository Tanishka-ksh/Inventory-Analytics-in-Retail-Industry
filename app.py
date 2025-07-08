import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Title
st.title("📦 Inventory Analytics - Retail Industry Dashboard")

# Load data
df = pd.read_csv("retail_store_inventory.csv", parse_dates=["Date"])
df = df.sort_values("Date")

# Sidebar filters
st.sidebar.header("Filter Options")
category_filter = st.sidebar.selectbox("Select Category", options=["All"] + sorted(df["Category"].dropna().unique().tolist()))
product_filter = st.sidebar.selectbox("Select Product ID", options=["All"] + sorted(df["Product ID"].dropna().unique().tolist()))
region_filter = st.sidebar.selectbox("Select Region", options=["All"] + sorted(df["Region"].dropna().unique().tolist()))

# Apply filters
filtered_df = df.copy()
if category_filter != "All":
    filtered_df = filtered_df[filtered_df["Category"] == category_filter]
if product_filter != "All":
    filtered_df = filtered_df[filtered_df["Product ID"] == product_filter]
if region_filter != "All":
    filtered_df = filtered_df[filtered_df["Region"] == region_filter]

# Preprocessing for multivariate LSTM
df_encoded = pd.get_dummies(filtered_df, columns=["Weather Condition", "Seasonality"])
feature_cols = ["Units Sold", "Demand Forecast", "Price", "Discount", "Competitor Pricing", "Holiday/Promotion"]
one_hot_cols = [col for col in df_encoded.columns if "Weather Condition_" in col or "Seasonality_" in col]

selected_columns = feature_cols + one_hot_cols
df_daily_multi = df_encoded.groupby("Date")[selected_columns].mean().dropna()

if len(df_daily_multi) < 60:
    st.warning("Not enough data for time series forecasting after applying filters. Please choose another filter.")
    st.stop()

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_daily_multi)

# Create multivariate sequences
sequence_length = 30
X, y = [], []
for i in range(len(scaled_features) - sequence_length):
    X.append(scaled_features[i:i+sequence_length])
    y.append(scaled_features[i+sequence_length][0])  # Predict Units Sold

X = np.array(X)
y = np.array(y)

# Train-test split
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

@st.cache_data
def load_data():
    df = pd.read_csv("retail_store_inventory.csv", parse_dates=["Date"])
    return df.sort_values("Date")

df = load_data()


# --- LSTM Model ---
model = keras.models.Sequential()
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(32))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
with st.spinner("Training LSTM Model..."):
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)


# Predict
lstm_predictions = model.predict(X_test)
lstm_predictions_rescaled = lstm_predictions * scaler.data_range_[0] + scaler.data_min_[0]
y_test_rescaled = y_test * scaler.data_range_[0] + scaler.data_min_[0]

# --- ARIMA Forecasting ---
df_arima = df_daily_multi["Units Sold"]
train_arima = df_arima.iloc[:split_idx + sequence_length]
test_arima = df_arima.iloc[split_idx + sequence_length:]
model_arima = ARIMA(train_arima, order=(5, 1, 0))
model_arima_fit = model_arima.fit()
arima_predictions = model_arima_fit.forecast(steps=len(test_arima))

# --- Evaluation ---
lstm_rmse = np.sqrt(mean_squared_error(y_test_rescaled, lstm_predictions_rescaled))
arima_rmse = np.sqrt(mean_squared_error(test_arima, arima_predictions))

# MAPE and MAE calculation for LSTM and ARIMA
lstm_mape = np.mean(np.abs((y_test_rescaled - lstm_predictions_rescaled) / y_test_rescaled)) * 100
lstm_mae = mean_absolute_error(y_test_rescaled, lstm_predictions_rescaled)

arima_mape = np.mean(np.abs((test_arima.values - arima_predictions) / test_arima.values)) * 100
arima_mae = mean_absolute_error(test_arima, arima_predictions)

# Display Model Evaluation
st.subheader("📊 Model Evaluation")
st.markdown(f"**ARIMA RMSE:** {arima_rmse:.2f} | **MAPE:** {arima_mape:.2f}% | **MAE:** {arima_mae:.2f}")
st.markdown(f"**LSTM RMSE:** {lstm_rmse:.2f} | **MAPE:** {lstm_mape:.2f}% | **MAE:** {lstm_mae:.2f}")

# --- Visualization ---
model_choice = st.radio("Select Model(s) to Visualize:", ("Both", "ARIMA Only", "LSTM Only"), horizontal=True)

st.subheader("🔍 Forecast Comparison")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test_arima.index, test_arima.values, label='Actual Demand', color='black')
if model_choice in ("Both", "LSTM Only"):
    ax.plot(test_arima.index, lstm_predictions_rescaled, label='LSTM', linestyle='--')
if model_choice in ("Both", "ARIMA Only"):
    ax.plot(test_arima.index, arima_predictions, label='ARIMA', linestyle=':')
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend()
ax.set_title("LSTM vs ARIMA Demand Forecasting")
st.pyplot(fig)

st.caption("Use the sidebar to filter by Category, Product ID, or Region.")

# --- Inventory Optimization ---
st.subheader("📦 Inventory Optimization Status")

def compute_inventory_status(row):
    avg_daily_demand = row['Units Sold'] / 30
    reorder_point = avg_daily_demand * 5 + 20
    overstock_threshold = 500 * 0.8
    if row['Inventory Level'] < reorder_point:
        return '🔴 Stockout Risk'
    elif row['Inventory Level'] > overstock_threshold:
        return '🟡 Overstock'
    else:
        return '🔵 Optimal'

latest_inventory = filtered_df.sort_values("Date", ascending=False).drop_duplicates(["Product ID", "Region"])
latest_inventory['Inventory Status'] = latest_inventory.apply(compute_inventory_status, axis=1)

with st.expander("🔍 View Inventory Status Table"):
    st.dataframe(latest_inventory[["Product ID", "Region", "Category", "Inventory Level", "Units Sold", "Inventory Status"]].reset_index(drop=True))

# --- Visualizations ---
st.subheader("📊 Inventory Optimization Visuals")

# Pie Chart
st.markdown("**Pie Chart:** Distribution of Inventory Status")
status_counts = latest_inventory['Inventory Status'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
st.pyplot(fig1)

# Bar Chart: Overstock vs Stockouts
st.markdown("**Bar Chart:** Category-wise Inventory Status")
category_status = latest_inventory.groupby(['Category', 'Inventory Status']).size().unstack().fillna(0)
st.bar_chart(category_status)

# Turnover Rate
st.markdown("**Bar Chart:** Turnover Rate by Region")
turnover = filtered_df.groupby('Region').apply(lambda x: x['Units Sold'].sum() / x['Inventory Level'].replace(0, np.nan).mean()).dropna()
st.bar_chart(turnover)

# Line Chart: Units Sold vs Inventory
st.markdown("**Line Chart:** Units Sold vs Inventory Level Over Time")

st.line_chart(filtered_df.groupby('Date')[['Units Sold', 'Inventory Level']].sum())

# Heatmap
heatmap_data = filtered_df.pivot_table(index='Date', columns='Product ID', values='Inventory Level', aggfunc='mean').fillna(0)
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.heatmap(heatmap_data.T, cmap="coolwarm", cbar_kws={'label': 'Inventory Level'})
ax2.set_title("📅 Inventory Heatmap (Products vs. Date)")
st.pyplot(fig2)

st.subheader("💸 Dynamic Pricing Suggestions")

# Define thresholds
high_demand_threshold = filtered_df["Units Sold"].quantile(0.75)
low_demand_threshold = filtered_df["Units Sold"].quantile(0.25)
high_inventory_threshold = filtered_df["Inventory Level"].quantile(0.75)
low_inventory_threshold = filtered_df["Inventory Level"].quantile(0.25)

# Prepare latest values
latest = filtered_df.sort_values("Date", ascending=False).drop_duplicates(["Product ID", "Region"])
suggestions = []

for _, row in latest.iterrows():
    current_price = row["Price"]
    demand = row["Units Sold"]
    inventory = row["Inventory Level"]
    promo = row["Holiday/Promotion"]
    
    suggested_price = current_price
    reason = "No change"

    if demand > high_demand_threshold and inventory < low_inventory_threshold:
        suggested_price *= 1.10
        reason = "High demand & low inventory"
    elif demand < low_demand_threshold and inventory > high_inventory_threshold:
        suggested_price *= 0.90
        reason = "Low demand & high inventory"
    elif promo == 1:
        suggested_price *= 0.95
        reason = "Ongoing promotion"

    suggestions.append({
        "Product ID": row["Product ID"],
        "Region": row["Region"],
        "Current Price": round(current_price, 2),
        "Suggested Price": round(suggested_price, 2),
        "% Change": f"{((suggested_price - current_price) / current_price * 100):.1f}%",
        "Reason": reason
    })

# Display
suggestions_df = pd.DataFrame(suggestions)
st.dataframe(suggestions_df)
st.subheader("📊 Suggested vs Current Prices (per Product)")

fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
sampled = suggestions_df.head(10)  # Avoid clutter, show top 10
x = np.arange(len(sampled))

ax_bar.bar(x - 0.2, sampled["Current Price"], width=0.4, label="Current Price", color='skyblue')
ax_bar.bar(x + 0.2, sampled["Suggested Price"], width=0.4, label="Suggested Price", color='orange')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(sampled["Product ID"], rotation=45)
ax_bar.set_ylabel("Price")
ax_bar.set_title("Current vs Suggested Prices")
ax_bar.legend()
st.pyplot(fig_bar)
st.subheader("📈 Price Trend Over Time")

product_ids = filtered_df["Product ID"].unique().tolist()
selected_pid = st.selectbox("Choose Product ID", product_ids)

product_df = filtered_df[filtered_df["Product ID"] == selected_pid].sort_values("Date")
fig_line, ax_line = plt.subplots(figsize=(10, 4))
ax_line.plot(product_df["Date"], product_df["Price"], label="Historical Price", color='blue')
ax_line.set_title(f"Price Trend for Product {selected_pid}")
ax_line.set_ylabel("Price")
ax_line.set_xlabel("Date")
ax_line.legend()
st.pyplot(fig_line)
st.subheader("🥧 Reasons for Suggested Price Changes")

reason_counts = suggestions_df["Reason"].value_counts()

fig_reason_pie, ax_reason_pie = plt.subplots()
ax_reason_pie.pie(
    reason_counts,
    labels=reason_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=sns.color_palette("pastel")[0:len(reason_counts)]
)
ax_reason_pie.axis("equal")
st.pyplot(fig_reason_pie)
import plotly.express as px

# Merge Category into suggestions_df
latest_category_info = latest[["Product ID", "Category"]]
suggestions_df = suggestions_df.merge(latest_category_info, on="Product ID", how="left")

# Calculate Price Change % numerically
st.subheader("📊 Price Change Distribution by Category")
suggestions_df["Price Change %"] = (suggestions_df["Suggested Price"] - suggestions_df["Current Price"]) / suggestions_df["Current Price"] * 100
fig_box, ax_box = plt.subplots(figsize=(10, 4))
sns.boxplot(data=suggestions_df, x="Category", y="Price Change %", ax=ax_box)
ax_box.tick_params(axis='x', rotation=45)
st.pyplot(fig_box)

# ----- Compute Inventory Insights -----
inventory_counts = latest_inventory['Inventory Status'].value_counts(normalize=True) * 100
optimal_inventory_pct = round(inventory_counts.get("🔵 Optimal", 0), 1)
stockout_risk_pct = round(inventory_counts.get("🔴 Stockout Risk", 0), 1)
overstock_pct = round(inventory_counts.get("🟡 Overstock", 0), 1)

# Find category and region with most stockouts
stockout_df = latest_inventory[latest_inventory['Inventory Status'] == "🟡 Overstock"]
problematic_category = stockout_df['Category'].value_counts().idxmax() if not stockout_df.empty else "N/A"
problematic_region = stockout_df['Region'].value_counts().idxmax() if not stockout_df.empty else "N/A"

# ----- Compute Pricing Insights -----
suggestions_df['Price Change %'] = ((suggestions_df['Suggested Price'] - suggestions_df['Current Price']) / suggestions_df['Current Price']) * 100

avg_price_change = suggestions_df['Price Change %'].mean()

# ----- Display Summary Insights -----
st.subheader("🔍 Summary Insights")

col1, col2, col3 = st.columns(3)
col1.metric("🟢 Optimal Inventory %", f"{optimal_inventory_pct}%")
col2.metric("🔴 Stockout Risk %", f"{stockout_risk_pct}%")
col3.metric("🟡 Overstock %", f"{overstock_pct}%")

st.markdown(f"- 📉 **Category with Most Overstocking:** {problematic_category}")
st.markdown(f"- 🌍 **Region with Most Overstocking:** {problematic_region}")
st.markdown(f"- 💰 **Average Suggested Price Change:** {avg_price_change:.2f}%")