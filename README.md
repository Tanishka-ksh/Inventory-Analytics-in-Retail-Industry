# 📦 Inventory Analytics in Retail Industry

This project explores intelligent inventory management and demand forecasting in the retail sector. It compares classical ARIMA models with deep learning-based LSTM models to forecast product demand and pricing trends, followed by a rule-based inventory optimization and dynamic pricing strategy.

## 📊 Problem Statement
Retail businesses often struggle with demand fluctuations, stockouts, and overstock. This project addresses:
- Product-level demand forecasting
- Inventory replenishment
- Dynamic pricing
- Real-time decision-making via dashboards

## 🧠 Key Technologies
- ARIMA (Statistical Time Series Forecasting)
- LSTM (Long Short-Term Memory Neural Networks)
- Rule-Based Inventory Optimization
- Dynamic Pricing Logic
- Python (pandas, statsmodels, keras, matplotlib)
- Data Visualization Dashboard

## 📌 Project Contributions
- 📈 **Demand Forecasting:** Compared ARIMA and LSTM, where LSTM outperformed with RMSE: `10.51` vs ARIMA’s `11.71`.
- 💰 **Price Prediction:** LSTM used to forecast price based on historical pricing and promotional patterns.
- 🧮 **Inventory Optimization:** Reorder points and thresholds defined using forecasted demand.
- 💡 **Dynamic Pricing:** Adjusted pricing based on demand-inventory balance.
- 📊 **Dashboard:** Interactive UI showcasing inventory levels, price recommendations, and forecast insights.

## 📂 Dataset Overview
- Historical sales by product and region
- Weather and promotion metadata
- Product and store features

## 🧪 Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| ARIMA | 11.71 | 9.31 | 6.65% |
| LSTM  | 10.51 | 8.50 | 6.35% |

## 🚀 Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/retail-inventory-forecasting.git
   cd retail-inventory-forecasting
