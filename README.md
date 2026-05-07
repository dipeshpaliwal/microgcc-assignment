# End-to-End Time Series Forecasting System with API

## 📌 Project Overview

This project is a production-ready time series forecasting system developed for forecasting the next 8 weeks of sales for each state using historical sales data.

The system:
- Trains multiple forecasting models
- Compares model performance
- Automatically selects the best model
- Exposes predictions using a FastAPI REST API
- Uses proper time-series feature engineering and validation logic

---

# 🚀 Features

## Forecasting Models Implemented

1. ARIMA / SARIMA
2. Facebook Prophet
3. XGBoost Regressor
4. LSTM (Deep Learning)

---

# 📊 Feature Engineering

### Lag Features
- lag_1
- lag_7
- lag_30

### Rolling Features
- rolling_mean_7
- rolling_std_7

### Date Features
- Day of week
- Month
- Week of year
- Weekend flag

---

# 📂 Project Structure

```bash
microgcc-assignment/
│
├── api/
│   └── main.py
│
├── data/
│   └── Copy of Forecasting Case- Study.xlsx
│
├── saved_models/
│   └── best_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   │
│   └── models/
│       ├── __init__.py
│       ├── arima_model.py
│       ├── prophet_model.py
│       ├── xgboost_model.py
│       ├── lstm_model.py
│       └── model_selector.py
│
├── requirements.txt
└── README.md
