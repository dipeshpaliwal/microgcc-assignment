import joblib
import pandas as pd

from src.data_processing import load_and_clean_data
from src.feature_engineering import create_features

from src.models.xgboost_model import train_xgb
from src.models.arima_model import train_arima
from src.models.prophet_model import train_prophet
from src.models.lstm_model import train_lstm
from src.models.model_selector import select_best_model

# Load data
df = load_and_clean_data("data/sales.xlsx")

# Feature engineering
df = create_features(df)
df = df.dropna()

# Features
features = [
    'lag_1','lag_7','lag_30',
    'rolling_mean_7','rolling_std_7',
    'day_of_week','month','week_of_year','is_weekend'
]

target = 'Total'

# Split
split_date = df['Date'].max() - pd.Timedelta(weeks=8)

train = df[df['Date'] < split_date]
test = df[df['Date'] >= split_date]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

results = {}

# Train models
results['XGBoost'], xgb_model = train_xgb(X_train, y_train, X_test, y_test)
results['ARIMA'], arima_model = train_arima(y_train, y_test)
results['Prophet'], prophet_model = train_prophet(train, test)
results['LSTM'], lstm_model = train_lstm(X_train, y_train, X_test, y_test)

# Select best
best_model_name = min(results, key=results.get)
print("Best Model:", best_model_name)

# Save best model
if best_model_name == "XGBoost":
    joblib.dump(xgb_model, "saved_models/best_model.pkl")

elif best_model_name == "ARIMA":
    joblib.dump(arima_model, "saved_models/best_model.pkl")

elif best_model_name == "Prophet":
    joblib.dump(prophet_model, "saved_models/best_model.pkl")

elif best_model_name == "LSTM":
    joblib.dump(lstm_model, "saved_models/best_model.pkl")

print("Best model saved!")