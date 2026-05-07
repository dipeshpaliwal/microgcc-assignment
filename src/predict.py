import joblib
import pandas as pd

from src.feature_engineering import create_features

model = joblib.load("saved_models/xgb_model.pkl")

def predict(data):
    df = pd.DataFrame(data)

    df = create_features(df)
    df = df.dropna()

    preds = model.predict(df)

    return preds