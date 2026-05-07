from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def train_xgb(X_train, y_train, X_test, y_test):

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    return mae, model