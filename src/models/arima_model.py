from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error


def train_arima(y_train, y_test):

    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7)
    )

    model_fit = model.fit(disp=False)

    preds = model_fit.forecast(steps=len(y_test))

    mae = mean_absolute_error(y_test, preds)

    return mae, model_fit