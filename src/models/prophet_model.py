from prophet import Prophet
from sklearn.metrics import mean_absolute_error


def train_prophet(train_df, test_df):

    prophet_train = train_df[['Date', 'Total']].rename(
        columns={
            'Date': 'ds',
            'Total': 'y'
        }
    )

    model = Prophet()

    model.fit(prophet_train)

    future = model.make_future_dataframe(
        periods=len(test_df)
    )

    forecast = model.predict(future)

    preds = forecast['yhat'][-len(test_df):].values

    mae = mean_absolute_error(
        test_df['Total'],
        preds
    )

    return mae, model