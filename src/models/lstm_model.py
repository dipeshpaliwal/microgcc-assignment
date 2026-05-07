import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from sklearn.metrics import mean_absolute_error


def train_lstm(X_train, y_train, X_test, y_test):

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_train = X_train.reshape(
        (X_train.shape[0], 1, X_train.shape[1])
    )

    X_test = X_test.reshape(
        (X_test.shape[0], 1, X_test.shape[1])
    )

    model = Sequential()

    model.add(
        LSTM(
            50,
            activation='relu',
            input_shape=(1, X_train.shape[2])
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    model.fit(
        X_train,
        y_train,
        epochs=5,
        verbose=0
    )

    preds = model.predict(X_test).flatten()

    mae = mean_absolute_error(
        y_test,
        preds
    )

    return mae, model