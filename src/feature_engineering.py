def create_features(df):
    df = df.sort_values(['State', 'Date'])

    # Lag features
    df['lag_1'] = df.groupby('State')['Total'].shift(1)
    df['lag_7'] = df.groupby('State')['Total'].shift(7)
    df['lag_30'] = df.groupby('State')['Total'].shift(30)

    # Rolling
    df['rolling_mean_7'] = df.groupby('State')['Total'].shift(1).rolling(7).mean()
    df['rolling_std_7'] = df.groupby('State')['Total'].shift(1).rolling(7).std()

    # Date features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    return df