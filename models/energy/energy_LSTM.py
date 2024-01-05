import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, TimeDistributed
from keras.models import Model
from datetime import timedelta, datetime, date
from sklearn.preprocessing import MinMaxScaler


def build_and_forecast_lstm(df, date_str):
    # Constants
    if date_str==None:
        date_str = date.today()
    start_date = datetime.strptime(date_str, '%Y-%m-%d')
    feature_columns = ['hour', 'day', 'month', 'weekday']
    horizons = [36, 40, 44, 60, 64, 68]
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

    # Extracting temporal features from DataFrame
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday

    # Splitting the data into train and test sets chronologically
    train_size = int(len(df) * 0.8)
    train_df, test_df = df[:train_size], df[train_size:]

    # Selecting the features and target for training
    train_X = train_df[feature_columns].values
    train_y = train_df['gesamt'].values.reshape(-1, 1)
    test_X = test_df[feature_columns].values
    test_y = test_df['gesamt'].values.reshape(-1, 1)

    # Reshaping input data for LSTM
    train_X_lstm = train_X.reshape(train_X.shape[0], 1, len(feature_columns))
    test_X_lstm = test_X.reshape(test_X.shape[0], 1, len(feature_columns))

    # Building LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, len(feature_columns)), return_sequences=True))
    model.add(TimeDistributed(Dense(len(quantiles))))

    # Compiling model with custom quantile loss function
    model.compile(optimizer='adam', loss='mean_squared_error')  # Replace with your quantile loss

    # Training model
    model.fit(train_X_lstm, train_y, epochs=10, batch_size=32, verbose=1)

    # Generating predictions for future timestamps
    future_timestamps = [start_date + timedelta(hours=h) for h in horizons]
    prediction_inputs = np.array([extract_features_for_timestamp(ts, df) for ts in future_timestamps])
    prediction_inputs = prediction_inputs.reshape(-1, 1, len(feature_columns))
    predictions = model.predict(prediction_inputs)
    predictions_reshaped = predictions.reshape(predictions.shape[0], -1)

    df_static = pd.DataFrame({
        'forecast_date': [start_date.strftime('%Y-%m-%d')] * len(predictions_reshaped),
        'target': ['energy'] * len(predictions_reshaped),
        'horizon': [f'{h} hour' for h in horizons for _ in range(len(predictions_reshaped) // len(horizons))]
    })
    columns = [f'q{q}' for q in quantiles]
    # DataFrame for predictions
    df_predictions = pd.DataFrame(predictions_reshaped, columns=columns)

    # Concatenate static DataFrame with predictions DataFrame
    final_df = pd.concat([df_static, df_predictions], axis=1)

    return final_df


# Helper function to extract features for a given timestamp
def extract_features_for_timestamp(timestamp, df):
    return [timestamp.hour, timestamp.day, timestamp.month, timestamp.weekday()]


