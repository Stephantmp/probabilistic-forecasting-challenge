from keras.models import Model
from keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functions import reorder_quantiles
from functions import get_DAX


def DAX_LSTM(data=None, date_str=None):
    if data is None:
        data = get_DAX.get()
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    df = pd.DataFrame(data)
    # Assuming `df` is your DataFrame
    df = df.iloc[6:]
    window_size = 5  # This can be adjusted based on your analysis needs

    # Fill NaN values with a moving average for each column
    for column in df.columns:
        # Calculate the moving average, ignoring NaNs. `min_periods=1` ensures we get a value even if there are fewer than `window_size` non-NaN values.
        moving_average = df[column].rolling(window=window_size, min_periods=1).mean()

        # Use `fillna` to replace NaNs in the original column with the moving average
        df[column].fillna(moving_average, inplace=True)

    # df = df.iloc[:-(max(range(1, 6)))]  # This line would be removed or commented out
    # df.dropna(inplace=True)  # This line would be removed or commented out
    # Ensure the DataFrame index is of datetime type and normalize to remove the time part (if needed)
    df.index = pd.to_datetime(df.index)
    date_str = pd.to_datetime(date_str).strftime('%Y-%m-%d')
    # Convert date_str to datetime, taking into account the timezone
    target_date = pd.to_datetime(date_str).tz_localize(
        'Europe/Berlin')  # Adjust the timezone as per your data

    # Validate if target_date is in the DataFrame index
    # if target_date not in df.index:
    # raise ValueError(f"date_str {date_str} not found in dataset index")
    # else:
    # print("Date found, proceeding with model evaluation", date_str)

    # Assuming these columns are what we're using to predict
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'lag_ret1', 'lag_ret2',
                       'lag_ret3', 'lag_ret4', 'lag_ret5']
    features = df.loc[:target_date, feature_columns].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))

    position = np.where(df.index == df.index[df.index <= target_date].max())[0][0]
    print('Last Timestamp', df.index[df.index <= target_date].max())
    # Now use this position to index into `features_scaled`
    last_features = features_scaled[position].reshape(1, 1, -1)

    # Targets for quantile regression would ideally be structured for direct prediction,
    # but here we'll simulate as if the model is predicting 5 specific values for simplicity.
    # Adjust this as necessary for real quantile prediction.
    target_columns = ['future_ret1', 'future_ret2', 'future_ret3', 'future_ret4', 'future_ret5']
    targets = df.loc[:target_date, target_columns]
    scaler_y = StandardScaler()
    targets_scaled = scaler_y.fit_transform(targets)

    # Model
    input_seq = Input(shape=(features_scaled.shape[1], features_scaled.shape[2]))
    encoder_out, state_h, state_c = LSTM(100, return_state=True)(input_seq)
    encoder_states = [state_h, state_c]

    decoder_lstm = LSTM(100, return_sequences=True)
    decoder_out = decoder_lstm(RepeatVector(1)(encoder_out), initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(25))  # Adjusting for 5 quantiles * 5 future returns
    decoder_outputs = decoder_dense(decoder_out)
    model = Model(inputs=input_seq, outputs=decoder_outputs)
    model.compile(optimizer='adam', loss='mse')

    # Training
    history = model.fit(features_scaled, np.repeat(targets_scaled, 5, axis=1), epochs=10, batch_size=72, verbose=2,
                        shuffle=False)

    # Predictions for the last available day
    predictions_scaled = model.predict(last_features)

    # Since predictions_scaled is (1, 25), and we need it to match the original targets' shape for inverse_transform,
    # Let's first reshape predictions to mimic 5 future returns for a single sample, ignoring the quantile dimension.
    # This is a workaround and simplifies the interpretation of the predictions.
    # A more accurate approach would involve handling each quantile's predictions separately.

    # Assuming the predictions_scaled array is shaped (1, 25), corresponding to 5 days and 5 quantiles each
    predictions = predictions_scaled.reshape(5, 5)  # Reshape to (5 days, 5 quantiles)

    # Prepare the output DataFrame directly from predictions
    forecast_date = pd.to_datetime(date_str).strftime('%Y-%m-%d')  # This is the reference forecast date

    # Horizons are based on forecast_date + 1 day, +2 days, etc.
    horizons = [(pd.to_datetime(forecast_date) + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 6)]

    predictions_df = pd.DataFrame(predictions, columns=['q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975'])
    predictions_df.insert(0, 'horizon', ['1 day', '2 day', '3 day', '4 day','5 day'])  # Now horizon represents the actual forecast dates
    predictions_df.insert(0, 'target', 'DAX')

    # Since forecast_date should be locked, it means all forecasts are made on this date
    forecast_date_column = [forecast_date] * 5  # Repeat the forecast_date for all rows
    predictions_df.insert(0, 'forecast_date', forecast_date_column)
    predictions_df = reorder_quantiles.reorder_quantiles(predictions_df)

    return predictions_df