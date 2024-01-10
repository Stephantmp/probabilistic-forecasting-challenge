import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, TimeDistributed
from keras.models import Model
from datetime import timedelta, datetime, date
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K

# Helper function to extract features for a given timestamp
def extract_features_for_timestamp(timestamp, df):
    return [timestamp.hour, timestamp.day, timestamp.month, timestamp.weekday()]


def quantile_loss(quantiles, y_true, y_pred):
    e = y_true - y_pred
    losses = []
    for q in quantiles:
        loss = K.maximum(q * e, (q - 1) * e)
        losses.append(K.mean(loss, axis=-1))
    return K.mean(K.stack(losses, axis=-1), axis=-1)


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
import tensorflow.keras.backend as K
from functions import get_energy


def energy_forecast(input_data, date_str=None):
    # Load default energy data if none provided
    if input_data is None:
        energydata = get_energy.get()
        df = pd.DataFrame(energydata)
    else:
        df = pd.DataFrame(input_data)

    # Default forecast date is the current date
    if date_str is None:
        date_str = datetime.now()
    else:
        date_str = pd.to_datetime(date_str)
    #date_str=datetime.strptime(date_str, '%Y-%m-%d')
    # Data preprocessing
    if 'date_time' in df.columns:
        df.set_index('date_time', inplace=True)

    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['week'] = df.index.weekday

    train_size = int(len(df) * 0.8)
    train_df, test_df = df[:train_size], df[train_size:]

    feature_columns = ['hour', 'day', 'month', 'week']
    train_X = train_df[feature_columns].values
    train_y = train_df['gesamt'].values.reshape(-1, 1)
    test_X = test_df[feature_columns].values
    test_y = test_df['gesamt'].values.reshape(-1, 1)

    train_X_lstm = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    test_X_lstm = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

    scaler = StandardScaler()
    train_X_lstm_scaled = scaler.fit_transform(train_X_lstm.reshape(train_X_lstm.shape[0], -1))
    train_X_lstm_scaled = train_X_lstm_scaled.reshape(train_X_lstm.shape)
    test_X_lstm_scaled = scaler.transform(test_X_lstm.reshape(test_X_lstm.shape[0], -1))
    test_X_lstm_scaled = test_X_lstm_scaled.reshape(-1, 1, train_X_lstm.shape[2])

    scaler_y = StandardScaler()
    train_y_scaled = scaler_y.fit_transform(train_y)
    test_y_scaled = scaler_y.transform(test_y)

    # Model building
    input_seq = Input(shape=(train_X_lstm_scaled.shape[1], train_X_lstm_scaled.shape[2]))
    encoder_out, state_h, state_c = LSTM(100, return_state=True)(input_seq)
    encoder_states = [state_h, state_c]

    decoder_steps = train_X_lstm_scaled.shape[1]
    decoder_lstm = LSTM(100, return_sequences=True)
    decoder_out = decoder_lstm(RepeatVector(decoder_steps)(encoder_out), initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(5))  # Change 5 to the number of quantiles
    decoder_outputs = decoder_dense(decoder_out)
    model = Model(inputs=input_seq, outputs=decoder_outputs)

    # Quantile Loss Function
    def quantile_loss(quantiles, y_true, y_pred):
        e = y_true - y_pred
        losses = []
        for q in quantiles:
            loss = K.maximum(q * e, (q - 1) * e)
            losses.append(K.mean(loss, axis=-1))
        return K.mean(K.stack(losses, axis=-1), axis=-1)

    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(quantiles, y_true, y_pred))

    # Model training
    history = model.fit(train_X_lstm_scaled, train_y_scaled, epochs=10, batch_size=72,
                        validation_data=(test_X_lstm_scaled, test_y_scaled), verbose=2, shuffle=False)

    # Prediction preparation
    horizons = [36, 40, 44, 60, 64, 68]
    future_timestamps = [date_str + timedelta(hours=h) for h in horizons]

    def extract_features_for_timestamp(timestamp, df):
        features = {
            'hour': timestamp.hour,
            'day': timestamp.day,
            'month': timestamp.month,
            'week': timestamp.weekday()
        }
        feature_vector = [features[col] for col in feature_columns]
        return feature_vector

    prediction_inputs = [extract_features_for_timestamp(ts, df) for ts in future_timestamps]
    prediction_inputs = np.array(prediction_inputs).reshape(-1, 1, len(feature_columns))
    prediction_inputs_scaled = scaler.transform(prediction_inputs.reshape(prediction_inputs.shape[0], -1))
    prediction_inputs_scaled = prediction_inputs_scaled.reshape(-1, 1, len(feature_columns))

    # Generate predictions
    predictions = model.predict(prediction_inputs_scaled)
    predictions_reshaped = predictions.reshape(predictions.shape[0], -1)
    original_scale_predictions = scaler_y.inverse_transform(predictions_reshaped)

    # Format predictions
    df_predictions = pd.DataFrame()
    columns = [f'q{q}' for q in quantiles]
    df_predictions[columns] = original_scale_predictions
    df_static = pd.DataFrame({
        'forecast_date': [date_str.strftime('%Y-%m-%d')] * len(original_scale_predictions),
        'target': ['energy'] * len(original_scale_predictions),
        'horizon': [f'{h} hour' for h in horizons for _ in range(len(original_scale_predictions)//len(horizons))]
    })
    final_df = pd.concat([df_static, df_predictions], axis=1)
    model.reset_states()
    return final_df


