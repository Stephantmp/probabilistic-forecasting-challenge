from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, TimeDistributed
from keras.models import Model
from datetime import timedelta, datetime, date
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K
from datetime import date, datetime
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from functions import get_energy
from functions.reorder_quantiles import reorder_quantiles


def energy_baseline(df, date_str=None):
    # %%if date str is none, use today's date
    if date_str==None:
        date_str = date.today()
    tau = [.025, .25, .5, .75, .975]
    # Define the lead times
    horizons_def = [36, 40, 44, 60, 64, 68]
    horizons = [h + 1 for h in horizons_def]

    # Function to calculate future dates based on a horizon
    def get_date_from_horizon(last_ts, horizon):
        if isinstance(last_ts, str):
            # Convert string to datetime
            last_ts = pd.to_datetime(last_ts)

        # Add DateOffset
        return last_ts + pd.DateOffset(hours=horizon)

    # Generating horizon dates
    horizon_date = [get_date_from_horizon(date_str , h) for h in horizons]
    # rows correspond to horizon, columns to quantile level
    pred_baseline = np.zeros((6, 5))
    # %%
    # seasonal regression
    # Create dummy variables for months and hours
    df['month'] = df.index.month
    df['hour'] = df.index.hour

    # Get dummies for months and hours, excluding the first month and hour to avoid multicollinearity
    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
    hour_dummies = pd.get_dummies(df['hour'], prefix='hour', drop_first=True)

    # Join the dummies with the original DataFrame
    df = df.join(month_dummies).join(hour_dummies)

    # %%
    # baseline
    last_t = 100
    LAST_IDX = -1

    for i, d in enumerate(horizon_date):
        weekday = d.weekday()
        hour = d.hour

        df_tmp = df.iloc[:LAST_IDX]

        cond = (df_tmp.weekday == weekday) & (df_tmp.index.time == d.time())

        pred_baseline[i, :] = np.quantile(df_tmp[cond].iloc[-last_t:]["gesamt"], q=tau)

    if date_str==None:
        date_str = date.today()
    # %%
    df_sub = pd.DataFrame({
        "forecast_date": date_str,
        "target": "energy",
        "horizon": [str(h) + " hour" for h in horizons_def],
        "q0.025": pred_baseline[:, 0],
        "q0.25": pred_baseline[:, 1],
        "q0.5": pred_baseline[:, 2],
        "q0.75": pred_baseline[:, 3],
        "q0.975": pred_baseline[:, 4]})
    df_sub
    return df_sub



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


def energy_LSTM(input_data, date_str=None):
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
            loss = K.maximum(2*q * e, 2*(q - 1) * e)
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
    final_df=reorder_quantiles(final_df)
    return final_df


def energy_quantile_regression(df,date_str=None):
    # Fetching and initial preprocessing of the dataset
    if df is None:
        df = get_energy.get()
    if date_str==None:
        date_str = date.today()
    # Define the lead times
    horizons_def = [36, 40, 44, 60, 64, 68]
    horizons = [h + 1 for h in horizons_def]

    # Function to calculate future dates based on a horizon
    def get_date_from_horizon(last_ts, horizon):
        if isinstance(last_ts, str):
            # Convert string to datetime
            last_ts = pd.to_datetime(last_ts)

        # Add DateOffset
        return last_ts + pd.DateOffset(hours=horizon)


    # Generating horizon dates
    horizon_date = [get_date_from_horizon(date_str, h) for h in horizons]

    # Define the quantiles for prediction
    tau = [0.025, 0.25, 0.5, 0.75, 0.975]

    # Preprocessing for regression
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
    hour_dummies = pd.get_dummies(df['hour'], prefix='hour', drop_first=True)
    df = df.join(month_dummies).join(hour_dummies)

    # Implementing Quantile Regression
    import statsmodels.api as sm

    # Define the independent variables (X) and the dependent variable (y)
    exclude_columns = ['gesamt', 'month', 'hour']
    X = df[[col for col in df.columns if col not in exclude_columns]]
    y = df['gesamt']

    # Convert data types
    X = X.apply(pd.to_numeric)
    y = y.apply(pd.to_numeric)
    X = X.astype(int)

    # Add a constant to the model (for the intercept)
    X = sm.add_constant(X)

    # Train a model for each quantile
    quantile_models = {quantile: sm.QuantReg(y, X).fit(q=quantile) for quantile in tau}

    # Print summary of one of the quantile models (e.g., median model)
    print(quantile_models[0.5].summary())

    # Additional steps for visualization or output formatting can be added here
    #%%
    # Create a new DataFrame for the forecast
    forecast_df = pd.DataFrame()
    forecast_df = pd.DataFrame({'date_time': horizon_date})
    # Extract and one-hot encode month and hour from the date_time
    forecast_df['month'] = forecast_df['date_time'].dt.month
    forecast_df['hour'] = forecast_df['date_time'].dt.hour
    forecast_df = forecast_df.join(pd.get_dummies(forecast_df['month'], prefix='month', drop_first=True))
    forecast_df = forecast_df.join(pd.get_dummies(forecast_df['hour'], prefix='hour', drop_first=True))

    # Drop the original month and hour columns
    forecast_df.drop(['month', 'hour'], axis=1, inplace=True)

    # Add constant and weekday columns
    forecast_df['const'] = 1.0
    forecast_df['weekday'] = forecast_df['date_time'].dt.dayofweek

    # Ensure all columns in X are also in forecast_df
    for col in X.columns:
        if col not in forecast_df.columns and col != 'date_time':
            forecast_df[col] = 0

    # Reorder columns to match X (excluding date_time)
    forecast_columns = [col for col in X.columns if col != 'date_time']
    forecast_df = forecast_df[['date_time'] + forecast_columns]
    forecast_df.drop(['date_time'], axis=1, inplace=True)
    forecast_df.head()
    #%%
    quantile_models
    #%%
    forecast_df2 = forecast_df.copy()
    #%%
    print(forecast_df)

    # Make predictions for each quantile
    for quantile, model in quantile_models.items():
        forecast_var = model.predict(forecast_df)
        forecast_df2[f'prediction_q{quantile}'] = forecast_var
    print(forecast_df2)
    '''
    # Formatting the output
    df_sub = pd.DataFrame()
    for quantile in tau:
        df_sub[f'q{quantile}'] = forecast_df[f'prediction_q{quantile}']
    print(df_sub)
    '''

    if date_str == None:
        date_str = date.today()
    df_sub = pd.DataFrame({
        "forecast_date": date_str,
        "target": "energy",
        "horizon": [str(h) + " hour" for h in horizons_def],
        "q0.025": forecast_df2["prediction_q0.025"],
        "q0.25": forecast_df2["prediction_q0.25"],
        "q0.5": forecast_df2["prediction_q0.5"],
        "q0.75": forecast_df2["prediction_q0.75"],
        "q0.975": forecast_df2["prediction_q0.975"]})
    return df_sub

import pandas as pd
from datetime import date, datetime
from functions import get_energy
def energy_quantile_regression_temp(df, temperature_forecast, date_str=None):
    # Fetching and initial preprocessing of the dataset
    if df is None:
        df = get_energy.get()
    if date_str is None:
        date_str = pd.to_datetime('today').normalize()

    # Define the lead times
    horizons_def = [36, 40, 44, 60, 64, 68]
    horizons = [h + 1 for h in horizons_def]

    # Function to calculate future dates based on a horizon
    def get_date_from_horizon(last_ts, horizon):
        if isinstance(last_ts, str):
            # Convert string to datetime
            last_ts = pd.to_datetime(last_ts)
        return last_ts + pd.DateOffset(hours=horizon)

    # Generating horizon dates
    horizon_date = [get_date_from_horizon(date_str, h) for h in horizons]

    # Define the quantiles for prediction
    tau = [0.025, 0.25, 0.5, 0.75, 0.975]

    # Preprocessing for regression
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
    hour_dummies = pd.get_dummies(df['hour'], prefix='hour', drop_first=True)
    df = df.join(month_dummies).join(hour_dummies)

    # Implementing Quantile Regression
    import statsmodels.api as sm

    # Define the independent variables (X) and the dependent variable (y)
    exclude_columns = ['gesamt', 'month', 'hour']
    X = df[[col for col in df.columns if col not in exclude_columns]]
    y = df['gesamt']

    # Convert data types
    X = X.apply(pd.to_numeric)
    y = y.apply(pd.to_numeric)
    X = X.astype(int)

    # Add a constant to the model (for the intercept)
    X = sm.add_constant(X)

    # Train a model for each quantile
    quantile_models = {quantile: sm.QuantReg(y, X).fit(q=quantile) for quantile in tau}

    # Print summary of one of the quantile models (e.g., median model)
    print(quantile_models[0.5].summary())

    # Creating the forecast DataFrame
    forecast_df = pd.DataFrame({'date_time': horizon_date})

    # Check if temperature forecast is provided and has correct length
    if temperature_forecast is not None and len(temperature_forecast) == len(horizon_date):
        forecast_df['temperature_2m'] = temperature_forecast
    else:
        raise ValueError("Temperature forecast is not provided or does not match the length of the forecast dates.")

    # Extract and one-hot encode month and hour from the date_time
    forecast_df['month'] = forecast_df['date_time'].dt.month
    forecast_df['hour'] = forecast_df['date_time'].dt.hour
    forecast_df = forecast_df.join(pd.get_dummies(forecast_df['month'], prefix='month', drop_first=True))
    forecast_df = forecast_df.join(pd.get_dummies(forecast_df['hour'], prefix='hour', drop_first=True))

    # Drop the original month and hour columns
    forecast_df.drop(['month', 'hour'], axis=1, inplace=True)

    # Add constant and weekday columns
    forecast_df['const'] = 1.0
    forecast_df['weekday'] = forecast_df['date_time'].dt.dayofweek

    # Ensure all columns in X are also in forecast_df
    for col in X.columns:
        if col not in forecast_df.columns and col != 'date_time':
            forecast_df[col] = 0

    # Reorder columns to match X (excluding date_time)
    forecast_columns = [col for col in X.columns if col != 'date_time']
    forecast_df = forecast_df[['date_time'] + forecast_columns]
    forecast_df.drop(['date_time'], axis=1, inplace=True)

    # Make predictions for each quantile
    forecast_df2 = forecast_df.copy()
    for quantile, model in quantile_models.items():
        forecast_var = model.predict(forecast_df)
        forecast_df2[f'prediction_q{quantile}'] = forecast_var

    # Formatting the output
    df_sub = pd.DataFrame({
        "forecast_date": date_str,
        "target": "energy",
        "horizon": [str(h) + " hour" for h in horizons_def],
        "q0.025": forecast_df2["prediction_q0.025"],
        "q0.25": forecast_df2["prediction_q0.25"],
        "q0.5": forecast_df2["prediction_q0.5"],
        "q0.75": forecast_df2["prediction_q0.75"],
        "q0.975": forecast_df2["prediction_q0.975"]
    })

    return df_sub


def energy_XGBoost(input_data=None, date_str=None):
    # Helper function to extract features for a given timestamp
    def extract_features_for_timestamp(timestamp, df):
        return [timestamp.hour, timestamp.day, timestamp.month, timestamp.weekday()]

    # Gradient and Hessian for quantile regression with XGBoost
    def xgb_quantile_grad_hess(quantile, y_true, y_pred):
        error = y_true - y_pred
        grad = np.where(error > 0, -quantile, -(quantile - 1))
        hess = np.ones_like(y_pred)
        return grad, hess

    # Load data
    if input_data is None:
        df = pd.DataFrame(get_energy.get())
    else:
        df = pd.DataFrame(input_data)

    # Set index as datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df['date_time'] = pd.to_datetime(df['date_time'])
        df.set_index('date_time', inplace=True)

    # Extract features
    feature_columns = ['hour', 'day', 'month', 'week']
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['week'] = df.index.weekday

    # Prepare data for model
    X = df[feature_columns].values
    y = df['gesamt'].values.reshape(-1, 1)

    # Scaling
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    # Train models for each quantile
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
    models = {}
    for q in quantiles:
        model = xgb.XGBRegressor(objective=lambda y_true, y_pred: xgb_quantile_grad_hess(q, y_true, y_pred))
        model.fit(X_scaled, y_scaled)
        models[q] = model

    # Generate predictions for future timestamps
    horizons = [36, 40, 44, 60, 64, 68]
    base_date = datetime.strptime(date_str, '%Y-%m-%d')
    #add 24 hrs to base date
    base_date_plus_1=base_date+timedelta(days=1)
    future_timestamps = [base_date_plus_1 + timedelta(hours=h) for h in horizons]
    prediction_inputs = [extract_features_for_timestamp(ts, df) for ts in future_timestamps]
    prediction_inputs_scaled = scaler_X.transform(prediction_inputs)

    predictions = {}
    for q in quantiles:
        model = xgb.XGBRegressor(objective=lambda y_true, y_pred: xgb_quantile_grad_hess(q, y_true, y_pred))
        model.fit(X_scaled, y_scaled)
        pred_scaled = model.predict(prediction_inputs_scaled)
        # Check if predictions are valid (not NaN)
        if np.isnan(pred_scaled).any():
            print(f"Warning: NaN predictions for quantile {q}")
        else:
            pred_original_scale = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
            predictions[q] = pred_original_scale.flatten()  # Flatten the predictions

    # Formatting predictions into DataFrame
    # Initialize DataFrame for static information
    df_static = pd.DataFrame({
        'forecast_date': [base_date] * len(horizons),
        'target': ['energy'] * len(horizons),
        'horizon': [f'{h} hour' for h in horizons]
    })
    if predictions:
        df_predictions = pd.DataFrame(predictions)
        # Correctly format column names
        df_predictions.columns = [f'q{q}' for q in quantiles]
        final_df = pd.concat([df_static.reset_index(drop=True), df_predictions], axis=1)
    else:
        print("Error: No valid predictions were generated.")
        final_df = pd.DataFrame()
    final_df=reorder_quantiles(final_df)
    return final_df


