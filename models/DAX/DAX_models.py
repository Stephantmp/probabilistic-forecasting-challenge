from keras.models import Model
from keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from datetime import datetime
from functions import reorder_quantiles
import statsmodels.formula.api as smf
from functions import get_DAX
from datetime import date
from functions.get_DAX import get
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

def DAX_baseline(hist=None, date_str=None):
    if hist is None:
        hist = get_DAX.get()

    # Quantiles to predict
    tau = [.025, .25, .5, .75, .975]

    # Initialize array for predictions
    pred_baseline = np.zeros((5, len(tau)))
    # The look-back period for calculating quantiles
    last_t = 1000  # You can adjust this number based on the period you consider

    # Calculate the baseline quantiles for each future return period
    for i in range(1, 6):
        future_ret_str = f"future_ret{i}"
        # Calculate the historical quantiles for the future returns
        pred_baseline[i-1, :] = np.quantile(hist[future_ret_str].iloc[-last_t:], q=tau)

    # Default to today's date if no date is provided
    if date_str is None:
        date_str = date.today().strftime('%Y-%m-%d')

    # Create a DataFrame to hold the predictions
    df_sub = pd.DataFrame({
        "forecast_date": [date_str] * 5,
        "target": ["DAX"] * 5,
        "horizon": [f"{i} day" for i in range(1, 6)],
        "q0.025": pred_baseline[:, 0],
        "q0.25": pred_baseline[:, 1],
        "q0.5": pred_baseline[:, 2],
        "q0.75": pred_baseline[:, 3],
        "q0.975": pred_baseline[:, 4]
    })

    return df_sub

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

    df_sub = pd.DataFrame(predictions, columns=['q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975'])
    df_sub.insert(0, 'horizon', ['1 day', '2 day', '3 day', '4 day','5 day'])  # Now horizon represents the actual forecast dates
    df_sub.insert(0, 'target', 'DAX')

    # Since forecast_date should be locked, it means all forecasts are made on this date
    forecast_date_column = [forecast_date] * 5  # Repeat the forecast_date for all rows
    df_sub.insert(0, 'forecast_date', forecast_date_column)
    df_sub = reorder_quantiles.reorder_quantiles(df_sub)

    return df_sub

def DAX_quantile_regression(hist=None, date_str=None):
    if hist is None:
        hist = get_DAX.get()

    tau = [.025, .25, .5, .75, .975]
    pred_quantile_regression = np.zeros((5, len(tau)))

    # Iterate over the future returns to be predicted
    for i in range(1, 6):
        ret_str = f"future_ret{i}"
        y = hist[ret_str]
        # Prepare the formula string with the future return and lagged returns
        formula_str = f"{ret_str} ~ lag_ret1 + lag_ret2 + lag_ret3 + lag_ret4 + lag_ret5"

        for j, q in enumerate(tau):
            mod = smf.quantreg(formula_str, hist)
            res = mod.fit(q=q)

            # Predict for the last available row of lagged returns
            pred_quantile_regression[i - 1, j] = res.predict(hist.iloc[-1:])[0]

    if date_str is None:
        date_str = date.today().strftime('%Y-%m-%d')

    # Create a DataFrame to hold the predictions
    df_sub = pd.DataFrame({
        "forecast_date": [date_str] * 5,
        "target": ["DAX"] * 5,
        "horizon": [f"{i} day" for i in range(1, 6)],
        "q0.025": pred_quantile_regression[:, 0],
        "q0.25": pred_quantile_regression[:, 1],
        "q0.5": pred_quantile_regression[:, 2],
        "q0.75": pred_quantile_regression[:, 3],
        "q0.975": pred_quantile_regression[:, 4]
    })

    return df_sub

def DAX_XGBoost(data=None, date_str=None):
    """
    Preprocess data, train quantile regression models, and predict future returns for a given forecast date using the provided data.

    Parameters:
    - data: DataFrame containing the DAX data.
    - date_str: The date for which to predict future returns, in 'YYYY-MM-DD' format.

    Returns:
    - df_sub: DataFrame containing the forecast predictions for different quantiles and horizons.
    """
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np
    from xgboost import XGBRegressor
    from functions.reorder_quantiles import reorder_quantiles as reorder
    if data is None:
        daxdata= get()
    if date_str is None:
        date_str = pd.to_datetime('today').strftime('%Y-%m-%d')

    # Preprocess data
    daxdata, scaler_X, features = preprocess_data_for_prediction(data)

    # Train models
    quantile_models = train_and_predict(daxdata, features)

    # Predict future returns for the specified forecast date
    forecast_predictions = predict_for_date(daxdata, date_str, features, scaler_X, quantile_models)

    # Store forecast predictions in a DataFrame
    df_sub = pd.DataFrame({
        "forecast_date": [date_str] * 5,
        "target": ["DAX"] * 5,
        "horizon": [f"{i} day" for i in range(1, 6)],
        "q0.025": [forecast_predictions[i][0.025] for i in range(1, 6)],
        "q0.25": [forecast_predictions[i][0.25] for i in range(1, 6)],
        "q0.5": [forecast_predictions[i][0.5] for i in range(1, 6)],
        "q0.75": [forecast_predictions[i][0.75] for i in range(1, 6)],
        "q0.975": [forecast_predictions[i][0.975] for i in range(1, 6)]
    })

    # Reorder quantiles if necessary
    df_sub = reorder(df_sub)

    return df_sub


def preprocess_data_for_prediction(df):
    df.index = pd.to_datetime(df.index)
    lag_features = ['Close']
    for feature in lag_features:
        for lag in range(1, 6):
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
    df['Close_MA5'] = df['Close'].rolling(window=5).mean().shift(1)
    features = [f'{feat}_lag{j}' for feat in lag_features for j in range(1, 6)] + ['Close_MA5']
    scaler_X = StandardScaler()
    df[features] = scaler_X.fit_transform(df[features].fillna(0))
    return df, scaler_X, features


def xgb_quantile_grad_hess(quantile, y_true, y_pred):
    error = y_true - y_pred
    grad = np.where(error > 0, -quantile, -(quantile - 1))
    hess = np.ones_like(y_pred)
    return grad, hess


def train_and_predict(df, features, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]):
    quantile_models = {}
    for days in range(1, 6):
        X = df[features].iloc[:-days]
        y = df[f'future_ret{days}'].shift(-days).dropna()
        for quantile in quantiles:
            model = XGBRegressor(objective=lambda y_true, y_pred: xgb_quantile_grad_hess(quantile, y_true, y_pred),
                                 n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
            model.fit(X.iloc[:len(y)], y)
            quantile_models[(days, quantile)] = model
    return quantile_models


def predict_for_date(df, forecast_date, features, scaler_X, quantile_models):
    """
    Adjusted to use the most recent data before the forecast date to prepare features
    and predict future returns using quantile models.
    """
    forecast_date = pd.to_datetime(forecast_date).tz_localize('Europe/Berlin')
    # Find the most recent date in the DataFrame before the forecast_date
    most_recent_date = df.index[df.index <= forecast_date].max()
    print(f"Most recent date: {most_recent_date}")
    # Ensure there is data available up to the most recent date
    if pd.isnull(most_recent_date):
        raise ValueError(f"No historical data available before forecast_date: {forecast_date}")

    # Prepare and normalize features for the most recent date
    X_forecast_df = df.loc[[most_recent_date], features]
    X_forecast_scaled = scaler_X.transform(X_forecast_df)

    forecast_predictions = {}
    for (days, quantile), model in quantile_models.items():
        forecast_predictions.setdefault(days, {})
        forecast_predictions[days][quantile] = model.predict(X_forecast_scaled)[0]

    return forecast_predictions



