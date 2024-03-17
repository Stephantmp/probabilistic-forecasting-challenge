# Import necessary libraries and functions
from functions.get_DAX import get
from models.DAX import DAX_quantile_regression, DAX_baseline
from functions.evaluation import evaluate_horizon
from tqdm import tqdm
from functions.evaluate_dax import evaluate
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from xgboost import XGBRegressor


def run_model(data=None, date_str=None):
    """
    Preprocess data, train quantile regression models, and predict future returns for a given forecast date using the provided data.

    Parameters:
    - data: DataFrame containing the DAX data.
    - date_str: The date for which to predict future returns, in 'YYYY-MM-DD' format.

    Returns:
    - df_forecast: DataFrame containing the forecast predictions for different quantiles and horizons.
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
    df_forecast = pd.DataFrame({
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
    df_forecast = reorder(df_forecast)

    return df_forecast


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
    most_recent_date = df.index[df.index < forecast_date].max()

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



