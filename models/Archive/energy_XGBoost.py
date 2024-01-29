import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from functions import get_energy

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

    return final_df