
import pandas as pd
from datetime import date, datetime
from functions import get_energy
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

