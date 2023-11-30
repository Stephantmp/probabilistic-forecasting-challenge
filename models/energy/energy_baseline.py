from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np


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


    date_str = datetime.today().strftime('%Y%m%d')
    # %%
    date_str = date.today()  # - timedelta(days=1)
    date_str = date_str
    date_str
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