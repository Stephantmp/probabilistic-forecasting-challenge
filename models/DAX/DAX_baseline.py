import pandas as pd
import numpy as np
from datetime import date
from functions import get_DAX

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

