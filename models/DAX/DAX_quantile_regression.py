import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from functions import get_DAX
from datetime import date

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

