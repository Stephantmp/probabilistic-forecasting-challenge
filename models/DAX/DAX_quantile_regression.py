import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.tools import add_constant
from functions import get_DAX
from datetime import date

def DAX_quantile_regression(hist=None):
    if hist is None:
        hist = get_DAX.get()

    tau = [.025, .25, .5, .75, .975]
    pred_quantile_regression = np.zeros((5, 5))

    for i in range(5):
        ret_str = f"ret{i + 1}"
        y = hist[ret_str]
        X = hist[[f'lag_ret{j}' for j in range(1, 6)]]
        X = add_constant(X)

        for j, q in enumerate(tau):
            mod = smf.quantreg(f'{ret_str} ~ lag_ret1 + lag_ret2 + lag_ret3 + lag_ret4 + lag_ret5', hist)
            res = mod.fit(q=q)
           # print(f"Quantile {q}: Model Summary for {ret_str}")
           # print(res.summary())

            pred_quantile_regression[i, j] = res.predict(X.iloc[-1:]).iloc[0]

    date_str = date.today()
    df_sub = pd.DataFrame({
        "forecast_date": date_str,
        "target": "DAX",
        "horizon": [f"{i} day" for i in (1, 2, 5, 6, 7)],
        "q0.025": pred_quantile_regression[:, 0],
        "q0.25": pred_quantile_regression[:, 1],
        "q0.5": pred_quantile_regression[:, 2],
        "q0.75": pred_quantile_regression[:, 3],
        "q0.975": pred_quantile_regression[:, 4]
    })

    #print("Final Predictions:")
    #print(df_sub)

    return df_sub
