import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.tools import add_constant
from functions import get_DAX
from datetime import datetime, date
def DAX_quantile_regression(hist=pd.DataFrame()):
    # %%
    hist = get_DAX.get()
    # %%
    hist.head()
    # %%

    tau = [.025, .25, .5, .75, .975]
    # %%
    pred_quantile_regression = np.zeros((5, 5))
    # %%
    # Perform quantile regression for each horizon and quantile
    for i in range(5):
        ret_str = f"ret{i + 1}"
        y = hist[ret_str]  # Dependent variable
        X = hist[[f'lag_ret{j}' for j in range(1, 6)]]  # Independent variables
        X = add_constant(X)  # Adds a constant term to the predictors

        for j, q in enumerate(tau):
            # Fit the model for the q-th quantile
            mod = smf.quantreg(f'{ret_str} ~ lag_ret1 + lag_ret2 + lag_ret3 + lag_ret4 + lag_ret5', hist)
            res = mod.fit(q=q)

            # Predict the quantile for the last observation
            pred_quantile_regression[i, j] = res.predict(X.iloc[-1:])[0]
    # %%
    x = np.arange(5) + 1
    _ = plt.plot(x, pred_quantile_regression, ls="", marker="o", c="black")
    _ = plt.xticks(x, x)
    _ = plt.plot((x, x), (pred_quantile_regression[:, 0], pred_quantile_regression[:, -1]), c='black')
    # %%
    date_str = date.today()  # .strftime('%Y%m%d')
    # %%
    df_sub = pd.DataFrame({
        "forecast_date": date_str,
        "target": "DAX",
        "horizon": [str(i) + " day" for i in (1, 2, 5, 6, 7)],
        "q0.025": pred_quantile_regression[:, 0],
        "q0.25": pred_quantile_regression[:, 1],
        "q0.5": pred_quantile_regression[:, 2],
        "q0.75": pred_quantile_regression[:, 3],
        "q0.975": pred_quantile_regression[:, 4]
    })
    # %%
    return df_sub