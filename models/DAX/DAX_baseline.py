import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import get_DAX
from datetime import datetime, date

def DAX_baseline(hist=pd.DataFrame()):


    tau = [.025, .25, .5, .75, .975]

    pred_baseline = np.zeros((5, 5))
    # %%
    last_t = 1000

    for i in range(5):
        ret_str = "ret" + str(i + 1)

        pred_baseline[i, :] = np.quantile(hist[ret_str].iloc[-last_t:], q=tau)

    date_str = date.today()  # .strftime('%Y%m%d')
    df_sub = pd.DataFrame({
        "forecast_date": date_str,
        "target": "DAX",
        "horizon": [str(i) + " day" for i in (1, 2, 5, 6, 7)],
        "q0.025": pred_baseline[:, 0],
        "q0.25": pred_baseline[:, 1],
        "q0.5": pred_baseline[:, 2],
        "q0.75": pred_baseline[:, 3],
        "q0.975": pred_baseline[:, 4]})
    return df_sub