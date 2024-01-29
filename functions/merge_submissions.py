from datetime import datetime, date
import pandas as pd
def merge(df_energy, df_dax):
    date_str = date.today()#.strftime('%Y%m%d')
    #%%
    df_infections = pd.DataFrame({
        "forecast_date": date_str,
        "target": "infections",
        "horizon": [str(i) + " week" for i in (0 ,1, 2, 3, 4)],
        "q0.025": "NA",
        "q0.25": "NA",
        "q0.5": "NA",
        "q0.75": "NA",
        "q0.975": "NA"
    })

    df_sub=pd.concat([df_dax, df_energy, df_infections])
    df_sub['forecast_date'] = pd.to_datetime(df_sub['forecast_date'], format='%Y-%m-%d')
    print(df_sub)
    #need to change this
    PATH = "../../forecasts"
    date_str = datetime.today().strftime('%Y%m%d')
    df_sub.to_csv(PATH + "/" + date_str + "_JonSnow.csv", index=False)
    #%%
    return df_sub
