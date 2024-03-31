import pandas as pd
from tqdm import tqdm
from datetime import datetime
import holidays


def get(last_years=3):


    def get_energy_data():
        import requests
        # get all available time stamps
        stampsurl = "https://www.smard.de/app/chart_data/410/DE/index_quarterhour.json"
        response = requests.get(stampsurl)
        # ignore first 4 years (don't need those in the baseline and speeds the code up a bit)
        timestamps = list(response.json()["timestamps"])[4 * 52:]

        col_names = ['date_time', 'Netzlast_Gesamt']
        energydata = pd.DataFrame(columns=col_names)

        # loop over all available timestamps
        for stamp in tqdm(timestamps):

            dataurl = "https://www.smard.de/app/chart_data/410/DE/410_DE_quarterhour_" + str(stamp) + ".json"
            response = requests.get(dataurl)
            rawdata = response.json()["series"]

            for i in range(len(rawdata)):
                rawdata[i][0] = datetime.fromtimestamp(int(str(rawdata[i][0])[:10])).strftime("%Y-%m-%d %H:%M:%S")

            energydata = pd.concat([energydata, pd.DataFrame(rawdata, columns=col_names)])

        energydata = energydata.dropna()
        energydata["date_time"] = pd.to_datetime(energydata.date_time)
        # set date_time as index
        energydata.set_index("date_time", inplace=True)
        # resample
        energydata = energydata.resample("1h", label="left").sum()

        return energydata

    df = get_energy_data()
    df = df.rename(columns={"Netzlast_Gesamt": "gesamt"})
    df['gesamt'] = df['gesamt'] / 1000
    df["weekday"] = df.index.weekday  # Monday=0, Sunday=6

    def get_energy_data_excluding_holidays_and_old_data(df, last_years=3):
        # Determine the current year and calculate the start year for the data inclusion
        current_year = pd.to_datetime('now').year
        start_year = current_year - last_years

        # Generate the cutoff date for data inclusion
        cutoff_date = pd.Timestamp(year=start_year, month=1, day=1)

        # Generate German holidays for the specified years
        de_holidays = holidays.DE(years=range(start_year, current_year + 1))

        # Convert holiday dates to string format for filtering
        holiday_strs = {date.strftime('%Y-%m-%d') for date in de_holidays.keys()}
        print(holiday_strs)
        # Manually add the Christmas to Heilige Drei Könige period for each year
        for year in range(start_year, current_year + 1):
            christmas_period = pd.date_range(start=f"{year}-12-25", end=f"{year + 1}-01-06")
            holiday_strs.update(set(christmas_period.strftime('%Y-%m-%d')))

        # Ensure the DataFrame's index is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        # First, filter out the dates older than the last_years threshold
        df_filtered = df[df.index >= cutoff_date]

        # Next, filter out the holiday dates
        df_filtered = df_filtered[~df_filtered.index.strftime('%Y-%m-%d').isin(holiday_strs)]

        return df_filtered
    df_filtered=get_energy_data_excluding_holidays_and_old_data(df,last_years)
    return df_filtered

