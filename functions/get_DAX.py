# Load the data from Yahoo Finance and compute the returns
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm


# Function returns the data from Yahoo Finance and compute the returns
def get(last_years=3):
    def compute_return(y, r_type="log", h=1):
        if r_type == "log":
            ret = (np.log(y) - np.log(y.shift(h))) * 100
        else:
            ret = ((y - y.shift(h)) / y.shift(h)) * 100
        return ret

    msft = yf.Ticker("^GDAXI")
    hist = msft.history(period="max")

    # Compute future returns for 1 to 5 days as the dependent variable (Y)
    for i in range(1, 6):
        hist[f'future_ret{i}'] = compute_return(hist["Close"], h=-i)

    # Compute lagged returns for 1 to 5 days as independent variables (X)
    for i in range(1, 6):
        hist[f'lag_ret{i}'] = compute_return(hist["Close"], h=i)

    # Only consider last 3 years of data
    current_year= pd.to_datetime('now').year
    start_year = current_year - last_years
    hist = hist[hist.index.year >= start_year]
    return hist



