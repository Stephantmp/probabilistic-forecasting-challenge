# Load the data from Yahoo Finance and compute the returns
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

#function returns the data from Yahoo Finance and compute the returns
def get():
    def compute_return(y, r_type="log", h=1):
        if r_type == "log":
            ret = (np.log(y) - np.log(y.shift(h))) * 100
        else:
            ret = ((y - y.shift(h)) / y.shift(h)) * 100
        return ret

    msft = yf.Ticker("^GDAXI")
    hist = msft.history(period="max")
    for i in range(5):
        hist["ret" + str(i + 1)] = compute_return(hist["Close"], h=i + 1)

    # Create lagged returns as independent variables
    for i in range(1, 6):
        hist[f'lag_ret{i}'] = hist['Close'].shift(i)

    # Remove rows with NaN values that result from lagging
    hist.dropna(inplace=True)
    return hist
