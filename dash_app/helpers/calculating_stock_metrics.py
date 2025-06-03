
import os
import requests
import pandas as pd
from scipy.stats import skew, kurtosis

# Get Polygon API from environment
from dotenv import load_dotenv
load_dotenv()
 
api_key = os.getenv("POLYGON_API_KEY")

def get_stock_ticker_information(ticker, api_key):
    
    url = f'https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={api_key}'

    response = requests.get(url)
    return response.json()

def get_stock_ticker_news(ticker):

    url = f'https://api.polygon.io/v2/reference/news?ticker={ticker}&apiKey={api_key}'

    response = requests.get(url)
    return response.json()

def get_stock_ticker_data(ticker, start_date, end_date):

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    response = requests.get(url, params=params)
    data = response.json()
    print(data)

    if "results" not in data:
        raise Exception(f"Polygon API error: {data}")

    df = pd.DataFrame(data["results"])
    df["t"] = pd.to_datetime(df["t"], unit="ms")  # timestamp to datetime
    df = df.rename(columns={"t": "date", "c": "close", "o": "open", "h": "high", "l": "low", "v": "volume"})
    return df[["date", "open", "high", "low", "close", "volume"]]

def get_returns_metrics(stock_df):

    df = stock_df.copy()
    df = df.sort_values("date")
    df["returns"] = df["close"].pct_change().dropna()

    # Basic stats
    total_return = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
    mean_daily_return = df["returns"].mean()
    median_daily_return = df["returns"].median()
    std_daily_return = df["returns"].std()
    var_daily_return = df["returns"].var()

    # Higher moments
    return_skew = skew(df["returns"].dropna())
    return_kurtosis = kurtosis(df["returns"].dropna(), fisher=False)

    return {
        "Total Return": round(total_return, 4),
        "Mean Daily Return": round(mean_daily_return, 4),
        "Median Daily Return": round(median_daily_return, 4),
        "Daily Volatility (Std)": round(std_daily_return, 4),
        "Daily Variance": round(var_daily_return, 4),
        "Skewness": round(return_skew, 4),
        "Kurtosis": round(return_kurtosis, 4),
    }

print(get_stock_ticker_news(ticker="AAPL"))