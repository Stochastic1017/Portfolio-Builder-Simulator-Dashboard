
import requests
import pandas as pd
from datetime import datetime

class StockTickerInformation():

    def __init__(self, ticker, api_key):

        """
        Initializing function.

        Args:
            ticker: unique abbreviation of publicly traded stock ticker.
            api_key: Polygon.io api key
        """

        self.ticker = ticker
        self.api_key = api_key

    def get_metadata(self):

        """ 
        Retrieve detailed reference information for a specific stock ticker from the Polygon.io API.
        
        Returns:
            dict: A dictionary containing metadata about the specified stock ticker.
        
        Raises:
            requests.RequestException: If the HTTP request fails.
        """
        
        try:
            url = f'https://api.polygon.io/v3/reference/tickers/{self.ticker}?apiKey={self.api_key}'
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as error:
            raise Exception(f"HTTP request failed: {error}")

    def get_news(self):
        
        """ 
        Retrieve detailed news about specific stock ticker from the Polygon.io API.
        
        Returns:
            dict: A dictionary containing news and sentiments about the specified stock ticker.
        
        Raises:
            requests.RequestException: If the HTTP request fails.
        """

        try:
            url = f'https://api.polygon.io/v2/reference/news?ticker={self.ticker}&apiKey={self.api_key}'
            response = requests.get(url)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:
            raise Exception(f"HTTP request failed: {error}")

    def get_all_data(self):

        """ 
        Fetches historical daily aggregated stock data for a given ticker symbol within a specified date range 
        using the Polygon.io API.

        Parameters:
            start_date (str): The start date of the data range in 'YYYY-MM-DD' format.
            end_date (str): The end date of the data range in 'YYYY-MM-DD' format.

        Returns:
            Pandas DataFrame containing daily stock data with the date, open, high, low, close, and volume.

        Raises:
            Exception: If the API response does not contain the expected 'results' key or if another error occurs.
        """

        start_date = "2000-01-01"
        end_date = datetime.today().strftime("%Y-%m-%d")
        url = f"https://api.polygon.io/v2/aggs/ticker/{self.ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()

            if "results" not in data:
                raise Exception(f"Polygon API error: {data}")

            df = pd.DataFrame(data["results"])
            df["t"] = pd.to_datetime(df["t"], unit="ms")  # timestamp to datetime
            df = df.rename(columns={"t": "date", 
                                    "c": "close", 
                                    "o": "open", 
                                    "h": "high", 
                                    "l": "low", 
                                    "v": "volume"})
            
            return df.sort_values(by="date")

        except requests.RequestException as error:
            raise Exception(f"HTTP request failed: {error}")
