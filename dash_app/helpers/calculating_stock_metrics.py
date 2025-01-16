
import yfinance as yf
import numpy as np
from scipy import stats

def read_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=f"2y")

    # Calculate log returns
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    
    metrics = {
        'current_date': hist.index[-1].strftime('%Y-%m-%d'),
        'current_price': hist['Close'][-1],
        'mean': np.mean(log_returns),
        'std': np.std(log_returns),
        'variance': np.var(log_returns),
        'skewness': stats.skew(log_returns),
        'kurtosis': stats.kurtosis(log_returns)
    }
    
    return metrics, hist, log_returns
