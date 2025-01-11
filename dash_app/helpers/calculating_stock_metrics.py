
import yfinance as yf
import numpy as np
from scipy import stats

def calculate_stock_metrics(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=f"2y")

    # Calculate log returns
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    
    metrics = {
        'current_price': hist['Close'][-1],
        'total_return': ((hist['Close'][-1] / hist['Close'][0]) - 1) * 100,
        'avg_log_return': log_returns.mean() * 252,  
        'volatility': log_returns.std() * np.sqrt(252),  
        'sharpe_ratio': (log_returns.mean() * 252) / (log_returns.std() * np.sqrt(252)),
        'skewness': stats.skew(log_returns),
        'kurtosis': stats.kurtosis(log_returns)
    }
    
    return metrics, hist, log_returns
