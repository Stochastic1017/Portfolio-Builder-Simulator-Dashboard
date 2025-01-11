
import numpy as np
import pandas as pd
import yfinance as yf

def compute_portfolio_var_es(tickers, weights, start_date, end_date, confidence_level=0.95, portfolio_value=1_000_000):
    # Fetch historical prices
    prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    # Compute portfolio returns
    weights = np.array(weights)
    portfolio_returns = returns.dot(weights)
    
    # Sort portfolio returns
    sorted_returns = np.sort(portfolio_returns)
    
    # Calculate VaR
    alpha = 1 - confidence_level
    var_index = int(alpha * len(sorted_returns))
    var = sorted_returns[var_index]  # VaR in return terms
    portfolio_var = var * portfolio_value  # VaR in monetary terms
    
    # Calculate Expected Shortfall (ES)
    es = sorted_returns[sorted_returns <= var].mean()  # ES in return terms
    portfolio_es = es * portfolio_value  # ES in monetary terms
    
    return {
        "VaR": portfolio_var,
        "ES": portfolio_es,
        "VaR (Return)": var,
        "ES (Return)": es
    }
