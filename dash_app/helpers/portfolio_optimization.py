
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from helpers.calculating_stock_metrics import read_data

def portfolio_performance(weights, expected_returns, cov_matrix):
    """Calculate portfolio return and volatility."""
    portfolio_return = np.sum(expected_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.02):
    """Calculate negative Sharpe ratio for minimization."""
    portfolio_return, portfolio_volatility = portfolio_performance(weights, expected_returns, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate=0.02):
    """Find optimal portfolio weights."""
    num_assets = len(expected_returns)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
    )
    bounds = tuple((0, 1) for _ in range(num_assets))  # weights between 0 and 1
    
    # Initial guess (equal weights)
    initial_weights = np.array([1/num_assets] * num_assets)
    
    # Optimize!
    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(expected_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

def generate_efficient_frontier(expected_returns, cov_matrix, num_portfolios=1000):
    """Generate efficient frontier points."""
    num_assets = len(expected_returns)
    returns = []
    volatilities = []
    
    for _ in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio performance
        portfolio_return, portfolio_volatility = portfolio_performance(weights, expected_returns, cov_matrix)
        returns.append(portfolio_return)
        volatilities.append(portfolio_volatility)
    
    return np.array(volatilities), np.array(returns)

def get_expected_returns_covariance_matrix(tickers):
    """
    Fetch stock data for the provided tickers, calculate expected returns and the covariance matrix.
    
    Args:
        tickers (list): List of stock tickers.
        
    Returns:
        dict: Dictionary with the following keys:
            - 'expected_returns': Series of annualized expected returns for each ticker.
            - 'covariance_matrix': Covariance matrix of daily returns.
            - 'data': Dictionary of stock data with ticker as key and historical data as value.
    """
    data = {}  # Dictionary to store historical data for each ticker
    
    for ticker in tickers:
        metrics, hist, log_returns = read_data(ticker)
        if not hist.empty:
            data[ticker] = hist['Close']  # Store only 'Close' prices for the ticker

    if not data:
        raise ValueError("No valid data for the given tickers.")

    # Convert the data dictionary to a DataFrame for return calculations
    prices_df = pd.DataFrame(data)

    # Calculate daily returns
    returns = prices_df.pct_change().dropna()

    # Calculate annualized expected returns and covariance matrix
    expected_returns = (1 + returns.mean()) ** 252 - 1
    cov_matrix = returns.cov() * 252

    return {
        'expected_returns': expected_returns,
        'covariance_matrix': cov_matrix,
        'data': data
    }
