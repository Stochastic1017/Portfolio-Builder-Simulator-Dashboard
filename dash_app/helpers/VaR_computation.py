
import numpy as np

def VaR_and_ES_log_returns(log_returns, confidence_level):
    """
    Calculate the Value at Risk (VaR) and Expected Shortfall (ES) using log-returns.

    Parameters:
    - prices (numpy array): Historical price data.
    - confidence_level (float): The confidence level for VaR and ES (e.g., 0.95 for 95% confidence).

    Returns:
    - tuple: (VaR, ES), where:
        - VaR (float): Value at Risk
        - ES (float): Expected Shortfall
    """
    
    # Sort the log-returns in ascending order
    sorted_log_returns = np.sort(log_returns)
    
    # Determine the percentile index for VaR
    cutoff_index = int((1 - confidence_level) * len(sorted_log_returns))
    
    # Calculate VaR
    var = -sorted_log_returns[cutoff_index]  # Negate to express as a positive loss
    
    # Calculate Expected Shortfall (average of returns below VaR)
    tail_losses = sorted_log_returns[:cutoff_index]  # Losses below the VaR threshold
    es = -np.mean(tail_losses) if len(tail_losses) > 0 else np.nan  # Handle edge case
    
    return var, es