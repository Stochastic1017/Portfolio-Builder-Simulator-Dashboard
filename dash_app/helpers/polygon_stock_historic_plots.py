
import os
import sys
import numpy as np
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dash import html
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, norm
from polygon_stock_api import StockTickerInformation

# Define color constants
COLORS = {
    'primary': '#FFD700',      # Golden Yellow
    'secondary': '#FFF4B8',    # Light Yellow
    'background': '#1A1A1A',   # Dark Background
    'card': '#2D2D2D',         # Card Background
    'text': '#FFFFFF'          # White Text
}

def empty_placeholder_figure():

    """
    Placeholder figure before any potential user input. 
    """

    empty_fig = go.Figure()

    empty_fig.add_annotation(
        text="Please input stock ticker",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color=COLORS['primary']))

    empty_fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        height=1000)

    return empty_fig

def compute_metrics_from_historical(returns):

        """
        Retrieve return metrics (total returns, mean returns, median returns, volatility, variance, skewness, 
        and kurtosis).

        Parameters:
            returns : np.ndarray of daily returns

        Returns:
            Dictionary containing daily return metrics.
        """

        # Basic stats
        mean_daily_return = np.mean(returns)
        median_daily_return = np.median(returns)
        std_daily_return = np.std(returns)
        var_daily_return = np.var(returns)

        # Higher moments
        skew_daily_return = skew(returns)
        kurt_daily_return = kurtosis(returns, fisher=False)

        return {
            "daily_mean": round(mean_daily_return, 4),
            "daily_median": round(median_daily_return, 4),
            "daily_std": round(std_daily_return, 4),
            "daily_variance": round(var_daily_return, 4),
            "daily_skewness": round(skew_daily_return, 4),
            "daily_kurtosis": round(kurt_daily_return, 4),
        }

def create_historic_plots(stock_ticker, api_key):

    # Set up API call to fetch metadata and historical daily data
    polygon_api = StockTickerInformation(ticker=stock_ticker, api_key=api_key)
    
    # Fetch metadata
    full_name = polygon_api.get_metadata()['results']['name']
    
    # Fetch historical daily data
    hist = polygon_api.get_all_data()
    hist = hist.sort_values('date')

    # Compute return metrics
    dates = np.asarray(hist['date'])
    daily_prices = np.asarray(hist['close'])
    daily_returns = np.asarray(hist['close'].pct_change().dropna())
    metrics = compute_metrics_from_historical(daily_returns)

    ######################
    ### Defining Subplots
    ######################

    # Create subplots
    historical_daily_plot = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=[
            "Daily Prices",
            "Daily Returns",
            "Daily Returns Distribution"
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    ################
    ### Price Plot
    ################

    # Add traces for closing prices
    historical_daily_plot.add_trace(
        go.Scatter(
            x=dates, 
            y=daily_prices,
            mode='lines', 
            name='Daily Closing Prices',
            line=dict(color=COLORS['primary'])
        ),
        row=1, col=1
    )

    #################################
    ### Line Plot for Daily Returns
    ### with 95% Confidence Interval
    #################################

    # Calculate the two-sided 95% confidence bounds
    mean = metrics['daily_mean']
    std = metrics['daily_std']
    lower_bound = mean - 1.96 * std
    upper_bound = mean + 1.96 * std

    # Add shaded rectangle for the two-sided 95% confidence interval
    historical_daily_plot.add_shape(
        type="rect",
        x0=min(dates),  
        x1=max(dates),  
        y0=lower_bound,      
        y1=upper_bound,      
        fillcolor="rgba(255, 255, 255, 0.3)", 
        layer="below", 
        line_width=0,
        row=2, col=1
    )

    # Add the line plot for daily returns vs day
    historical_daily_plot.add_trace(
        go.Scatter(
            x=dates,
            y=daily_returns,
            mode='lines',
            name='Daily Returns',
            line=dict(color=COLORS['primary']),
        ),
        row=2, col=1
    )

    # Add horizontal line for the lower bound
    historical_daily_plot.add_shape(
        type="line",
        x0=min(dates),  
        x1=max(dates),  
        y0=lower_bound, 
        y1=lower_bound,
        line=dict(color="white", dash="dash", width=2),
        row=2, col=1
    )

    # Add horizontal line for the upper bound
    historical_daily_plot.add_shape(
        type="line",
        x0=min(dates),  
        x1=max(dates),  
        y0=upper_bound, 
        y1=upper_bound,
        line=dict(color="white", dash="dash", width=2),
        row=2, col=1
    )

    ##############################
    ### Histogram of Log Returns
    ### with KDE + Gaussian Curve
    ##############################

    # Histogram of Log Returns
    historical_daily_plot.add_trace(
        go.Histogram(
            x=daily_returns,
            name='Daily Return Distribution',
            histnorm='probability density',
            marker=dict(color=COLORS['primary']),
        ),
        row=2, col=2
    )

    # Calculate KDE
    kde = gaussian_kde(daily_returns)
    x_kde = np.linspace(min(daily_returns), max(daily_returns), 500)
    y_kde = kde(x_kde)

    # Add KDE trace
    historical_daily_plot.add_trace(
        go.Scatter(
            x=x_kde,
            y=y_kde,
            mode='lines',
            name='Kernel Density Estimator',
            line=dict(color='green', width=2),
        ),
        row=2, col=2
    )

    # Calculate Normal Distribution Curve
    x_norm = np.linspace(min(daily_returns), max(daily_returns), 500)
    y_norm = norm.pdf(x_norm, 
                      loc=metrics['daily_mean'], 
                      scale=metrics['daily_std'])

    # Add Normal Distribution trace
    historical_daily_plot.add_trace(
        go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2),
        ),
        row=2, col=2
    )

    # Add shaded rectangle for 95% confidence interval
    historical_daily_plot.add_shape(
        type="rect",
        x0=lower_bound,  # Lower bound
        x1=upper_bound,  # Upper bound
        y0=0,            # Start of the y-axis
        y1=max(y_kde) * 1.1,  # Extend slightly above the KDE
        fillcolor="rgba(255, 255, 255, 0.2)",  # Semi-transparent white
        layer="below",  # Behind other elements
        line_width=0,
        row=2, col=2
    )

    # Add vertical dashed line for the lower bound
    historical_daily_plot.add_shape(
        type="line",
        x0=lower_bound,
        x1=lower_bound,
        y0=0,
        y1=max(y_kde) * 1.1,  # Align with rectangle height
        line=dict(color="white", dash="dash", width=2),
        row=2, col=2
    )

    # Add vertical dashed line for the upper bound
    historical_daily_plot.add_shape(
        type="line",
        x0=upper_bound,
        x1=upper_bound,
        y0=0,
        y1=max(y_kde) * 1.1,  # Align with rectangle height
        line=dict(color="white", dash="dash", width=2),
        row=2, col=2
    )
    
    historical_daily_plot.update_layout(
        template="plotly_dark",
        title=f"Historical Daily Performance Analysis for {full_name}",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        title_font=dict(color=COLORS['primary']),
        showlegend=False
    )

    return historical_daily_plot
