
import os
import sys
import numpy as np
import plotly.graph_objects as go

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dash import dcc
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, norm

def empty_placeholder_figure(COLORS):

    """
    Placeholder figure before any potential user input. 
    """

    empty_fig = go.Figure()

    empty_fig.add_annotation(
        text="Please input stock ticker and verify it.",
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

def create_historic_plots(full_name, historical_data, COLORS):
    
    # Compute return metrics
    dates = np.asarray(historical_data['date'])
    daily_prices = np.asarray(historical_data['close'])
    daily_returns = np.asarray(historical_data['close'].pct_change().dropna())

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
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)
    lower_bound = mean_return - 1.96 * std_return
    upper_bound = mean_return + 1.96 * std_return

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
            line=dict(color='magenta', width=2),
        ),
        row=2, col=2
    )

    # Calculate Normal Distribution Curve
    x_norm = np.linspace(min(daily_returns), max(daily_returns), 500)
    y_norm = norm.pdf(x_norm, 
                      loc = mean_return, 
                      scale = std_return)

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
        x0=lower_bound,  
        x1=upper_bound,  
        y0=0,            
        y1=max(y_kde) * 1.1,  
        fillcolor="rgba(255, 255, 255, 0.2)",  
        layer="below",  
        line_width=0,
        row=2, col=2
    )

    # Add vertical dashed line for the lower bound
    historical_daily_plot.add_shape(
        type="line",
        x0=lower_bound,
        x1=lower_bound,
        y0=0,
        y1=max(y_kde) * 1.1,  
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
        showlegend=False,
    )

    return dcc.Graph(
            id="main-output-graph",
            figure=historical_daily_plot,
            config={'responsive': True},
            style={'height': '100%', 'width': '100%'}
        )
