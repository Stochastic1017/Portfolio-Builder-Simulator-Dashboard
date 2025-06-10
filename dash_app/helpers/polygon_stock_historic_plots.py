
import os
import sys
import numpy as np
import plotly.graph_objects as go

from dash import dcc, html
from datetime import date
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, norm
from dateutil.relativedelta import relativedelta

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def dash_range_selector(default_style):
    
    return html.Div(
        id="range-selector-container",
        children=[
            html.Div(
                style={"display": "flex", 
                       "justifyContent": "start", 
                       "flexWrap": "wrap",
                       "gap": "10px",
                       "paddingTop": "10px", 
                       "marginBottom": "10px"},

                children=[
                    html.Button("1M", id="range-1M", n_clicks=0, style=default_style, className="simple"),
                    html.Button("3M", id="range-3M", n_clicks=0, style=default_style, className="simple"),
                    html.Button("6M", id="range-6M", n_clicks=0, style=default_style, className="simple"),
                    html.Button("1Y", id="range-1Y", n_clicks=0, style=default_style, className="simple"),
                    html.Button("5Y", id="range-5Y", n_clicks=0, style=default_style, className="simple"),
                    html.Button("All", id="range-all", n_clicks=0, style=default_style, className="simple"),
                ],

            )
        ]
    )

def create_historic_plots(full_name, historical_df, filtered_df, COLORS):
    
    # Extract relevant metrics (filtered)
    dates = np.asarray(filtered_df['date'])
    daily_prices = np.asarray(filtered_df['close'])
    daily_returns = np.asarray(filtered_df['close'].pct_change().dropna())
    long_run_returns = np.asarray(historical_df['close'].pct_change().dropna())

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
        horizontal_spacing=0.1,
        shared_xaxes=True,
        shared_yaxes=False
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
    ### Histogram of Returns
    ### with KDE + Gaussian Curve
    ##############################

    # Red histogram for all data
    historical_daily_plot.add_trace(
        go.Histogram(
            x=long_run_returns,
            name="Long-run Daily Returns Distribution",
            histnorm="probability density",
            marker=dict(color="red", opacity=0.4),
            nbinsx=60,
        ),
        row=2, col=2
    )
   
    # Yellow histogram for filtered data
    historical_daily_plot.add_trace(
        go.Histogram(
            x=daily_returns,
            name="Short-Run Daily Returns Distribution",
            histnorm="probability density",
            marker=dict(color="yellow", opacity=0.4),
            nbinsx=60,
        ),
        row=2, col=2
    )

    # Calculate KDE for Long-Run
    kde_all = gaussian_kde(long_run_returns)
    x_vals = np.linspace(min(long_run_returns.min(), daily_returns.min()), 
                         max(long_run_returns.max(), daily_returns.max()), 
                         500)
    y_kde_all = kde_all(x_vals)

    # Add KDE trace (Long-Run)
    historical_daily_plot.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_kde_all,
            mode="lines",
            name="KDE (Long-Run)",
            line=dict(color="red", width=2)
        ),
        row=2, col=2
    )

    # Calculate KDE for Short-Run
    kde_filtered = gaussian_kde(daily_returns)
    y_kde_filtered = kde_filtered(x_vals)
    
    # Add KDE trace (Short-Run)
    historical_daily_plot.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_kde_filtered,
            mode="lines",
            name="KDE (Short-Run)",
            line=dict(color="yellow", width=2)
        ),
        row=2, col=2
    )

        # Vertical dashed upper bound line
    
    # Final layout matching dash color theme
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

"""
    # Shaded confidence interval box
    historical_daily_plot.add_shape(
        type="rect",
        x0=lower_bound,  
        x1=upper_bound,  
        y0=y_lower_limit - 2,  # go a bit below for visual extension
        y1=y_upper_limit + 5,  # go above actual peak
        xref="x2",
        yref="y2",
        fillcolor="rgba(255, 255, 255, 0.2)",  
        layer="below",  # make sure this sits under histogram
        line_width=0,
    )

    # Vertical dashed lower bound line
    historical_daily_plot.add_shape(
        type="line",
        x0=lower_bound,
        x1=lower_bound,
        y0=y_lower_limit - 2,
        y1=y_upper_limit + 5,
        xref="x2",
        yref="y2",
        line=dict(color="white", dash="dash", width=2)
    )

    # Vertical dashed upper bound line
    historical_daily_plot.add_shape(
        type="line",
        x0=upper_bound,
        x1=upper_bound,
        y0=y_lower_limit - 2,
        y1=y_upper_limit + 5,
        xref="x2",
        yref="y2",
        line=dict(color="white", dash="dash", width=2)
    )

    # Histogram plot (KDE + histogram)
    historical_daily_plot.update_yaxes(
        row=2, col=2,
        range=[0, y_upper_limit + 5]
    )
"""