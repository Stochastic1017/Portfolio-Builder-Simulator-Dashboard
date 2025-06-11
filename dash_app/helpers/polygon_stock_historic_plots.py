
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
        shared_yaxes=False,
        column_widths=[0.7, 0.3]
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
            y=long_run_returns,
            name="Long-run Daily Returns Distribution",
            histnorm="probability density",
            marker=dict(color="red", opacity=0.4),
            orientation='h',
            nbinsy=50
        ),
        row=2, col=2
    )
   
    # Yellow histogram for filtered data
    historical_daily_plot.add_trace(
        go.Histogram(
            y=daily_returns,
            name="Short-Run Daily Returns Distribution",
            histnorm="probability density",
            marker=dict(color="yellow", opacity=0.4),
            orientation='h',
            nbinsy=50,
        ),
        row=2, col=2
    )

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
