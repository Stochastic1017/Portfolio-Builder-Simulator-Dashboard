
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy import stats
from dash import dcc, html, dash_table
from plotly.subplots import make_subplots

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

def create_historic_plots(full_name, dates, daily_prices, daily_returns, COLORS):
    
    ######################
    ### Defining Subplots
    ######################

    # Create subplots
    historical_daily_plot = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=[
            "Daily Prices with Bollinger Bands (Rolling Window = 5 days)",
            "Daily Returns with 95% Confidence Intervals",
            "Histogram of Daily Returns with 95% Confidence Intervals"
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        shared_xaxes=False,
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

    # Calculate 5-day rolling statistics
    rolling_mean = pd.Series(daily_prices).rolling(window=5).mean().to_numpy()
    rolling_std = pd.Series(daily_prices).rolling(window=5).std().to_numpy()

    # Compute Bollinger Bands
    upper_band = rolling_mean + 1.96 * rolling_std
    lower_band = rolling_mean - 1.96 * rolling_std

    # Upper Bollinger Band
    historical_daily_plot.add_trace(
        go.Scatter(
            x=dates,
            y=upper_band,
            mode='lines',
            name='Upper Bollinger Bound',
            line=dict(color="rgba(255,255,255,0.5)"),
        ),
        row=1, col=1
    )

    # Lower Bollinger Band
    historical_daily_plot.add_trace(
        go.Scatter(
            x=dates,
            y=lower_band,
            mode='lines',
            name='Lower Bollinger Bound',
            line=dict(color="rgba(255,255,255,0.5)"),
            fill='tonexty', 
            fillcolor="rgba(255,255,255,0.05)",
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

    # Add horizontal line for the upper bound
    historical_daily_plot.add_trace(
            go.Scatter(
                    x=dates,  
                    y=[upper_bound] * len(dates),
                    mode='lines',
                    name='Upper 95% Confidence Bound',
                    line=dict(color="rgba(255,255,255,0.5)")
                ),
        row=2, col=1
    )

    # Add horizontal line for the lower bound
    historical_daily_plot.add_trace(
            go.Scatter(
                    x=dates,  
                    y=[lower_bound] * len(dates),
                    mode='lines',
                    name='Upper 95% Confidence Bound',
                    line=dict(color="rgba(255,255,255,0.5)"),
                    fill='tonexty',
                    fillcolor="rgba(255,255,255,0.05)",
                ),
        row=2, col=1
    )

    ##############################
    ### Histogram + Gaussian Fit
    ##############################

    # Combined range to align both histogram and normal PDF
    return_range = np.linspace(daily_returns.min(), daily_returns.max(), 500)
    hist_counts, bin_edges = np.histogram(daily_returns, bins=50, density=True)

    # Histogram of daily returns
    historical_daily_plot.add_trace(
        go.Histogram(
            y=daily_returns,
            name="Daily Returns Histogram",
            marker_color=COLORS['primary'],
            nbinsy=50,
            orientation='h',
            opacity=0.6,
            histnorm='probability density'
        ),
        row=2, col=2
    )

    # Gaussian fit using sample estimates
    historical_daily_plot.add_trace(
        go.Scatter(
            x=stats.norm.pdf(return_range, mean_return, std_return),           
            y=return_range,         
            mode="lines",
            name=f"Gaussian fit with sample estimates",
            line=dict(color="#E0E0E0", width=4),
            showlegend=True
        ),
        row=2, col=2
    )

    # Horizontal line for upper bound of 95% confidence
    historical_daily_plot.add_trace(
        go.Scatter(
            x=[0, hist_counts.max()],
            y=[upper_bound] * len(dates),
            mode='lines',
            name='Upper 95% Bound',
            line=dict(color="rgba(255,255,255,0.5)"),
            showlegend=True
        ),
        row=2, col=2
    )

    # Horizontal line for lower bound of 95% confidence
    historical_daily_plot.add_trace(
        go.Scatter(
            x=[0, hist_counts.max()],
            y=[lower_bound] * len(dates),
            mode='lines',
            name='Lower 95% Bound',
            line=dict(color="rgba(255,255,255,0.5)"),
            fill='tonexty',
            fillcolor="rgba(255,255,255,0.05)",
            showlegend=True
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
        showlegend=False
    )

    return dcc.Graph(
                id="main-output-graph",
                figure=historical_daily_plot,
                config={'responsive': True},
                style={'height': '100%', 'width': '100%'}
            )

def summarize_daily_returns(daily_returns):
    
    daily_returns = pd.Series(daily_returns).dropna()

    # Basic stats
    mean = daily_returns.mean()
    median = daily_returns.median()
    std = daily_returns.std()
    skew = daily_returns.skew()
    kurt = daily_returns.kurt()
    min_val = daily_returns.min()
    max_val = daily_returns.max()

    # Hypothesis test: mean = 0
    t_stat, p_val = stats.ttest_1samp(daily_returns, popmean=0)

    # Normality test (Shapiro-Wilk is good for <5000 samples)
    normality_test = stats.shapiro(daily_returns)
    normal_stat, normal_pval = normality_test

    return {
        "Mean": mean,
        "Median": median,
        "Standard Deviation": std,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Minimum": min_val,
        "Maximum": max_val,
        "t-Statistic (mean = 0)": t_stat,
        "p-Value (mean = 0)": p_val,
        f"Shapiro-Wilk Test Statistic": normal_stat,
        f"Shapiro-Wilk Test p-Value": normal_pval
    }

def create_statistics_table(daily_returns, COLORS):
    stats_dict = summarize_daily_returns(daily_returns)
    
    sections = {
        "Sample Statistics": ["Mean", "Median", "Standard Deviation", "Skewness", "Kurtosis", 
                              "Minimum", "Maximum"],
        "Hypothesis Test (Mean = 0)": ["t-Statistic (mean = 0)", "p-Value (mean = 0)"],
        "Normality Test": [key for key in stats_dict.keys() if "Test" in key],
    }

    table_rows = []
    for section, metrics in sections.items():
        # Add section header as a row
        table_rows.append({
            "Metric": f"{section}",
            "Value": ""
        })
        # Add each metric under the section
        for metric in metrics:
            table_rows.append({
                "Metric": metric,
                "Value": stats_dict.get(metric, "")
            })

    # Now render a single DataTable
    return html.Div([
        dash_table.DataTable(
            data=table_rows,
            columns=[
                {"name": "Metric", "id": "Metric"},
                {"name": "Value", "id": "Value"}
            ],
            style_cell={
                'textAlign': 'left',
                'padding': '8px',
                'color': 'white',
                'border': 'none',
                'backgroundColor': COLORS['background'],
                'fontFamily': 'monospace'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Value} = ""'},
                    'fontWeight': 'bold',
                    'backgroundColor': COLORS['background'],
                    'color': COLORS['primary'],
                }
            ],
            style_header={
                'display': 'none',
            },
            style_table={
                "marginTop": "30px",
                "padding": "10px 20px",
                "borderTop": "2px solid #ddd",
                "maxHeight": "500px",
                "overflowY": "auto",
            },

            # Disable all interactivity
            editable=False,
            row_selectable=False,
            selected_rows=[],
            active_cell=None,
            cell_selectable=False,
            sort_action="none",
            filter_action="none",
            page_action="none",
            style_as_list_view=True,
        )
    ])
