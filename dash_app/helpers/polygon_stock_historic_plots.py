
import os
import sys
import dash
import numpy as np
import dash.dash_table as dt
import plotly.graph_objects as go

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dash import html
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, norm
from helpers.polygon_stock_api import StockTickerInformation

# Load .env to fetch api key
load_dotenv()
api_key = os.getenv("POLYGON_API_KEY")

# Define color constants
COLORS = {
    'primary': '#FFD700',      # Golden Yellow
    'secondary': '#FFF4B8',    # Light Yellow
    'background': '#1A1A1A',   # Dark Background
    'card': '#2D2D2D',         # Card Background
    'text': '#FFFFFF'          # White Text
}

def empty_placeholder_figure():

    fig = go.Figure()

    fig.add_annotation(
        text="Please input stock ticker",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color=COLORS['primary']))

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=1000)

    return fig

"""
def create_historic_plots(stock_ticker):

    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark", 
                            paper_bgcolor=COLORS['card'],
                            plot_bgcolor=COLORS['card'],
                            xaxis =  {'visible': False},
                            yaxis = {'visible': False})
    
    empty_fig.add_annotation(text="Search for ticker and click 'Show plot'",
                             xref="paper",
                             yref="paper",
                             x=0.5,
                             y=0.5,
                             showarrow=False,
                             font=dict(size=16, color="white"),  # Adjust font size and color
                             align="center")
    
    error_message = ""
    portfolio_data = portfolio_data or []

    # Check if no ticker is provided
    if not stock_ticker:
        error_message = "Enter at least 2 valid stock ticker."
        return html.Div(error_message), empty_fig, portfolio_data, error_message

    try:

        polygon_api = StockTickerInformation(ticker=stock_ticker, api_key=api_key)
        full_name = polygon_api.get_metadata()['results']['name']


        # Display metrics
        metrics_display = html.Div([
            html.P(
                f"{full_name}",
                style={'color': COLORS['primary'], 'fontWeight': 'bold', 'fontSize': '1.1em'}
            ),
            html.P(f"Current Date: {metrics['current_date']}"),
            html.P(f"Current Price: ${metrics['current_price']:.5f}"),
            html.P(f"Mean: {metrics['mean']:.5f}"),
            html.P(f"Standard Deviation: {metrics['std']:.5f}"),
            html.P(f"Variance: {metrics['variance']:.5f}"),
            html.P(f"Skewness: {metrics['skewness']:.5f}"),
            html.P(f"Kurtosis: {metrics['kurtosis']:.5f}")
        ])

        # Calculate the 95% confidence interval
        lower_bound = metrics['mean'] - 1.96 * metrics['std']
        upper_bound = metrics['mean'] + 1.96 * metrics['std']

        # Handle "Show Plot" button click
        ctx = dash.callback_context  # Get the context of the triggered callback
        if ctx.triggered and "add-plot-button" in ctx.triggered[0]['prop_id']:
            
            ####################
            ### Define Subplots
            ####################
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"colspan": 2}, None], [{}, {}]],
                subplot_titles=[
                    "Price with Bollinger Bands",
                    "Log Returns",
                    "Log Return Distribution"
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )

            #####################
            ### Price Plot with
            ### Bollinger Bands
            #####################

            # Add Bollinger Bands and other plots
            rolling_mean = hist['Close'].rolling(window=20).mean()
            rolling_std = hist['Close'].rolling(window=20).std()
            bollinger_upper = rolling_mean + 1.96 * rolling_std
            bollinger_lower = rolling_mean - 1.96 * rolling_std

            # Add traces for price with Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=hist.index, 
                    y=hist['Close'],
                    mode='lines', 
                    name='Adjusted Close',
                    line=dict(color=COLORS['primary'])
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=hist.index, 
                    y=bollinger_upper,
                    mode='lines',
                    name='Bollinger Upper',
                    line=dict(color='white')
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=hist.index, 
                    y=bollinger_lower,
                    mode='lines', 
                    name='Bollinger Lower',
                    fill='tonexty', 
                    fillcolor='rgba(231, 240, 247, 0.4)',
                    line=dict(color='white')
                ),
                row=1, col=1
            )

            #################################
            ### Line Plot for Log Returns
            ### with 95% Confidence Interval
            #################################

            fig.add_trace(
                go.Scatter(x=hist.index, y=log_returns, name='Log Returns'),
                row=2, col=1
            )

            # Add shaded rectangle for the 95% confidence interval
            fig.add_shape(
                type="rect",
                x0=min(hist.index),  # Start of the x-axis
                x1=max(hist.index),  # End of the x-axis
                y0=lower_bound,      # Lower confidence bound
                y1=upper_bound,      # Upper confidence bound
                fillcolor="rgba(255, 255, 255, 0.3)", 
                layer="below", 
                line_width=0,
                row=2, col=1
            )

            # Add the line plot for log returns vs day
            fig.add_trace(
                go.Scatter(
                    x=hist.index,  # Assuming hist contains the dates as the index
                    y=log_returns,
                    mode='lines',
                    name='Log Returns',
                    line=dict(color=COLORS['primary']),
                ),
                row=2, col=1
            )

            # Add horizontal line for the lower bound
            fig.add_shape(
                type="line",
                x0=min(hist.index),  # Start of the x-axis
                x1=max(hist.index),  # End of the x-axis
                y0=lower_bound,      # Horizontal line at lower bound
                y1=lower_bound,
                line=dict(color="white", dash="dash", width=2),
                row=2, col=1
            )

            # Add horizontal line for the upper bound
            fig.add_shape(
                type="line",
                x0=min(hist.index),  # Start of the x-axis
                x1=max(hist.index),  # End of the x-axis
                y0=upper_bound,      # Horizontal line at upper bound
                y1=upper_bound,
                line=dict(color="white", dash="dash", width=2),
                row=2, col=1
            )

            ##############################
            ### Histogram of Log Returns
            ### with KDE + Gaussian Curve
            ##############################

            # Histogram of Log Returns
            fig.add_trace(
                go.Histogram(
                    x=log_returns,
                    name='Log Return Distribution',
                    histnorm='probability density',
                    marker=dict(color=COLORS['primary']),
                ),
                row=2, col=2
            )

            # Calculate KDE
            kde = gaussian_kde(log_returns)
            x_kde = np.linspace(min(log_returns), max(log_returns), 500)
            y_kde = kde(x_kde)

            # Add KDE trace
            fig.add_trace(
                go.Scatter(
                    x=x_kde,
                    y=y_kde,
                    mode='lines',
                    name='KDE',
                    line=dict(color='orange', width=2),
                ),
                row=2, col=2
            )

            # Calculate Normal Distribution Curve
            x_norm = np.linspace(min(log_returns), max(log_returns), 500)
            y_norm = norm.pdf(x_norm, loc=metrics['mean'], scale=metrics['std'])

            # Add Normal Distribution trace
            fig.add_trace(
                go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', width=2),
                ),
                row=2, col=2
            )

            # Calculate the 95% confidence bounds
            mean = metrics['mean']
            std = metrics['std']
            lower_bound = mean - 1.96 * std
            upper_bound = mean + 1.96 * std

            # Add shaded rectangle for 95% confidence interval
            fig.add_shape(
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
            fig.add_shape(
                type="line",
                x0=lower_bound,
                x1=lower_bound,
                y0=0,
                y1=max(y_kde) * 1.1,  # Align with rectangle height
                line=dict(color="white", dash="dash", width=2),
                row=2, col=2
            )

            # Add vertical dashed line for the upper bound
            fig.add_shape(
                type="line",
                x0=upper_bound,
                x1=upper_bound,
                y0=0,
                y1=max(y_kde) * 1.1,  # Align with rectangle height
                line=dict(color="white", dash="dash", width=2),
                row=2, col=2
            )
            
            fig.update_layout(
                template="plotly_dark",
                title=f"Analysis for {full_name}",
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['background'],
                font=dict(color=COLORS['text']),
                title_font=dict(color=COLORS['primary']),
                showlegend=False
            )

            return metrics_display, fig, portfolio_data, error_message

        # Handle "Add to Portfolio" button click
        if ctx.triggered and "add-stock-button" in ctx.triggered[0]['prop_id']:
            # Check if the ticker is already in the portfolio
            if any(row['ticker'] == stock_ticker for row in portfolio_data):
                error_message = f"{stock_ticker} is already in the portfolio."
                return metrics_display, empty_fig, portfolio_data, error_message

            # Add stock to portfolio
            portfolio_data.append({
                'number': len(portfolio_data) + 1,
                'ticker': stock_ticker,
                'full_name': full_name
            })
            return metrics_display, empty_fig, portfolio_data, error_message

        # Default response if no button is clicked
        return metrics_display, empty_fig, portfolio_data, error_message

    except Exception as error:
        error_message = "Error retrieving data. Check the ticker."
        return html.Div(), empty_fig, portfolio_data, error_message
"""