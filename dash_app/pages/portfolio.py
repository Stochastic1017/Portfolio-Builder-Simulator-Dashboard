
import os
import sys
import dash
import yfinance as yf
import numpy as np
import dash.dash_table as dt
import plotly.graph_objects as go

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dash import html, dcc, Input, Output, State, callback
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, norm
from helpers.calculating_stock_metrics import read_data

# Register the page
dash.register_page(__name__, path="/portfolio")

# Updated CSS with purple theme
app_css = {
    'background': 'linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%)',
    'fontFamily': '"Inter", system-ui, -apple-system, sans-serif',
    'color': '#FFFFFF',
}

# Color constants
PURPLE_PRIMARY = '#8B5CF6'    
PURPLE_DARK = '#5B21B6'       
GRAY_DARK = '#1E1E1E'        
GRAY_LIGHT = '#374151'       

layout = html.Div(

    style={
        **app_css,
        'minHeight': '100vh',
        'padding': '10px',
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '10px',
    },

    children=[

        # Header
        html.H1(
            "Portfolio Analytics Dashboard",
            style={
                'color': PURPLE_PRIMARY,
                'fontSize': '2em',
                'textAlign': 'center',
                'margin': '10px 0'
            }
        ),

        # Main content grid
        html.Div(
            style={
                'display': 'grid',
                'gridTemplateColumns': '1fr 3fr',
                'gridTemplateRows': 'auto 1fr',
                'gap': '10px',
                'flex': 1,
                'width': '100%',
                'maxWidth': '1400px',
                'margin': '0 auto',
            },
            children=[

                # Sidebar
                html.Div(
                    style={
                        'backgroundColor': GRAY_DARK,
                        'borderRadius': '10px',
                        'padding': '15px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'gap': '15px',
                    },
                    children=[

                        # Stock Input
                        dcc.Input(
                            id="stock-input",
                            type="text",
                            placeholder="Enter stock ticker...",
                            style={
                                'width': '95%',
                                'padding': '8px',
                                'backgroundColor': '#2D2D2D',
                                'border': f'1px solid {PURPLE_PRIMARY}',
                                'borderRadius': '5px',
                                'color': '#FFFFFF',
                                'marginBottom': '10px'
                            }
                        ),

                        # Stock Metrics Display
                        html.Div(
                            id="stock-metrics",
                            style={
                                'backgroundColor': '#2D2D2D',
                                'borderRadius': '5px',
                                'padding': '10px',
                                'minHeight': '100px',
                                'color': '#FFFFFF'
                            }
                        ),

                        # Buttons
                        html.Div(
                            style={
                                'display': 'flex',
                                'gap': '10px',
                                'justifyContent': 'space-between',
                            },
                            children=[
                                html.Button(
                                    "Show Plot",
                                    id="add-plot-button",
                                    style={
                                        'flex': '1',
                                        'backgroundColor': PURPLE_PRIMARY,
                                        'color': 'white',
                                        'padding': '10px',
                                        'border': 'none',
                                        'borderRadius': '5px',
                                        'cursor': 'pointer',
                                        'fontWeight': 'bold',
                                    }
                                ),

                                html.Button(
                                    "Add to Portfolio",
                                    id="add-stock-button",
                                    style={
                                        'flex': '1',
                                        'backgroundColor': PURPLE_PRIMARY,
                                        'color': 'white',
                                        'padding': '10px',
                                        'border': 'none',
                                        'borderRadius': '5px',
                                        'cursor': 'pointer',
                                        'fontWeight': 'bold',
                                    }
                                ),
                            ]
                        ),

                        # Portfolio table of added tickers
                        dt.DataTable(
                            id='portfolio-table',
                            columns=[
                                {'name': 'Number', 'id': 'number'},
                                {'name': 'Ticker', 'id': 'ticker'},
                                {'name': 'Full Name', 'id': 'full_name'},
                            ],
                            style_table={
                                'overflowY': 'auto',  # Enables vertical scrolling
                                'height': '300px',   # Set height for the scrollable table
                                'border': f'1px solid {PURPLE_PRIMARY}'
                            },
                            style_cell={
                                'backgroundColor': GRAY_DARK,
                                'color': '#FFFFFF',
                                'textAlign': 'center',
                                'padding': '10px'
                            },
                            style_header={
                                'backgroundColor': PURPLE_DARK,
                                'fontWeight': 'bold',
                                'textAlign': 'center'
                            },
                            editable=False,
                            row_deletable=True
                        ),


                        html.Div(
                            id='error-message',
                            style={
                                'color': 'red',
                                'marginTop': '10px',
                                'textAlign': 'center'
                            }
                        ),

                        # Navigation to Efficient Frontier Page
                        html.Div(
                            style={
                                'marginTop': '20px',
                                'textAlign': 'center',
                            },
                            children=[
                                html.Button(
                                    id="efficient-frontier-button",
                                    children=dcc.Link(
                                        "Go to Portfolio Optimization",
                                        href="/efficient-frontier",
                                        style={
                                            "color": "white",
                                            "textDecoration": "none",
                                            "fontWeight": "bold",
                                            "display": "block",
                                            "textAlign": "center",
                                            "width": "100%"
                                        }
                                    ),
                                    style={
                                        'backgroundColor': PURPLE_PRIMARY,
                                        'color': 'white',
                                        'padding': '10px 20px',
                                        'border': 'none',
                                        'borderRadius': '5px',
                                        'cursor': 'not-allowed',
                                        'fontWeight': 'bold',
                                        'textAlign': 'center',
                                        'width': '100%',
                                    }
                                )
                            ]
                        ),
                    ]
                ),

                # Main Plot Area
                html.Div(
                    style={
                        'backgroundColor': GRAY_DARK,
                        'borderRadius': '10px',
                        'padding': '15px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'gap': '10px',
                    },
                    children=[
                        dcc.Tabs(
                            id='plot-tabs',
                            value='price',
                            style={
                                'backgroundColor': GRAY_DARK,
                                'color': '#FFFFFF',
                                'fontWeight': 'bold',
                            }
                        ),
                        dcc.Graph(
                            id='main-plot',
                            style={
                                'width': '100%',
                                'height': '100%',
                                'flexGrow': 1
                            },
                            config={
                                'responsive': True
                            }
                        )
                    ]
                )
            ]
        ),

        # Footer Section
        html.Footer(
            id="footer-section",
            style={
                'backgroundColor': '#1E1E1E',
                'padding': '10px',
                'textAlign': 'center',
                'color': '#B3B3B3',
                'fontSize': '0.9rem',
                'marginTop': 'auto',
                'borderTop': '1px solid #282828',
            },
            children=[
                html.P("Developed by Shrivats Sudhir | Contact: stochastic1017@gmail.com"),
                html.P(
                    [
                        "GitHub Repository: ",
                        html.A(
                            "Portfolio Optimization and Visualization Dashboard",
                            href="https://github.com/Stochastic1017/Portfolio-Analysis-Dashboard",
                            target="_blank",
                            style={'color': PURPLE_PRIMARY, 'textDecoration': 'none'}
                        ),
                    ]
                ),
            ]
        )
    ]
)

@callback(
    [
        Output('stock-metrics', 'children'),
        Output('main-plot', 'figure'),
        Output('portfolio-table', 'data'),
        Output('error-message', 'children'),
    ],
    [
        Input('add-plot-button', 'n_clicks'),
        Input('add-stock-button', 'n_clicks'),
    ],
    [
        State('stock-input', 'value'),
        State('portfolio-table', 'data')
    ]
)
def update_dashboard(show_plot_clicks, add_stock_clicks, stock_ticker, portfolio_data):
    # Initialize empty plot and default error message
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark")
    error_message = ""
    portfolio_data = portfolio_data or []

    # Check if no ticker is provided
    if not stock_ticker:
        error_message = "Enter at least 2 valid stock ticker."
        return html.Div(error_message), empty_fig, portfolio_data, error_message

    try:
        # Fetch stock data and metrics
        ticker = yf.Ticker(stock_ticker)
        full_name = ticker.info.get("longName", "Name not available")
        metrics, hist, log_returns = read_data(stock_ticker)

        # Prepare stock metrics display
        metrics_display = html.Div([
            html.P(f"Current Price: ${metrics['current_price']:.5f}"),
            html.P(f"Mean: {metrics['mean']:.5f}"),
            html.P(f"Standard Deviation: {metrics['std']:.5f}"),
            html.P(f"Variance: {metrics['variance']:.5f}"),
            html.P(f"Skewness: {metrics['skewness']:.5f}"),
            html.P(f"Kurtosis: {metrics['kurtosis']:.5f}")
        ])

        # Handle "Show Plot" button click
        ctx = dash.callback_context  # Get the context of the triggered callback
        if ctx.triggered and "add-plot-button" in ctx.triggered[0]['prop_id']:
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

            # Add Bollinger Bands and other plots
            rolling_mean = hist['Close'].rolling(window=20).mean()
            rolling_std = hist['Close'].rolling(window=20).std()
            bollinger_upper = rolling_mean + 2 * rolling_std
            bollinger_lower = rolling_mean - 2 * rolling_std

            # Bollinger Bands
            rolling_mean = hist['Close'].rolling(window=20).mean()
            rolling_std = hist['Close'].rolling(window=20).std()
            bollinger_upper = rolling_mean + 2 * rolling_std
            bollinger_lower = rolling_mean - 2 * rolling_std

            # Add traces for price with Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=hist.index, y=hist['Close'],
                    mode='lines', name='Adjusted Close',
                    line=dict(color='#007dc6')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=hist.index, y=bollinger_upper,
                    mode='lines', name='Bollinger Upper',
                    line=dict(color='lightblue')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=hist.index, y=bollinger_lower,
                    mode='lines', name='Bollinger Lower',
                    fill='tonexty', fillcolor='rgba(231, 240, 247, 0.4)',
                    line=dict(color='lightblue')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=hist.index, y=log_returns, name='Log Returns'),
                row=2, col=1
            )

            # Histogram of Log Returns with KDE and Normal Distribution
            fig.add_trace(
                go.Histogram(
                    x=log_returns,
                    name='Log Return Distribution',
                    histnorm='probability density',
                    marker=dict(color='#8B5CF6'),
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
            mean = metrics['mean']
            std = metrics['std']
            x_norm = np.linspace(min(log_returns), max(log_returns), 500)
            y_norm = norm.pdf(x_norm, loc=mean, scale=std)

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
            fig.update_layout(template="plotly_dark", title=f"Analysis for {stock_ticker}")
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
        return html.Div(error_message), empty_fig, portfolio_data, error_message


@callback(
    Output('portfolio-tickers', 'data'),
    [Input('portfolio-table', 'data')]  # Current portfolio_data from table
)
def update_portfolio_store(table_data):
    if not table_data:
        return []
    return [row['ticker'] for row in table_data]  # Extract tickers from portfolio_data

@callback(
    Output("efficient-frontier-button", "style"),
    Input("portfolio-table", "data")
)
def update_button_style(portfolio_data):
    # Ensure portfolio_data is initialized
    portfolio_data = portfolio_data or []
    is_enabled = len(portfolio_data) >= 2

    return {
        'backgroundColor': PURPLE_PRIMARY if is_enabled else GRAY_LIGHT,
        'color': 'white',
        'padding': '10px 20px',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer' if is_enabled else 'not-allowed',
        'fontWeight': 'bold',
        'textAlign': 'center',
        'width': '100%',
    }
