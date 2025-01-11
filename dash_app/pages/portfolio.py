
import os
import sys
import dash

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dash import html, dcc, Input, Output, State, callback
import dash.dash_table as dt
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from helpers.calculating_stock_metrics import calculate_stock_metrics

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

                        # Portfolio Table
                        dt.DataTable(
                            id='portfolio-table',
                            columns=[
                                {'name': 'Number', 'id': 'number'},
                                {'name': 'Ticker', 'id': 'ticker'},
                            ],
                            style_table={'overflowX': 'auto'},
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

                        # Hidden storage for shared state
                        dcc.Store(id="portfolio-tickers"),

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
                                    dcc.Link(
                                        "Go to Efficient Frontier",
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
                                        'backgroundColor': '#8B5CF6',
                                        'color': 'white',
                                        'padding': '10px 20px',
                                        'border': 'none',
                                        'borderRadius': '5px',
                                        'cursor': 'pointer',
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
                            href="https://github.com/Stochastic1017/Spotify-Podcast-Clustering",
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
    error_message = ""

    # Ensure portfolio data is initialized
    portfolio_data = portfolio_data or []

    # Validate stock ticker
    if not stock_ticker:
        error_message = ""
        return html.Div(error_message), empty_fig, portfolio_data, error_message

    try:
        # Calculate stock metrics
        metrics, hist, log_returns = calculate_stock_metrics(stock_ticker)

        # Prepare stock metrics display
        metrics_display = html.Div([
            html.P(f"Current Price: ${metrics['current_price']:.2f}"),
            html.P(f"Total Return: {metrics['total_return']:.2f}%"),
            html.P(f"Annualized Avg Log Return: {metrics['avg_log_return']:.2f}%"),
            html.P(f"Annualized Volatility: {metrics['volatility']:.2f}%"),
            html.P(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}"),
            html.P(f"Skewness: {metrics['skewness']:.2f}"),
            html.P(f"Kurtosis: {metrics['kurtosis']:.2f}")
        ])

        # Handle "Show Plot" button click
        if show_plot_clicks:
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

            # Log Returns
            fig.add_trace(
                go.Scatter(
                    x=hist.index, y=log_returns,
                    mode='lines', name='Log Returns',
                    line=dict(color='#79b9e7')
                ),
                row=2, col=1
            )

            # Histogram of Log Returns
            fig.add_trace(
                go.Histogram(
                    x=log_returns, histnorm='density',
                    name='Log Returns Distribution',
                    marker=dict(color='#8B5CF6')
                ),
                row=2, col=2
            )

            fig.update_layout(
                height=1000,
                template="plotly_dark",
                title_text=f"Analysis for {stock_ticker}",
                showlegend=False,
                autosize=True
            )
        else:
            fig = empty_fig

        # Handle "Add to Portfolio" button click
        if add_stock_clicks:
            # Validate portfolio size
            if len(portfolio_data) >= 10:
                error_message = "Portfolio cannot contain more than 10 tickers."
                return metrics_display, empty_fig, portfolio_data, error_message

            # Check if ticker is already in portfolio
            if any(row['ticker'] == stock_ticker for row in portfolio_data):
                error_message = "Ticker is already in the portfolio."
                return metrics_display, empty_fig, portfolio_data, error_message

            # Add ticker to portfolio
            portfolio_data.append({
                'number': len(portfolio_data) + 1,
                'ticker': stock_ticker
            })

        return metrics_display, fig, portfolio_data, error_message

    except Exception as e:
        # Handle errors
        error_message = "No price data found. Check ticker."
        return html.Div(error_message), empty_fig, portfolio_data, error_message

@callback(
    Output('portfolio-tickers', 'data'),
    [Input('portfolio-table', 'data')],
)
def store_portfolio_tickers(table_data):
    if not table_data:
        return []
    # Extract just the tickers from the table data and return as a list
    return [row['ticker'] for row in table_data]