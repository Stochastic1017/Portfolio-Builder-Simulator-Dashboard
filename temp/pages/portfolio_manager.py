
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
from dash_app.helpers.polygon_stock_api import read_data

# Register the page
# dash.register_page(__name__, path="/portfolio-management")

# Define color constants
COLORS = {
    'primary': '#FFD700',      # Golden Yellow
    'secondary': '#FFF4B8',    # Light Yellow
    'background': '#1A1A1A',   # Dark Background
    'card': '#2D2D2D',         # Card Background
    'text': '#FFFFFF'          # White Text
}     

layout = html.Div(

    style={
        'background': COLORS['background'],
        'minHeight': '100vh',
        'padding': '20px',
        'color': COLORS['text'],
        'fontFamily': '"Inter", system-ui, -apple-system, sans-serif'
    },

    children=[

        # Header Section
        html.Div(
            style={
                'marginBottom': '30px',
                'textAlign': 'center',
                'borderBottom': f'2px solid {COLORS["primary"]}',
                'paddingBottom': '20px'
            },
            children=[
                html.H1(
                    "Portfolio Manager",
                    style={
                        'color': COLORS['primary'],
                        'fontSize': '2.5em',
                        'marginBottom': '10px'
                    }
                )
            ]
        ),

        # Main content grid
        html.Div(
            style={
                'display': 'grid',
                'gridTemplateColumns': '1fr 3fr',
                'gap': '20px',
                'maxWidth': '1800px',
                'margin': '0 auto',
            },
            children=[
                # Left Sidebar
                html.Div(
                    style={
                        'backgroundColor': COLORS['card'],
                        'borderRadius': '10px',
                        'padding': '20px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'gap': '15px'
                    },
                    children=[
                        # Stock Input
                        dcc.Input(
                            id="stock-input",
                            type="text",
                            placeholder="Enter stock ticker...",
                            style={
                                'width': '95%',
                                'padding': '10px',
                                'backgroundColor': COLORS['background'],
                                'border': f'1px solid {COLORS["primary"]}',
                                'borderRadius': '5px',
                                'color': COLORS['text'],
                                'fontSize': '1em'
                            }
                        ),

                        # Stock Metrics Display
                        html.Div(
                            id="stock-metrics",
                            style={
                                'backgroundColor': COLORS['background'],
                                'borderRadius': '8px',
                                'padding': '15px',
                                'minHeight': '100px'
                            }
                        ),

                        # Buttons Container
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
                                        'backgroundColor': COLORS['primary'],
                                        'color': COLORS['background'],
                                        'padding': '12px',
                                        'border': 'none',
                                        'borderRadius': '5px',
                                        'cursor': 'pointer',
                                        'fontWeight': 'bold',
                                        'transition': 'all 0.3s ease'
                                    }
                                ),
                                html.Button(
                                    "Add to Portfolio",
                                    id="add-stock-button",
                                    style={
                                        'flex': '1',
                                        'backgroundColor': COLORS['primary'],
                                        'color': COLORS['background'],
                                        'padding': '12px',
                                        'border': 'none',
                                        'borderRadius': '5px',
                                        'cursor': 'pointer',
                                        'fontWeight': 'bold',
                                        'transition': 'all 0.3s ease'
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
                                {'name': 'Full Name', 'id': 'full_name'},
                            ],
                            style_table={
                                'overflowY': 'auto',  # Enable scrolling
                                'height': '300px',    # Limit height for consistent UI
                            },
                            style_cell={
                                'backgroundColor': COLORS['card'],    # Card background
                                'color': COLORS['text'],              # White text
                                'textAlign': 'center',                # Centered text alignment
                                'padding': '10px',                    # Padding for readability
                                'border': 'none',                     # Remove border lines
                                'fontFamily': '"Inter", sans-serif'
                            },
                            style_header={
                                'backgroundColor': COLORS['primary'],  # Golden yellow for header
                                'color': COLORS['background'],         # Dark text for contrast
                                'fontWeight': 'bold',
                                'textAlign': 'center'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'state': 'selected'},  # Remove selected row styling
                                    'backgroundColor': COLORS['card'],
                                    'color': COLORS['text'],
                                    'fontWeight': 'normal'
                                },
                                {
                                    'if': {'state': 'active'},  # Remove active cell styling
                                    'backgroundColor': COLORS['card'],
                                    'color': COLORS['text'],
                                    'fontWeight': 'normal'
                                },
                                {
                                    'if': {'state': 'hover'},  # Remove hover styling
                                    'backgroundColor': COLORS['card'],
                                    'color': COLORS['text']
                                }
                            ],
                            editable=False,           # Disable editing
                            row_deletable=True,       # Disable row deletion
                            sort_action="none",       # Disable sorting
                            filter_action="none",     # Disable filtering
                            page_action="none",       # Disable pagination
                            style_as_list_view=True,  # Remove any default interactivity styling
                            selected_rows=[],         # Prevent row selection
                            active_cell=None,         # Prevent active cell highlighting
                            row_selectable=None,      # Disable row selection
                        ),

                        # Error Message
                        html.Div(
                            id='error-message',
                            style={
                                'color': '#FF6B6B',
                                'textAlign': 'center',
                                'padding': '10px',
                                'minHeight': '20px'
                            }
                        ),

                        # Navigation Button
                        html.Button(
                            id="efficient-frontier-button",
                            children=dcc.Link(
                                "Go to Portfolio Optimization",
                                href="/portfolio-optimization-performance",
                                style={
                                    "color": COLORS['background'],
                                    "textDecoration": "none",
                                    "fontWeight": "bold",
                                    "width": "100%",
                                    "display": "block"
                                }
                            ),
                            style={
                                'backgroundColor': COLORS['primary'],
                                'padding': '12px',
                                'border': 'none',
                                'borderRadius': '5px',
                                'cursor': 'pointer',
                                'width': '100%',
                                'marginTop': '15px',
                                'transition': 'all 0.3s ease'
                            }
                        ),
                    ]
                ),

                # Main Plot Area
                html.Div(
                    style={
                        'backgroundColor': COLORS['card'],
                        'borderRadius': '10px',
                        'padding': '20px',
                    },
                    children=[
                        dcc.Graph(
                            id='main-plot',
                            style={
                                'height': '100%'
                            },
                            config={
                                'responsive': True,
                                'displayModeBar': True
                            }
                        )
                    ]
                )
            ]
        ),

        # Footer
        html.Footer(
            style={
                'backgroundColor': COLORS['card'],
                'padding': '15px',
                'textAlign': 'center',
                'borderRadius': '10px',
                'marginTop': '20px'
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
                            style={'color': COLORS['primary'], 'textDecoration': 'none'}
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
        # Fetch stock data and metrics
        ticker = yf.Ticker(stock_ticker)
        full_name = ticker.info.get("longName", "Name not available")
        metrics, hist, log_returns = read_data(stock_ticker)

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
        'backgroundColor': COLORS['primary'] if is_enabled else COLORS['text'],
        'color': 'white',
        'padding': '10px 20px',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer' if is_enabled else 'not-allowed',
        'fontWeight': 'bold',
        'textAlign': 'center',
        'width': '100%',
    }
