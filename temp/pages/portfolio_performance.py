import os
import sys
import dash
import numpy as np
from scipy.stats import norm, gaussian_kde
import pandas as pd

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc, callback, Input, Output
from helpers.portfolio_optimization import (optimize_portfolio,
                                            generate_efficient_frontier,
                                            portfolio_performance,
                                            optimize_min_variance, 
                                            get_expected_returns_covariance_matrix)

#dash.register_page(__name__, path="/portfolio-optimization-performance")

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
                    "Portfolio Optimization and Performance Dashboard",
                    style={
                        'color': COLORS['primary'],
                        'fontSize': '2.5em',
                        'marginBottom': '10px'
                    }
                )
            ]
        ),

        # Main Content Grid
        html.Div(
            style={
                'display': 'grid',
                'gridTemplateColumns': '1fr 1fr',  # Two columns
                'gridGap': '20px',                # Consistent gap between containers
                'alignItems': 'start',            # Align containers to the top
            },
            children=[
                # Left Column: Portfolio Weights & Distribution
                html.Div(
                    style={
                        'backgroundColor': COLORS['card'],
                        'borderRadius': '10px',
                        'padding': '20px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'gap': '20px',
                        'height': 'auto',
                        'minHeight': '500px'
                    },
                    children=[
                        html.H3(
                            "Optimized Portfolio Weights",
                            style={
                                'color': COLORS['primary'],
                                'textAlign': 'center',
                                'marginBottom': '15px'
                            }
                        ),
                        dcc.Loading(
                            id="loading-optimized-weights",
                            type="circle",
                            children=[
                                html.Div(
                                    id="optimized-weights-table",
                                    style={
                                        'maxHeight': '200px',
                                        'overflowY': 'auto',
                                        'marginBottom': '20px'
                                    }
                                )
                            ]
                        ),
                        dcc.Graph(
                            id="weight-distribution-bar-chart",
                            style={'height': '300px'}
                        )
                    ]
                ),

                # Right Column: Efficient Frontier
                html.Div(
                    style={
                        'backgroundColor': COLORS['card'],
                        'borderRadius': '10px',
                        'padding': '20px',
                        'height': 'auto',
                        'minHeight': '500px'
                    },
                    children=[
                        html.H3(
                            "Efficient Frontier",
                            style={
                                'color': COLORS['primary'],
                                'textAlign': 'center',
                                'marginBottom': '15px'
                            }
                        ),
                        dcc.Graph(
                            id="efficient-frontier-plot",
                            style={'height': '500px'}
                        )
                    ]
                ),
            ]
        ),

        # Portfolio Performance Plot (Full Width)
        html.Div(
            style={
                'backgroundColor': COLORS['card'],
                'borderRadius': '10px',
                'padding': '20px',
                'marginTop': '20px'
            },
            children=[
                html.H3(
                    "Portfolio Performance Overview",
                    style={
                        'color': COLORS['primary'],
                        'textAlign': 'center',
                        'marginBottom': '15px'
                    }
                ),
                dcc.Graph(
                    id="portfolio-performance-plot",
                    config={'responsive': True},
                    style={'height': '600px'}
                )
            ]
        ),

        # Controls Section
        html.Div(
            style={
                'display': 'flex',
                'justifyContent': 'center',
                'marginTop': '20px',
                'gap': '20px'
            },
            children=[
                html.Button(
                    "Monitor Volatility using GARCH",
                    id="monitor-volatility-button",
                    style={
                        'backgroundColor': COLORS['primary'],
                        'color': COLORS['background'],
                        'padding': '12px 24px',
                        'border': 'none',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                        'fontWeight': 'bold',
                        'transition': 'all 0.3s ease'
                    }
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
        Output('optimized-weights-table', 'children'),
        Output('efficient-frontier-plot', 'figure'),
        Output('weight-distribution-bar-chart', 'figure'),
        Output('portfolio-performance-plot', 'figure'),
    ],
    [    
        Input('portfolio-tickers', 'data')
    ]
)
def update_portfolio_performance(tickers):

    # Fetch stock data and calculate expected returns & covariance
    results = get_expected_returns_covariance_matrix(tickers)
    expected_returns = results['expected_returns']
    cov_matrix = results['covariance_matrix']

    # Generate efficient frontier and find optimal weights
    volatilities, returns = generate_efficient_frontier(expected_returns, cov_matrix)
    optimal_weights = optimize_portfolio(expected_returns, cov_matrix)
    opt_return, opt_vol = portfolio_performance(optimal_weights, expected_returns, cov_matrix)

    # Calculate Minimum Variance Portfolio (MVP)
    min_var_weights = optimize_min_variance(cov_matrix)
    min_var_return, min_var_vol = portfolio_performance(min_var_weights, expected_returns, cov_matrix)

    # Create weights table
    weights_table = [
        html.Tr([
            html.Th("Ticker"),
            html.Th("Weight (%)"),
        ])
    ] + [
        html.Tr([
            html.Td(ticker),
            html.Td(f"{weight * 100:.2f}")
        ]) for ticker, weight in zip(tickers, optimal_weights)
    ]

    # Create efficient frontier plot
    fig_frontier = go.Figure()
    
    # Add portfolios to the frontier
    fig_frontier.add_trace(go.Scatter(
        x=volatilities,
        y=returns,
        mode='markers',
        name='Portfolios',
        marker=dict(size=5, color=returns / volatilities, colorscale='Viridis', showscale=True)
    ))

    # Add optimal portfolio
    fig_frontier.add_trace(go.Scatter(
        x=[opt_vol],
        y=[opt_return],
        mode='markers',
        name='Optimal Portfolio',
        marker=dict(size=15, color='red', symbol='star')
    ))

    # Add horizontal dashed line at the return of the MVP
    fig_frontier.add_shape(
        type="line",
        x0=min(volatilities),
        x1=max(volatilities),
        y0=min_var_return,
        y1=min_var_return,
        line=dict(
            color="red",
            dash="dash",
            width=2
        ),
        name="MVP Cut-off"
    )

    # Update layout
    fig_frontier.update_layout(
        template="plotly_dark",
        title="Efficient Frontier",
        xaxis_title="Volatility (Risk)",
        yaxis_title="Expected Return",
        height=600
    )

    # Create bar chart for weight distribution
    fig_weights = go.Figure(
        data=go.Bar(
            x=tickers,
            y=[weight * 100 for weight in optimal_weights],
            marker=dict(color='#8B5CF6')
        )
    )

    fig_weights.update_layout(
        template="plotly_dark",
        title="Optimized Weight Distribution by Ticker",
        xaxis_title="Tickers",
        yaxis_title="Weight (%)",
        height=400
    )

    # Fetch stock data and calculate expected returns & covariance
    results = get_expected_returns_covariance_matrix(tickers)
    expected_returns = results['expected_returns']
    cov_matrix = results['covariance_matrix']
    historical_prices = pd.DataFrame(results['data'])  # Assuming prices are included

    # Compute optimized portfolio weights
    optimal_weights = optimize_portfolio(expected_returns, cov_matrix)

    # Compute portfolio log returns
    log_returns = np.log(historical_prices / historical_prices.shift(1)).dropna()
    portfolio_log_returns = log_returns.dot(optimal_weights)

    # Calculate performance metrics
    mean_return = portfolio_log_returns.mean() * 252
    volatility = portfolio_log_returns.std() * np.sqrt(252)

    ####################
    ### Define Subplots
    ####################

    portfolio_fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=[
            "Portfolio Price with Bollinger Bands",
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

    # Calculate portfolio prices and Bollinger Bands
    portfolio_prices = (1 + portfolio_log_returns).cumprod()
    rolling_mean = portfolio_prices.rolling(window=20).mean()
    rolling_std = portfolio_prices.rolling(window=20).std()
    bollinger_upper = rolling_mean + 1.96 * rolling_std
    bollinger_lower = rolling_mean - 1.96 * rolling_std

    portfolio_fig.add_trace(
        go.Scatter(
            x=portfolio_prices.index,
            y=portfolio_prices,
            mode='lines',
            name='Portfolio Price',
            line=dict(color='#FFD700', width=2)
        ),
        row=1, col=1
    )
    portfolio_fig.add_trace(
        go.Scatter(
            x=portfolio_prices.index,
            y=bollinger_upper,
            mode='lines',
            name='Bollinger Upper',
            line=dict(color='white', dash='dot')
        ),
        row=1, col=1
    )
    portfolio_fig.add_trace(
        go.Scatter(
            x=portfolio_prices.index,
            y=bollinger_lower,
            mode='lines',
            name='Bollinger Lower',
            fill='tonexty',
            fillcolor='rgba(255, 255, 0, 0.2)',
            line=dict(color='white', dash='dot')
        ),
        row=1, col=1
    )

    #################################
    ### Line Plot for Log Returns
    ### with 95% Confidence Interval
    #################################

    lower_bound = mean_return - 1.96 * volatility / np.sqrt(len(portfolio_log_returns))
    upper_bound = mean_return + 1.96 * volatility / np.sqrt(len(portfolio_log_returns))

    portfolio_fig.add_trace(
        go.Scatter(
            x=portfolio_log_returns.index,
            y=portfolio_log_returns,
            mode='lines',
            name='Log Returns',
            line=dict(color='#FFD700', width=2)
        ),
        row=2, col=1
    )

    portfolio_fig.add_shape(
        type="rect",
        x0=min(portfolio_log_returns.index),
        x1=max(portfolio_log_returns.index),
        y0=lower_bound,
        y1=upper_bound,
        fillcolor="rgba(255, 255, 0, 0.2)",
        layer="below",
        line_width=0,
        row=2, col=1
    )

    ##############################
    ### Histogram of Log Returns
    ### with KDE + Gaussian Curve
    ##############################

    kde = gaussian_kde(portfolio_log_returns)
    x_kde = np.linspace(portfolio_log_returns.min(), portfolio_log_returns.max(), 500)
    y_kde = kde(x_kde)

    portfolio_fig.add_trace(
        go.Histogram(
            x=portfolio_log_returns,
            name='Log Return Distribution',
            histnorm='probability density',
            marker=dict(color='#FFD700'),
        ),
        row=2, col=2
    )
    portfolio_fig.add_trace(
        go.Scatter(
            x=x_kde,
            y=y_kde,
            mode='lines',
            name='KDE',
            line=dict(color='orange', width=2),
        ),
        row=2, col=2
    )

    portfolio_fig.update_layout(
        template="plotly_dark",
        title="Portfolio Performance (Last 2 Years)",
        title_font=dict(color='#FFD700'),
        font=dict(color='#FFD700'),
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=False
    )

    # Update figure styling for consistent theme
    for fig in [fig_frontier, fig_weights, portfolio_fig]:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS['card'],
            plot_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text']),
            title_font=dict(color=COLORS['primary']),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor=COLORS['primary'],
                borderwidth=1
            )
        )
        
        # Update axis styling
        fig.update_xaxes(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor=COLORS['primary']
        )
        fig.update_yaxes(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor=COLORS['primary']
        )

    # Style the weights table
    weights_table = [
        html.Tr(
            [html.Th("Ticker"), html.Th("Weight (%)")],
            style={
                'backgroundColor': COLORS['primary'],
                'color': COLORS['background']
            }
        )
    ] + [
        html.Tr(
            [html.Td(ticker), html.Td(f"{weight * 100:.2f}")],
            style={
                'backgroundColor': 'rgba(255,215,0,0.1)',
                'color': COLORS['text']
            }
        ) for ticker, weight in zip(tickers, optimal_weights)
    ]

    return weights_table, fig_frontier, fig_weights, portfolio_fig
