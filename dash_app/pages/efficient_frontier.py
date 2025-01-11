
import os
import sys
import dash

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output
from helpers.portfolio_optimization import optimize_portfolio, generate_efficient_frontier, portfolio_performance, get_expected_returns_covariance_matrix

dash.register_page(__name__, path="/efficient-frontier")

layout = html.Div(
    style={
        'background': 'linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%)',
        'minHeight': '100vh',
        'padding': '20px',
        'color': '#FFFFFF',
        'fontFamily': '"Inter", system-ui, -apple-system, sans-serif'
    },
    children=[
        # Page Header
        html.H1(
            "Efficient Frontier Analysis",
            style={
                'color': '#8B5CF6',
                'fontSize': '2.5em',
                'textAlign': 'center',
                'marginBottom': '30px',
                'fontWeight': 'bold',
            }
        ),

        # Section to display optimized weights in a table
        html.Div(
            style={
                'backgroundColor': '#1E1E1E',
                'padding': '20px',
                'borderRadius': '10px',
                'marginBottom': '20px',
            },
            children=[
                html.H3(
                    "Optimized Portfolio Weights",
                    style={'color': '#8B5CF6', 'textAlign': 'center'}
                ),
                dcc.Loading(
                    id="loading-optimized-weights",
                    type="circle",
                    children=[
                        html.Table(
                            id="optimized-weights-table",
                            style={
                                'width': '100%',
                                'borderSpacing': '10px',
                                'color': '#FFFFFF',
                                'fontSize': '1em',
                                'textAlign': 'center',
                            }
                        )
                    ]
                )
            ]
        ),

        # Section to display efficient frontier plot
        html.Div(
            style={
                'backgroundColor': '#1E1E1E',
                'padding': '20px',
                'borderRadius': '10px',
            },
            children=[
                html.H3(
                    "Efficient Frontier",
                    style={'color': '#8B5CF6', 'textAlign': 'center'}
                ),
                dcc.Graph(
                    id="efficient-frontier-plot",
                    style={
                        'backgroundColor': '#1E1E1E',
                        'borderRadius': '10px',
                    },
                    config={'responsive': True}
                )
            ]
        ),

        html.Div(
        style={
            'marginTop': '20px',
            'textAlign': 'center',
            'display': 'flex',
            'justifyContent': 'space-around',
        },
        children=[
            html.Button(
                "Historical Simulation",
                id="historical-var-button",
                style={
                    'backgroundColor': '#8B5CF6',
                    'color': 'white',
                    'padding': '10px 20px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold',
                }
            ),
            html.Button(
                "Delta-Normal Method",
                id="delta-normal-var-button",
                style={
                    'backgroundColor': '#8B5CF6',
                    'color': 'white',
                    'padding': '10px 20px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold',
                }
            ),
            html.Button(
                "Monte Carlo Method",
                id="monte-carlo-var-button",
                style={
                    'backgroundColor': '#8B5CF6',
                    'color': 'white',
                    'padding': '10px 20px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold',
                }
            ),
        ]
    ),

        # Footer
        html.Footer(
            id="footer-section",
            style={
                'backgroundColor': '#0A0A0A',
                'padding': '10px',
                'textAlign': 'center',
                'color': '#B3B3B3',
                'fontSize': '0.9rem',
                'marginTop': '20px',
                'borderTop': '1px solid #282828',
            },
            children=[
                html.P("Developed by Shrivats Sudhir | Contact: stochastic1017@gmail.com"),
                html.P(
                    [
                        "GitHub Repository: ",
                        html.A(
                            "Portfolio Optimization Dashboard",
                            href="https://github.com/Stochastic1017/Portfolio-Optimization",
                            target="_blank",
                            style={'color': '#8B5CF6', 'textDecoration': 'none'}
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
        Output('efficient-frontier-plot', 'figure')
    ],
    Input('portfolio-tickers', 'data')  # Retrieve selected tickers from the first page
)
def update_efficient_frontier(tickers):
    if not tickers or len(tickers) < 2:
        return (
            "No tickers in your portfolio. Add at least 2 to proceed.",
            [],
            go.Figure().add_annotation(
                text="Add at least 2 tickers to visualize the efficient frontier.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        )

    # Fetch stock data and calculate expected returns & covariance
    results = get_expected_returns_covariance_matrix(tickers)
    expected_returns = results['expected_returns']
    cov_matrix = results['covariance_matrix']

    # Generate efficient frontier and find optimal weights
    volatilities, returns = generate_efficient_frontier(expected_returns, cov_matrix)
    optimal_weights = optimize_portfolio(expected_returns, cov_matrix)
    opt_return, opt_vol = portfolio_performance(optimal_weights, expected_returns, cov_matrix)

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
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=volatilities,
        y=returns,
        mode='markers',
        name='Portfolios',
        marker=dict(size=5, color=returns / volatilities, colorscale='Viridis', showscale=True)
    ))
    fig.add_trace(go.Scatter(
        x=[opt_vol],
        y=[opt_return],
        mode='markers',
        name='Optimal Portfolio',
        marker=dict(size=15, color='red', symbol='star')
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Efficient Frontier",
        xaxis_title="Volatility (Risk)",
        yaxis_title="Expected Return",
        height=600
    )

    return weights_table, fig

