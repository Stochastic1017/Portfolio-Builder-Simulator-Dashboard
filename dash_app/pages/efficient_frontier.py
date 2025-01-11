from dash import html, dcc, callback, Input, Output, State
import dash
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px

dash.register_page(__name__, path="/efficient-frontier")

layout = html.Div(
    style={
        'background': 'linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%)',
        'minHeight': '100vh',
        'padding': '20px',
        'color': '#FFFFFF',
    },
    children=[
        html.H1(
            "Efficient Frontier Analysis",
            style={
                'color': '#8B5CF6',
                'fontSize': '2em',
                'textAlign': 'center',
                'marginBottom': '20px'
            }
        ),
        html.Div(
            id="portfolio-ticker-display",
            style={
                'backgroundColor': '#1E1E1E',
                'padding': '15px',
                'borderRadius': '10px',
                'marginBottom': '20px'
            }
        ),
        dcc.Graph(
            id="efficient-frontier-plot",
            style={
                'backgroundColor': '#1E1E1E',
                'borderRadius': '10px',
                'padding': '15px'
            }
        )
    ]
)

def get_stock_data(tickers, period='2y'):
    """Fetch stock data and calculate returns."""
    data = pd.DataFrame()
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if not hist.empty:
            data[ticker] = hist['Close']
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Calculate annualized expected returns and covariance
    expected_returns = (1 + returns.mean()) ** 252 - 1
    cov_matrix = returns.cov() * 252
    
    return expected_returns, cov_matrix

def portfolio_performance(weights, expected_returns, cov_matrix):
    """Calculate portfolio return and volatility."""
    portfolio_return = np.sum(expected_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.02):
    """Calculate negative Sharpe ratio for minimization."""
    portfolio_return, portfolio_volatility = portfolio_performance(weights, expected_returns, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate=0.02):
    """Find optimal portfolio weights."""
    num_assets = len(expected_returns)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
    )
    bounds = tuple((0, 1) for _ in range(num_assets))  # weights between 0 and 1
    
    # Initial guess (equal weights)
    initial_weights = np.array([1/num_assets] * num_assets)
    
    # Optimize!
    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(expected_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

def generate_efficient_frontier(expected_returns, cov_matrix, num_portfolios=1000):
    """Generate efficient frontier points."""
    num_assets = len(expected_returns)
    returns = []
    volatilities = []
    
    for _ in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio performance
        portfolio_return, portfolio_volatility = portfolio_performance(weights, expected_returns, cov_matrix)
        returns.append(portfolio_return)
        volatilities.append(portfolio_volatility)
    
    return np.array(volatilities), np.array(returns)

@callback(
    [
        Output('efficient-frontier-plot', 'figure'),
        Output('portfolio-ticker-display', 'children')
    ],
    [Input('portfolio-tickers', 'data')]
)
def update_efficient_frontier(tickers):
    if not tickers or len(tickers) < 2:
        return (
            go.Figure().add_annotation(
                text="Need at least 2 tickers for portfolio optimization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            ),
            "Please add at least 2 tickers to your portfolio"
        )
    
    try:
        # Get stock data
        expected_returns, cov_matrix = get_stock_data(tickers)
        
        # Generate efficient frontier points
        volatilities, returns = generate_efficient_frontier(expected_returns, cov_matrix)
        
        # Find optimal portfolio
        optimal_weights = optimize_portfolio(expected_returns, cov_matrix)
        opt_return, opt_vol = portfolio_performance(optimal_weights, expected_returns, cov_matrix)
        
        # Create figure
        fig = go.Figure()
        
        # Add efficient frontier scatter plot
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=returns,
                mode='markers',
                name='Possible Portfolios',
                marker=dict(
                    size=5,
                    color=returns/volatilities,  # Color by Sharpe ratio
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Sharpe Ratio')
                )
            )
        )
        
        # Add optimal portfolio point
        fig.add_trace(
            go.Scatter(
                x=[opt_vol],
                y=[opt_return],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star'
                )
            )
        )
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            title="Efficient Frontier",
            xaxis_title="Expected Volatility",
            yaxis_title="Expected Return",
            showlegend=True,
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        # Create portfolio summary
        portfolio_summary = html.Div([
            html.H3("Optimal Portfolio Weights:"),
            html.Ul([
                html.Li(f"{ticker}: {weight:.2%}")
                for ticker, weight in zip(tickers, optimal_weights)
            ]),
            html.P(f"Expected Annual Return: {opt_return:.2%}"),
            html.P(f"Expected Annual Volatility: {opt_vol:.2%}"),
            html.P(f"Sharpe Ratio: {(opt_return - 0.02) / opt_vol:.2f}")
        ])
        
        return fig, portfolio_summary
        
    except Exception as e:
        return (
            go.Figure().add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            ),
            f"Error occurred: {str(e)}"
        )
