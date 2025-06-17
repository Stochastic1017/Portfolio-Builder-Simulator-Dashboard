
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import dcc
from io import StringIO

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def plot_efficient_frontier(cache_data, COLORS):

    tickers = []
    returns_list = []
    for entry in cache_data:
        df = pd.read_json(StringIO(entry["historical_json"]), orient="records")
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        returns = df['close'].pct_change().dropna()
        tickers.append(entry['ticker'])
        returns_list.append(returns)

    # Combine into a DataFrame
    combined_returns = pd.concat(returns_list, axis=1)
    combined_returns.columns = tickers
    combined_returns.dropna(inplace=True)

    mean_returns = combined_returns.mean()
    cov_matrix = combined_returns.cov()

    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        results[0, i] = portfolio_std
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return / portfolio_std)  # Sharpe (assuming rf=0)

    # Plot
    frontier = go.Scatter(
        x=results[0],
        y=results[1],
        mode='markers',
        marker=dict(
            color=results[2],
            colorscale='Viridis',
            size=4,
            colorbar=dict(title='Sharpe Ratio')
        ),
        name='Random Portfolios'
    )

    layout = go.Layout(
        title="Efficient Frontier",
        xaxis=dict(title='Risk (Standard Deviation)', color=COLORS['text']),
        yaxis=dict(title='Expected Return', color=COLORS['text']),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=500
    )

    fig = go.Figure(data=[frontier], layout=layout)

    return dcc.Graph(figure=fig)