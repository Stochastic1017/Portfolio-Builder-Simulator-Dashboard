
import os
import sys
import numpy as np
import pandas as pd
import dash.dash_table as dt
import plotly.graph_objects as go

from dash import dcc
from io import StringIO
from scipy.optimize import minimize 

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def summary_table(cache_data, COLORS):
    # Prepare rows for the table
    table_data = []
    for entry in cache_data:
        hist_df = pd.read_json(StringIO(entry["historical_json"]), orient="records")
        hist_df['date'] = pd.to_datetime(hist_df['date'])

        # Get latest price from most recent record
        latest_price = hist_df['close'].iloc[-1]
        table_data.append({
            "Ticker": entry["ticker"],
            "Full Name": entry["fullname"],
            "Latest Price": latest_price,
            "SIC": entry["sic_description"],
            "Market Cap": entry["market_cap"]
        })

    # Column formatting and properties
    columns = [
        {"name": "Ticker", "id": "Ticker", "type": "text"},
        {"name": "Full Name", "id": "Full Name", "type": "text"},
        {"name": "SIC", "id": "SIC", "type": "text"},
        {"name": "Latest Price", "id": "Latest Price", "type": "numeric", "format": dt.FormatTemplate.money(0)},
        {"name": "Market Cap", "id": "Market Cap", "type": "numeric", "format": dt.FormatTemplate.money(0)},
    ]

    # Create DataTable
    summary_table = dt.DataTable(
        data=table_data,
        columns=columns,
        style_table={
            "maxHeight": "200px",
            "overflowY": "auto",
            "border": "1px solid #444"
        },
        style_cell={
            "padding": "10px",
            "backgroundColor": COLORS["background"],
            "color": COLORS["text"],
            "border": "1px solid #555",
            "textAlign": "left",
            "minWidth": "100px",
            "maxWidth": "200px",
            "whiteSpace": "normal"
        },
        style_header={
            "backgroundColor": COLORS["card"],
            "color": COLORS["primary"],
            "fontWeight": "bold"
        },
        row_deletable=False,
        sort_action='native'
    )

    return summary_table

def negative_sharpe_ratio(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -ret / std if std != 0 else 0

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

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

    combined_returns = pd.concat(returns_list, axis=1)
    combined_returns.columns = tickers
    combined_returns.dropna(inplace=True)

    mean_returns = combined_returns.mean()
    cov_matrix = combined_returns.cov()
    num_assets = len(tickers)
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 100)
    frontier_returns = []
    frontier_stddevs = []

    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}
        )

        result = minimize(
            portfolio_volatility,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            frontier_stddevs.append(portfolio_volatility(result.x, cov_matrix))
            frontier_returns.append(target_return)

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    )

    max_sharpe_std = np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)))
    max_sharpe_return = np.dot(result.x, mean_returns)

    # Min variance portfolio
    min_var_idx = np.argmin(frontier_stddevs)
    min_var_std = frontier_stddevs[min_var_idx]
    min_var_return = frontier_returns[min_var_idx]

    ##########################
    ### Initialize Plotly Figure
    ##########################

    efficient_frontier_fig = go.Figure()

    # Efficient Frontier Line
    efficient_frontier_fig.add_trace(
        go.Scatter(
            x=frontier_stddevs,
            y=frontier_returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color=COLORS['primary'], width=2),
            fill='tozeroy',
            fillcolor="rgba(255,255,255,0.05)",
            hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
        )
    )

    # Max Sharpe Point
    efficient_frontier_fig.add_trace(
        go.Scatter(
            x=[max_sharpe_std],
            y=[max_sharpe_return],
            mode='markers+text',
            marker=dict(color='yellow', size=12, symbol='star'),
            name='Max Sharpe',
            text=["Max Sharpe"],
            textposition="top center"
        )
    )

    # Min Variance Line (horizontal line clipped to frontier end)
    efficient_frontier_fig.add_trace(
        go.Scatter(
            x=[min_var_std, frontier_stddevs[-1]],
            y=[min_var_return, min_var_return],
            mode='lines',
            name='Min Variance Line',
            line=dict(color='white', dash='dot'),
            hoverinfo='skip'
        )
    )

    #####################
    ### Layout Styling
    #####################

    efficient_frontier_fig.update_layout(
        title="Efficient Frontier",
        xaxis=dict(
            title='Risk (Standard Deviation of Returns)',
            color=COLORS['text'],
            showgrid=True,
            gridcolor='rgba(0, 130, 180, 0.2)'
        ),
        yaxis=dict(
            title='Expected Annual Return',
            color=COLORS['text'],
            showgrid=True,
            gridcolor='rgba(0, 130, 180, 0.2)'
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        title_font=dict(color=COLORS['primary']),
        showlegend=False
    )

    return dcc.Graph(
        id="efficient-frontier-graph",
        figure=efficient_frontier_fig,
        config={'responsive': True},
        style={'height': '100%', 'width': '100%'}
    )
