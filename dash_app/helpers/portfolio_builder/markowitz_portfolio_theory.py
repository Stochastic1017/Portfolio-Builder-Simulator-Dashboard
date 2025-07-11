
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

def summary_table(cache_data, COLORS, weights=None, budget=None):
    table_data = []

    for i, entry in enumerate(cache_data):
        hist_df = pd.read_json(StringIO(entry["historical_json"]), orient="records")
        hist_df['date'] = pd.to_datetime(hist_df['date'])

        latest_price = hist_df['close'].iloc[-1]
        
        weight_pct = weights[i] if weights else 0
        dollar_alloc = weight_pct * budget if budget else 0
        shares = dollar_alloc / latest_price if latest_price else 0

        table_data.append({
            "Ticker": entry["ticker"],
            "Full Name": entry["fullname"],
            "SIC": entry["sic_description"],
            "Latest Price": latest_price,
            "Market Cap": entry["market_cap"],
            "Weight (%)": round(weight_pct * 100, 2),
            "Weight ($)": round(dollar_alloc, 2),
            "Shares to Buy": round(shares, 3)
        })

    columns = [
        {"name": "Ticker", "id": "Ticker"},
        {"name": "Full Name", "id": "Full Name"},
        {"name": "SIC", "id": "SIC"},
        {"name": "Latest Price", "id": "Latest Price", "type": "numeric", "format": dt.FormatTemplate.money(0)},
        {"name": "Market Cap", "id": "Market Cap", "type": "numeric", "format": dt.FormatTemplate.money(0)},
        {"name": "Weight (%)", "id": "Weight (%)", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "Weight ($)", "id": "Weight ($)", "type": "numeric", "format": dt.FormatTemplate.money(0)},
        {"name": "Shares to Buy", "id": "Shares to Buy", "type": "numeric", "format": {"specifier": ".3f"}},
    ]

    return dt.DataTable(
        data=table_data,
        columns=columns,
        style_table={
            "maxHeight": "400px",
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
        fixed_rows={"headers": True},
        row_deletable=False,
        sort_action='native'
    )

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def plot_efficient_frontier(max_sharpe_on, min_variance_on, equal_weights_on, cache_data, COLORS):
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
    frontier_weights = []

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
            frontier_stddevs.append(portfolio_volatility(result.x, 
                                                         cov_matrix))
            frontier_returns.append(target_return)
            frontier_weights.append(result.x)

    # Max Sharpe
    def neg_sharpe_ratio(weights):
        ret = np.dot(weights, mean_returns)
        vol = portfolio_volatility(weights, cov_matrix)
        return -ret / vol

    # Maximum sharpe optimization
    max_sharpe_result = minimize(
    neg_sharpe_ratio,
    initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    )
    max_sharpe_vol = portfolio_volatility(max_sharpe_result.x, cov_matrix)
    max_sharpe_ret = np.dot(max_sharpe_result.x, mean_returns)
    
    # Minimum variance optimization
    min_var_result = minimize(
        portfolio_volatility,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    )
    min_var_vol = portfolio_volatility(min_var_result.x, cov_matrix)
    min_var_ret = np.dot(min_var_result.x, mean_returns)

    # Equal weights portfolio
    equal_weights = np.array([1 / num_assets] * num_assets)
    equal_weight_ret = np.dot(equal_weights, mean_returns)
    equal_weight_vol = portfolio_volatility(equal_weights, cov_matrix)

    #############################
    ### Initialize Plotly Figure
    #############################

    efficient_frontier_fig = go.Figure()

    # Efficient Frontier Line
    efficient_frontier_fig.add_trace(
        go.Scatter(
            x=frontier_stddevs,
            y=frontier_returns,
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color=COLORS['primary'], width=2),
            fill='tozeroy',
            fillcolor="rgba(255,255,255,0.05)",
            hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>',
            customdata=frontier_weights,
        )
    )

    ####################
    ### Maximum Sharpe
    ### Toggle Button
    ####################

    # Maximum sharpe ratio portfolio
    if max_sharpe_on:

        # Plot maximum sharpe portfolio
        efficient_frontier_fig.add_trace(
            go.Scatter(
                x=[max_sharpe_vol],
                y=[max_sharpe_ret],
                mode='markers',
                marker=dict(size=12, color='red', symbol='x'),
                name='Maximum Sharpe Ratio',
                customdata=[max_sharpe_result.x],
                hovertemplate='Maximum Sharpe<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            )
        )

    #####################
    ### Minimum Variance
    ### Toggle Button
    #####################

    # Minimum variance portfolio
    if min_variance_on:
        
        # Plot minimum variance portfolio
        efficient_frontier_fig.add_trace(
            go.Scatter(
                x=[min_var_vol],
                y=[min_var_ret],
                mode='markers',
                marker=dict(size=12, color='white', symbol='hexagram'),
                name='Miimum Variance',
                customdata=[min_var_result.x],
                hovertemplate='Minimum Variance<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            )
        )

    #####################
    ### Equal Weights
    ### Toggle Button
    #####################

    # Equal weights portfolio
    if equal_weights_on:
        
        # Plot minimum variance portfolio
        efficient_frontier_fig.add_trace(
            go.Scatter(
                x=[equal_weight_vol],
                y=[equal_weight_ret],
                mode='markers',
                marker=dict(size=12, color='orange', symbol='star'),
                name='Equal-Weight Portfolio',
                customdata=[equal_weights],
                hovertemplate='Equal Weight<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            )
        )

    #####################
    ### Layout Styling
    #####################

    efficient_frontier_fig.update_layout(
        xaxis=dict(
            title='Risk (Standard Deviation) of Daily Returns)',
            color=COLORS['text'],
            tickformat=".2%",
            range=[
            min(frontier_stddevs) - 0.0005,
            max(frontier_stddevs) + 0.0005],
            showgrid=True,
            gridcolor='rgba(0, 130, 180, 0.2)'
        ),
        yaxis=dict(
            title='Expected (Mean) Daily Return',
            color=COLORS['text'],
            tickformat=".2%",
            range=[
            min(frontier_returns) - 0.0005,
            max(frontier_returns) + 0.0005],
            showgrid=True,
            gridcolor='rgba(0, 130, 180, 0.2)'
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        showlegend=False
    )

    return dcc.Graph(
        id="efficient-frontier-graph",
        figure=efficient_frontier_fig,
        config={'responsive': True},
        style={'height': '100%', 'width': '100%'})
