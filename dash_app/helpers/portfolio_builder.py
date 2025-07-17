
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
            "Sr. No.": i+1,
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
        {"name": "Sr. No.", "id": "Sr. No."},
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

def portfolio_optimization(cache_data):
    
    ### Maximum expected returns for given risk optimization
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

    ### Maximum sharpe optimization
    def neg_sharpe_ratio(weights):
        ret = np.dot(weights, mean_returns)
        vol = portfolio_volatility(weights, cov_matrix)
        return -ret / vol

    max_sharpe_result = minimize(
        neg_sharpe_ratio,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    )
    max_sharpe_vol = portfolio_volatility(max_sharpe_result.x, cov_matrix)
    max_sharpe_ret = np.dot(max_sharpe_result.x, mean_returns)
    
    ### Minimum variance optimization
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

    ### Minimum diversification ratio optimization
    def neg_diversification_ratio(weights, cov_matrix, std_devs):
        weighted_std = np.dot(weights, std_devs)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -weighted_std / portfolio_vol 

    mdp_result = minimize(
        neg_diversification_ratio,
        initial_weights,
        args=(cov_matrix, np.sqrt(np.diag(cov_matrix))),
        method='SLSQP',
        bounds=bounds,
        constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    )

    mdp_vol = portfolio_volatility(mdp_result.x, cov_matrix)
    mdp_ret = np.dot(mdp_result.x, mean_returns)

    ### Equal weights portfolio
    equal_weights = np.array([1 / num_assets] * num_assets)
    equal_weight_ret = np.dot(equal_weights, mean_returns)
    equal_weight_vol = portfolio_volatility(equal_weights, cov_matrix)

    return {
        "frontier_stddevs": frontier_stddevs,
        "frontier_returns": frontier_returns,
        "frontier_weights": frontier_weights,
        "max_sharpe": {"x": max_sharpe_vol, "y": max_sharpe_ret, "w": max_sharpe_result.x.tolist()},
        "min_var": {"x": min_var_vol, "y": min_var_ret, "w": min_var_result.x.tolist()},
        "mdp": {"x": mdp_vol, "y": mdp_ret, "w": mdp_result.x.tolist()},
        "equal": {"x": equal_weight_vol, "y": equal_weight_ret, "w": equal_weights.tolist()},
    }

def plot_efficient_frontier(max_sharpe_on, min_variance_on, min_diversification_on, equal_weights_on, 
                            optimization_dict, COLORS):
    # Extract data from the optimization dictionary
    frontier_stddevs = optimization_dict["frontier_stddevs"]
    frontier_returns = optimization_dict["frontier_returns"]
    frontier_weights = optimization_dict["frontier_weights"]

    max_sharpe = optimization_dict["max_sharpe"]
    min_var = optimization_dict["min_var"]
    mdp = optimization_dict["mdp"]
    equal = optimization_dict["equal"]

    # Initialize Plotly figure
    efficient_frontier_fig = go.Figure()

    # Efficient frontier line
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

    # Max Sharpe
    if max_sharpe_on:
        efficient_frontier_fig.add_trace(
            go.Scatter(
                x=[max_sharpe["x"]],
                y=[max_sharpe["y"]],
                mode='markers',
                marker=dict(size=12, color='red', symbol='x'),
                name='Maximum Sharpe Ratio',
                customdata=[max_sharpe["w"]],
                hovertemplate='Maximum Sharpe<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            )
        )

    # Min Variance
    if min_variance_on:
        efficient_frontier_fig.add_trace(
            go.Scatter(
                x=[min_var["x"]],
                y=[min_var["y"]],
                mode='markers',
                marker=dict(size=12, color='white', symbol='hexagram'),
                name='Minimum Variance',
                customdata=[min_var["w"]],
                hovertemplate='Minimum Variance<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            )
        )

    # Most Diversified Portfolio (MDP)
    if min_diversification_on:
        efficient_frontier_fig.add_trace(
            go.Scatter(
                x=[mdp["x"]],
                y=[mdp["y"]],
                mode='markers',
                marker=dict(size=12, color='lightblue', symbol='triangle-up'),
                name='Most Diversified Portfolio',
                customdata=[mdp["w"]],
                hovertemplate='Most Diversified<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            )
        )

    # Equal Weights
    if equal_weights_on:
        efficient_frontier_fig.add_trace(
            go.Scatter(
                x=[equal["x"]],
                y=[equal["y"]],
                mode='markers',
                marker=dict(size=12, color='orange', symbol='star'),
                name='Equal-Weight Portfolio',
                customdata=[equal["w"]],
                hovertemplate='Equal Weight<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            )
        )

    # Layout
    efficient_frontier_fig.update_layout(
        xaxis=dict(
            title='Risk (Standard Deviation of Daily Returns)',
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
        style={'height': '100%', 'width': '100%'}
    )
