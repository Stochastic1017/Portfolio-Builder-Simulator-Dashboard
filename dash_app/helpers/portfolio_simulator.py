
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from io import StringIO
from dash import dcc

def plot_portfolio_value(selected_tickers, portfolio_weights, portfolio_store, threshold=1e-6):
    ts_map = {}
    selected_values = {t['value'] for t in selected_tickers}
    total_value = None

    for entry, weight in zip(portfolio_store, portfolio_weights):
        if weight <= threshold:
            continue
        if entry['ticker'] not in selected_values:
            continue

        df = pd.read_json(StringIO(entry['historical_json']), orient="records")
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df.set_index('date', inplace=True)

        price_series = df['close'].sort_index()
        ts_map[entry['ticker']] = {"weight": weight, "price": price_series}

        weighted = price_series * weight
        total_value = weighted if total_value is None else total_value.add(weighted, fill_value=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=total_value.index,
        y=total_value.values,
        mode="lines",
        name="Weighted Portfolio"
    ))

    fig.update_layout(
        title="Simulated Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value (relative)",
        template="plotly_dark"
    )

    return dcc.Graph(
        id="portfolio-performance-plot",
        figure=fig,
        config={'responsive': True},
        style={'height': '100%', 'width': '100%'}
    )
