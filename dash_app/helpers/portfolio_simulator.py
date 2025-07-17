
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from io import StringIO
from scipy import stats
from dash import dcc, html
from plotly.subplots import make_subplots

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_ts_map(selected_tickers, portfolio_weights, portfolio_store, budget, threshold=1e-6):
    ts_map = {}
    selected_values = {t['value'] for t in selected_tickers}
    total_value = None

    for entry, weight in zip(portfolio_store, portfolio_weights):
        if weight <= threshold:
            continue
        if entry['ticker'] not in selected_values:
            continue

        df = pd.read_json(StringIO(entry["historical_json"]), orient="records")
        df['date'] = pd.to_datetime(df['date'], unit='ms' if isinstance(df['date'].iloc[0], (int, float)) else None)
        df.set_index('date', inplace=True)

        price_series = df['close'].sort_index()

        # Use budget to get actual dollar allocation and number of shares
        latest_price = price_series.iloc[-1]
        dollar_alloc = weight * budget
        shares = dollar_alloc / latest_price if latest_price > 0 else 0

        # Dollar value of the ticker over time
        dollar_ts = price_series * shares

        ts_map[entry['ticker']] = {
            "weight": weight,
            "shares": shares,
            "price": price_series,
            "value": dollar_ts,
        }

        total_value = dollar_ts if total_value is None else total_value.add(dollar_ts, fill_value=0)

    return ts_map, total_value
