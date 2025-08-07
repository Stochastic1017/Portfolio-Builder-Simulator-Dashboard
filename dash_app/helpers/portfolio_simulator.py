
import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from io import StringIO
from dash import dcc, html
from arch import arch_model
from datetime import timedelta
from pmdarima.arima import auto_arima
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def portfolio_dash_range_selector(default_style):
    
    return html.Div(
        id="portfolio-simulator-range-selector-container",
        children=[
            html.Div(
                style={"display": "flex", 
                       "justifyContent": "start", 
                       "flexWrap": "wrap",
                       "gap": "10px",
                       "paddingTop": "10px", 
                       "marginBottom": "10px"},

                children=[
                    html.Button("1M", id="portfolio-range-1M", n_clicks=0, style=default_style, className="simple"),
                    html.Button("3M", id="portfolio-range-3M", n_clicks=0, style=default_style, className="simple"),
                    html.Button("6M", id="portfolio-range-6M", n_clicks=0, style=default_style, className="simple"),
                    html.Button("1Y", id="portfolio-range-1Y", n_clicks=0, style=default_style, className="simple"),
                    html.Button("5Y", id="portfolio-range-5Y", n_clicks=0, style=default_style, className="simple"),
                    html.Button("All", id="portfolio-range-all", n_clicks=0, style=default_style, className="simple"),
                ],

            )
        ]
    )

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

def grid_search_arima_model(
    y,
    criterion='aic',             # Options: 'aic', 'bic', 'loglikelihood'
    p_range=range(0, 4),
    d_range=range(0, 3),
    q_range=range(0, 4)
):
    criterion = criterion.lower()
    if criterion not in ['aic', 'bic', 'loglikelihood']:
        raise ValueError("criterion must be one of: 'aic', 'bic', or 'loglikelihood'")

    best_score = -np.inf if criterion == 'loglikelihood' else np.inf
    best_model = None
    best_order = None

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)

                        model = ARIMA(y, order=(p, d, q))
                        result = model.fit()

                    score = {
                        'aic': result.aic,
                        'bic': result.bic,
                        'loglikelihood': result.llf
                    }[criterion]

                    is_better = (
                        score > best_score if criterion == 'loglikelihood'
                        else score < best_score
                    )

                    if is_better:
                        best_score = score
                        best_model = result
                        best_order = (p, d, q)

                except Exception:
                    continue

    return {
        'score': best_score,
        'result': best_model,
        'order': best_order
    }


def simulate_arima_paths(model_result, last_date, forecast_until, num_ensembles, inferred_freq='B'):
    forecast_until = pd.to_datetime(forecast_until)
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(1, unit='D'), 
                                   end=forecast_until, freq=inferred_freq)
    n_steps = len(forecast_index)

    simulations = np.zeros((n_steps, num_ensembles))
    for i in range(num_ensembles):
        simulations[:, i] = model_result.simulate(nsimulations=n_steps)

    return simulations, forecast_index

def arima_simulation_plot(title, dates, daily_prices, log_returns, simulations, forecast_index, COLORS):
    last_price = daily_prices[-1]
    _, num_ensembles = simulations.shape

    # Convert cumulative returns to price paths
    price_paths = np.zeros_like(simulations)
    for i in range(num_ensembles):
        cum_ret = np.cumsum(simulations[:, i])
        price_paths[:, i] = last_price * np.exp(cum_ret)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Simulated Price Paths (ARIMA)",
            "Simulated Log Returns"
        ],
        shared_xaxes=True,
        vertical_spacing=0.15,
    )

    fig.add_trace(go.Scatter(x=dates, y=daily_prices, mode='lines', name='Historical Price',
                             line=dict(color=COLORS['primary'])), row=1, col=1)

    for i in range(num_ensembles):
        fig.add_trace(go.Scatter(x=[dates[-1]] + list(forecast_index),
                                 y=[daily_prices[-1]] + list(price_paths[:, i]),
                                 mode='lines', line=dict(width=1, color='rgba(255,255,255,0.1)'),
                                 showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=log_returns, mode='lines', name='Log Returns',
                             line=dict(color=COLORS['primary'])), row=2, col=1)

    for i in range(num_ensembles):
        fig.add_trace(go.Scatter(x=[log_returns.index[-1]] + list(forecast_index),
                                 y=[log_returns.iloc[-1]] + list(simulations[:, i]),
                                 mode='lines', line=dict(width=1, color='rgba(255,255,255,0.1)'),
                                 showlegend=False), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        title=title,
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        title_font=dict(color=COLORS['primary']),
        showlegend=False,
        xaxis2=dict(
            rangeslider=dict(visible=True),
            type='date',
            range=[dates[-1] - timedelta(days=30), forecast_index[-1]]
        ),
    )

    fig.update_yaxes(tickformat=".2%", row=2, col=1)

    return dcc.Graph(id="arima-simulation-plot", figure=fig, config={'responsive': True}, style={'height': '100%', 'width': '100%'})

def select_best_garch_model(
    returns,
    p_range=range(1, 4),
    q_range=range(1, 4),
    criterions=('AIC', 'BIC', 'LogLikelihood'),
    dists=('normal', 't', 'skewt')
):
    """
    Evaluate GARCH(p,q) models across different error distributions and selection criteria.

    Returns:
        best_models: dict of {criterion: (result, (p,q), dist, score)}
    """
    best_models = {criterion: {'score': np.inf if criterion != 'LogLikelihood' else -np.inf,
                               'result': None,
                               'order': None,
                               'dist': None}
                   for criterion in criterions}

    for p in p_range:
        for q in q_range:
            for dist in dists:
                try:
                    model = arch_model(returns, vol='Garch', p=p, q=q, dist=dist)
                    res = model.fit(disp='off')
                    
                    scores = {
                        'AIC': res.aic,
                        'BIC': res.bic,
                        'LogLikelihood': res.loglikelihood
                    }

                    for criterion in criterions:
                        if criterion == 'LogLikelihood':
                            if scores[criterion] > best_models[criterion]['score']:
                                best_models[criterion] = {
                                    'score': scores[criterion],
                                    'result': res,
                                    'order': (p, q),
                                    'dist': dist
                                }
                        else:
                            if scores[criterion] < best_models[criterion]['score']:
                                best_models[criterion] = {
                                    'score': scores[criterion],
                                    'result': res,
                                    'order': (p, q),
                                    'dist': dist
                                }
                except Exception as e:
                    # Some combinations may not converge; skip silently
                    continue

    return best_models

