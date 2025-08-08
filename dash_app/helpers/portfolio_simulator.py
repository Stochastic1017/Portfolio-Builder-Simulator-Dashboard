
import os
import sys
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go

from io import StringIO
from dash import dcc, html
from arch import arch_model
from datetime import timedelta
from plotly.subplots import make_subplots
from keras.optimizers import Adam # type: ignore
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras import backend as K # type: ignore
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Disable GPU

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

def parse_ts_map(
    selected_tickers, 
    portfolio_weights, 
    portfolio_store, 
    budget, 
    threshold=1e-6
):
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

def simulate_arima_paths(
    model_result, 
    last_date, 
    forecast_until, 
    num_ensembles, 
    inferred_freq='B'
):

    forecast_until = pd.to_datetime(forecast_until)
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(1, unit='D'), 
                                   end=forecast_until, freq=inferred_freq)
    n_steps = len(forecast_index)

    simulations = np.zeros((n_steps, num_ensembles))
    for i in range(num_ensembles):
        simulations[:, i] = model_result.simulate(nsimulations=n_steps)

    return (simulations, forecast_index, np.mean(simulations, axis=1), np.std(simulations, axis=1))

def grid_search_garch_models(
    returns,
    criterion='aic',
    p_range=range(1, 4),
    q_range=range(1, 4),
    model_types=('GARCH', 'EGARCH', 'GJR-GARCH'),
    distribution_types=('gaussian', 'studentst', 'skewstudent', 'ged')
):

    best_model = None
    best_order = None
    best_model_type = None
    best_distribution = None

    best_score = -np.inf if criterion == 'loglikelihood' else np.inf

    for model_type in model_types:
        for dist in distribution_types:
            for p in p_range:
                for q in q_range:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            model = arch_model(
                                returns,
                                vol=model_type.lower(),  # 'garch', 'egarch', 'gjr-garch'
                                p=p,
                                q=q,
                                dist=dist
                            )
                            result = model.fit(disp='off')

                            score = {
                                'aic': result.aic,
                                'bic': result.bic,
                                'loglikelihood': result.loglikelihood
                            }[criterion]

                            is_better = (
                                score > best_score if criterion == 'loglikelihood'
                                else score < best_score
                            )

                            if is_better:
                                best_score = score
                                best_model = result
                                best_order = (p, q)
                                best_model_type = model_type
                                best_distribution = dist

                    except Exception:
                        continue

    return {
        'model': best_model,
        'model_type': best_model_type,
        'order': best_order,
        'distribution': best_distribution,
        'score': best_score
    }

def simulate_garch_paths(
    model_result, 
    last_date, 
    forecast_until, 
    num_ensembles, 
    inferred_freq='B'
):

    forecast_until = pd.to_datetime(forecast_until)
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(1, unit='D'), 
                                   end=forecast_until, freq=inferred_freq)
    n_steps = len(forecast_index)

    simulations = np.zeros((n_steps, num_ensembles))
    
    for i in range(num_ensembles):
        sim_data = model_result.model.simulate(
            params=model_result.params,
            nobs=n_steps,
            x=None,  # required for ConstantMean
            initial_value=None,
            burn=250  # optional
        )
        simulations[:, i] = sim_data['data']

    return (simulations, forecast_index, np.mean(simulations, axis=1), np.std(simulations, axis=1))

def train_lstm_model(
    log_returns,
    lookback=20,
    epochs=50,
    batch_size=32,
    lstm_units=50,
    dropout_rate=0.2,
    learning_rate=0.001
):

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_returns = scaler.fit_transform(log_returns.values.reshape(-1, 1))

    # Prepare sequences
    X, y = [], []
    for i in range(len(scaled_returns) - lookback):
        X.append(scaled_returns[i:i+lookback])
        y.append(scaled_returns[i+lookback])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Clear any previous TF graphs
    K.clear_session()

    # Build model (CPU context)
    with tf.device('/CPU:0'):
        model = Sequential([
            LSTM(lstm_units, input_shape=(lookback, 1)),
            Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        final_loss = float(history.history['loss'][-1])

    return {
        'model': model,
        'scaler': scaler,
        'lookback': lookback,
        'units': lstm_units,
        'epochs': epochs,
        'loss': final_loss
    }

def simulate_lstm_paths(
    model_result, 
    last_date, 
    forecast_until, 
    num_ensembles, 
    historical_returns, 
    inferred_freq='B'
):

    forecast_until = pd.to_datetime(forecast_until)
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(1, unit='D'), 
                                   end=forecast_until, freq=inferred_freq)
    n_steps = len(forecast_index)

    model = model_result['model']
    scaler = model_result['scaler']
    lookback = model_result['lookback']

    # Prepare initial lookback window from historical data
    scaled_returns = scaler.transform(historical_returns.values.reshape(-1, 1))
    init_window = scaled_returns[-lookback:].reshape(1, lookback, 1)

    simulations = np.zeros((n_steps, num_ensembles))

    # Estimate residual std from training predictions
    X_train, y_train = [], []
    for i in range(len(scaled_returns) - lookback):
        X_train.append(scaled_returns[i:i+lookback])
        y_train.append(scaled_returns[i+lookback])
    X_train, y_train = np.array(X_train), np.array(y_train)
    preds_train = model.predict(X_train, verbose=0)
    residual_std = np.std(y_train - preds_train)

    for i in range(num_ensembles):
        window = init_window.copy()
        sim_path = []
        for _ in range(n_steps):
            pred_scaled = model.predict(window, verbose=0)[0, 0]
            # Add Gaussian noise from residual distribution
            pred_scaled += np.random.normal(0, residual_std)
            sim_path.append(pred_scaled)
            # Update window
            window = np.roll(window, -1, axis=1)
            window[0, -1, 0] = pred_scaled

        # Inverse scale to returns
        sim_returns = scaler.inverse_transform(np.array(sim_path).reshape(-1, 1)).flatten()
        simulations[:, i] = sim_returns

    return (
        simulations,
        forecast_index,
        np.mean(simulations, axis=1),
        np.std(simulations, axis=1)
    )

def simulation_plot(
        model_used,
        title, 
        dates, 
        daily_prices, 
        log_returns, 
        simulations, 
        forecast_index, 
        mean_ensembles,
        std_ensembles, 
        COLORS
):
    last_price = daily_prices[-1]
    _, num_ensembles = simulations.shape

    # Convert cumulative returns to price paths
    price_paths = np.zeros_like(simulations)
    for i in range(num_ensembles):
        cum_ret = np.cumsum(simulations[:, i])
        price_paths[:, i] = last_price * np.exp(cum_ret)

    mean_price_path = price_paths.mean(axis=1)
    std_price_path = price_paths.std(axis=1)

    lower_price_bound = mean_price_path - 1.96 * std_price_path
    upper_price_bound = mean_price_path + 1.96 * std_price_path

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Simulated Price Paths (GARCH)",
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
        
    fig.add_trace(go.Scatter(
        x=[dates[-1]] + list(forecast_index),
        y=[daily_prices[-1]] + list(mean_price_path),
        mode='lines',
        name='Mean Simulated Price',
        line=dict(color='#ED3500', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[dates[-1]] + list(forecast_index) + list(forecast_index[::-1]) + [dates[-1]],
        y=[daily_prices[-1]] + list(upper_price_bound) + list(lower_price_bound[::-1]) + [daily_prices[-1]],
        fill='toself',
        fillcolor='rgba(255, 87, 34, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=log_returns, mode='lines', name='Log Returns',
                             line=dict(color=COLORS['primary'])), row=2, col=1)

    for i in range(num_ensembles):
        fig.add_trace(go.Scatter(x=[log_returns.index[-1]] + list(forecast_index),
                                 y=[log_returns.iloc[-1]] + list(simulations[:, i]),
                                 mode='lines', line=dict(width=1, color='rgba(255,255,255,0.1)'),
                                 showlegend=False), row=2, col=1)

    lower_return_bound = mean_ensembles - 1.96 * std_ensembles
    upper_return_bound = mean_ensembles + 1.96 * std_ensembles

    fig.add_trace(go.Scatter(
        x=[log_returns.index[-1]] + list(forecast_index),
        y=[log_returns.iloc[-1]] + list(mean_ensembles),
        mode='lines',
        name='Mean Simulated Return',
        line=dict(color='#ED3500', width=2)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[log_returns.index[-1]] + list(forecast_index) + list(forecast_index[::-1]) + [log_returns.index[-1]],
        y=[log_returns.iloc[-1]] + list(upper_return_bound) + list(lower_return_bound[::-1]) + [log_returns.iloc[-1]],
        fill='toself',
        fillcolor='rgba(255, 87, 34, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ), row=2, col=1)

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

    return dcc.Graph(id=f"{model_used}-simulation-plot", figure=fig, config={'responsive': True}, style={'height': '100%', 'width': '100%'})
