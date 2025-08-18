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
from keras.optimizers import Adam  # type: ignore
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential  # type: ignore
from keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras import backend as K  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from dash import dash_table
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor


# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress INFO/WARN/ERROR from TF C++
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # disable oneDNN optimizations

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)  # suppress absl warnings


def portfolio_dash_range_selector(default_style):

    return html.Div(
        id="portfolio-simulator-range-selector-container",
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "start",
                    "flexWrap": "wrap",
                    "gap": "10px",
                    "paddingTop": "10px",
                    "marginBottom": "10px",
                },
                children=[
                    html.Button(
                        "1M",
                        id="portfolio-range-1M",
                        n_clicks=0,
                        style=default_style,
                        className="simple",
                    ),
                    html.Button(
                        "3M",
                        id="portfolio-range-3M",
                        n_clicks=0,
                        style=default_style,
                        className="simple",
                    ),
                    html.Button(
                        "6M",
                        id="portfolio-range-6M",
                        n_clicks=0,
                        style=default_style,
                        className="simple",
                    ),
                    html.Button(
                        "1Y",
                        id="portfolio-range-1Y",
                        n_clicks=0,
                        style=default_style,
                        className="simple",
                    ),
                    html.Button(
                        "2Y",
                        id="portfolio-range-2Y",
                        n_clicks=0,
                        style=default_style,
                        className="simple",
                    ),
                ],
            )
        ],
    )


def parse_ts_map(
    selected_tickers, portfolio_weights, portfolio_store, budget, threshold=1e-6
):
    ts_map = {}
    selected_values = {ticker for ticker in selected_tickers}
    total_value = None

    for entry, weight in zip(portfolio_store, portfolio_weights):
        if weight <= threshold:
            continue
        if entry["ticker"] not in selected_values:
            continue

        df = pd.read_json(StringIO(entry["historical_json"]), orient="records")
        df["date"] = pd.to_datetime(
            df["date"],
            unit="ms" if isinstance(df["date"].iloc[0], (int, float)) else None,
        )
        df.set_index("date", inplace=True)

        price_series = df["close"].sort_index()

        # Use budget to get actual dollar allocation and number of shares
        latest_price = price_series.iloc[-1]
        dollar_alloc = weight * budget
        shares = dollar_alloc / latest_price if latest_price > 0 else 0

        # Dollar value of the ticker over time
        dollar_ts = price_series * shares

        ts_map[entry["ticker"]] = {
            "weight": weight,
            "shares": shares,
            "price": price_series,
            "value": dollar_ts,
        }

        total_value = (
            dollar_ts
            if total_value is None
            else total_value.add(dollar_ts, fill_value=0)
        )

    return ts_map, total_value


def grid_search_arima_model(
    y,
    criterion="aic",
    p_range=range(0, 4),
    d_range=range(0, 3),
    q_range=range(0, 4),
):
    criterion = criterion.lower()
    if criterion not in ["aic", "bic", "loglikelihood"]:
        raise ValueError("criterion must be one of: 'aic', 'bic', or 'loglikelihood'")

    best_score = -np.inf if criterion == "loglikelihood" else np.inf
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
                        "aic": result.aic,
                        "bic": result.bic,
                        "loglikelihood": result.llf,
                    }[criterion]

                    is_better = (
                        score > best_score
                        if criterion == "loglikelihood"
                        else score < best_score
                    )

                    if is_better:
                        best_score = score
                        best_model = result
                        best_order = (p, d, q)

                except Exception:
                    continue

    return {"score": best_score, "result": best_model, "order": best_order}


def simulate_arima_paths(
    model_result, last_date, forecast_until, num_ensembles, inferred_freq="B"
):

    forecast_until = pd.to_datetime(forecast_until)
    forecast_index = pd.date_range(
        start=last_date + pd.Timedelta(1, unit="D"),
        end=forecast_until,
        freq=inferred_freq,
    )
    n_steps = len(forecast_index)

    simulations = np.zeros((n_steps, num_ensembles))
    for i in range(num_ensembles):
        simulations[:, i] = model_result.simulate(nsimulations=n_steps)

    return (
        simulations,
        forecast_index,
        np.mean(simulations, axis=1),
        np.std(simulations, axis=1),
    )


def grid_search_garch_models(
    returns,
    criterion="aic",
    p_range=range(1, 4),
    q_range=range(1, 4),
    model_types=("GARCH", "EGARCH", "GJR-GARCH"),
    distribution_types=("gaussian", "studentst", "skewstudent", "ged"),
):

    best_model = None
    best_order = None
    best_model_type = None
    best_distribution = None

    best_score = -np.inf if criterion == "loglikelihood" else np.inf

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
                                dist=dist,
                            )
                            result = model.fit(disp="off")

                            score = {
                                "aic": result.aic,
                                "bic": result.bic,
                                "loglikelihood": result.loglikelihood,
                            }[criterion]

                            is_better = (
                                score > best_score
                                if criterion == "loglikelihood"
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
        "model": best_model,
        "model_type": best_model_type,
        "order": best_order,
        "distribution": best_distribution,
        "score": best_score,
    }


def simulate_garch_paths(
    model_result, last_date, forecast_until, num_ensembles, inferred_freq="B"
):

    forecast_until = pd.to_datetime(forecast_until)
    forecast_index = pd.date_range(
        start=last_date + pd.Timedelta(1, unit="D"),
        end=forecast_until,
        freq=inferred_freq,
    )
    n_steps = len(forecast_index)

    simulations = np.zeros((n_steps, num_ensembles))

    for i in range(num_ensembles):
        sim_data = model_result.model.simulate(
            params=model_result.params,
            nobs=n_steps,
            x=None,  # required for ConstantMean
            initial_value=None,
            burn=250,  # optional
        )
        simulations[:, i] = sim_data["data"]

    return (
        simulations,
        forecast_index,
        np.mean(simulations, axis=1),
        np.std(simulations, axis=1),
    )


def train_lstm_model(
    log_returns,
    lookback=20,
    epochs=50,
    batch_size=32,
    lstm_units=50,
    dropout_rate=0.2,
    learning_rate=0.001,
):

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_returns = scaler.fit_transform(log_returns.values.reshape(-1, 1))

    # Prepare sequences
    X, y = [], []
    for i in range(len(scaled_returns) - lookback):
        X.append(scaled_returns[i : i + lookback])
        y.append(scaled_returns[i + lookback])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Clear any previous TF graphs
    K.clear_session()

    # Build model (CPU context)
    with tf.device("/CPU:0"):
        model = Sequential(
            [
                LSTM(lstm_units, input_shape=(lookback, 1)),
                Dropout(dropout_rate),
                Dense(1),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        final_loss = float(history.history["loss"][-1])

    return {
        "model": model,
        "scaler": scaler,
        "lookback": lookback,
        "units": lstm_units,
        "epochs": epochs,
        "loss": final_loss,
    }


def simulate_lstm_paths(
    model_result,
    last_date,
    forecast_until,
    num_ensembles,
    historical_returns,
    inferred_freq="B",
):

    forecast_until = pd.to_datetime(forecast_until)
    forecast_index = pd.date_range(
        start=last_date + pd.Timedelta(1, unit="D"),
        end=forecast_until,
        freq=inferred_freq,
    )
    n_steps = len(forecast_index)

    model = model_result["model"]
    scaler = model_result["scaler"]
    lookback = model_result["lookback"]

    # Prepare initial lookback window from historical data
    scaled_returns = scaler.transform(historical_returns.values.reshape(-1, 1))
    init_window = scaled_returns[-lookback:].reshape(1, lookback, 1)

    simulations = np.zeros((n_steps, num_ensembles))

    # Estimate residual std from training predictions
    X_train, y_train = [], []
    for i in range(len(scaled_returns) - lookback):
        X_train.append(scaled_returns[i : i + lookback])
        y_train.append(scaled_returns[i + lookback])
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
        sim_returns = scaler.inverse_transform(
            np.array(sim_path).reshape(-1, 1)
        ).flatten()
        simulations[:, i] = sim_returns

    return (
        simulations,
        forecast_index,
        np.mean(simulations, axis=1),
        np.std(simulations, axis=1),
    )


def _build_features_from_series(series, lookback):
    """
    Build feature matrix for each time t using previous `lookback` raw lags,
    rolling mean, rolling std, and a normalized time index.
    Returns X (n_samples x n_features), y_delta (n_samples, ).
    """
    if isinstance(series, pd.Series):
        arr = series.values.astype(float)
    else:
        arr = np.asarray(series, dtype=float)

    n = len(arr)
    if n <= lookback + 1:
        raise ValueError("Not enough data for the chosen lookback.")

    X = []
    y = []
    for i in range(lookback, n - 1):  # predict delta at i+1 using history ending at i
        window = arr[
            i - lookback + 1 : i + 1
        ]  # last `lookback` values including arr[i]
        # features: flattened lags, rolling mean, rolling std
        mean_w = window.mean()
        std_w = window.std(ddof=0)
        # time index normalized (helps capture slow nonstationarity)
        time_idx = np.array([(i - lookback) / (n - lookback)], dtype=float)
        feat = np.hstack([window, mean_w, std_w, time_idx])
        X.append(feat)
        # target: delta next step (X_{i+1} - X_i)
        y.append(arr[i + 1] - arr[i])
    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y.squeeze()


def train_gbm_sde_model(
    log_returns,
    lookback=10,
    mu_params=None,
    var_params=None,
    test_size=0.1,
    random_state=42,
):
    """
    Train two gradient-boosting regressors to estimate drift (mu) and log-variance.
    Returns a dictionary with models and metadata.
    """
    # defaults for the GB models (fast, CPU-friendly)
    if mu_params is None:
        mu_params = dict(
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=random_state
        )
    if var_params is None:
        var_params = dict(
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=random_state
        )

    X, y_delta = _build_features_from_series(log_returns, lookback=lookback)

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_delta, test_size=test_size, random_state=random_state, shuffle=False
    )

    # 1) fit mu model (predict ΔX)
    mu_model = HistGradientBoostingRegressor(**mu_params)
    mu_model.fit(X_train, y_train)
    # training predictions
    mu_pred_train = mu_model.predict(X_train)
    mu_pred_val = mu_model.predict(X_val)

    # residuals on full training portion (use X_train to compute residuals)
    residuals = y_train - mu_pred_train
    # small epsilon to avoid log(0)
    eps = 1e-8
    log_var_target = np.log(residuals**2 + eps)

    # 2) fit log-variance model on same features
    log_var_model = HistGradientBoostingRegressor(**var_params)
    log_var_model.fit(X_train, log_var_target)

    # Compute simple diagnostics
    val_resid = y_val - mu_pred_val
    val_log_var_pred = log_var_model.predict(X_val)
    # Estimated sigma on validation
    val_sigma_est = np.exp(0.5 * val_log_var_pred)
    mu_rmse_val = np.sqrt(np.mean((y_val - mu_pred_val) ** 2))
    mean_sigma_val = val_sigma_est.mean()

    return {
        "mu_model": mu_model,
        "log_var_model": log_var_model,
        "lookback": lookback,
        "mu_params": mu_params,
        "var_params": var_params,
        "mu_rmse_val": float(mu_rmse_val),
        "mean_sigma_val": float(mean_sigma_val),
    }


def simulate_gbm_sde_paths(
    model_result,
    last_value,
    last_date,
    forecast_until,
    num_ensembles,
    historical_returns,
    inferred_freq="B",
    mu_clip=0.05,
    sigma_clip=(1e-6, 0.05),
    seed=None,
):
    """
    Simulate future log-return paths using Euler-Maruyama with GBM-fitted mu and log-variance.
    - model_result: dict output from train_gbm_sde_model
    - last_value: last observed log-return value (scalar)
    - last_date: pd.Timestamp (last date in history)
    - forecast_until: end date (str or Timestamp)
    - historical_returns: the series/array used for lookback (must contain at least lookback values)
    """

    np.random.seed(seed)

    mu_model = model_result["mu_model"]
    log_var_model = model_result["log_var_model"]
    lookback = model_result["lookback"]

    if isinstance(historical_returns, pd.Series):
        hist_arr = historical_returns.values.astype(float)
    else:
        hist_arr = np.asarray(historical_returns, dtype=float)

    if len(hist_arr) < lookback:
        raise ValueError("historical_returns too short for lookback")

    forecast_until = pd.to_datetime(forecast_until)
    forecast_index = pd.date_range(
        start=pd.to_datetime(last_date) + pd.Timedelta(days=1),
        end=forecast_until,
        freq=inferred_freq,
    )
    n_steps = len(forecast_index)

    simulations = np.zeros((n_steps, num_ensembles), dtype=np.float32)

    # Pre-allocate feature array for speed reuse
    for j in range(num_ensembles):
        # start with last lookback window from historical data
        hist = hist_arr[-lookback:].copy()
        current_value = float(last_value)

        for t in range(n_steps):
            # build features for this step (same as training)
            mean_w = hist.mean()
            std_w = hist.std(ddof=0)
            time_idx = (len(hist_arr) - lookback + t) / max(1, n_steps)
            feat = (
                np.hstack([hist, mean_w, std_w, time_idx])
                .astype(np.float32)
                .reshape(1, -1)
            )

            # predict mu and log-var
            mu_pred = mu_model.predict(feat)[0]  # ΔX prediction
            log_var_pred = log_var_model.predict(feat)[0]  # log(residual^2)

            # convert to sigma
            sigma_pred = np.exp(0.5 * log_var_pred)

            # Clip for stability (user-provided sensible caps)
            mu = float(np.clip(mu_pred, -mu_clip, mu_clip))
            sigma = float(np.clip(sigma_pred, sigma_clip[0], sigma_clip[1]))

            # Euler-Maruyama (Δt = 1)
            eps = np.random.normal()
            delta_x = mu + sigma * eps
            current_value = current_value + delta_x
            simulations[t, j] = current_value

            # update rolling history with the new simulated return (so path evolves)
            hist = np.roll(hist, -1)
            hist[-1] = current_value

    mean_forecast = simulations.mean(axis=1)
    std_forecast = simulations.std(axis=1)

    return simulations, forecast_index, mean_forecast, std_forecast


def simulation_plot(
    model_used,
    risk_return,
    dates,
    daily_prices,
    log_returns,
    simulations,
    forecast_index,
    mean_ensembles,
    std_ensembles,
    COLORS,
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
        rows=2,
        cols=1,
        subplot_titles=[
            f"Simulated Prices ({model_used})",
            f"Simulated Log Returns ({model_used})",
        ],
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=daily_prices,
            mode="lines",
            name="Historical Price",
            line=dict(color=COLORS["primary"]),
        ),
        row=1,
        col=1,
    )

    for i in range(num_ensembles):
        fig.add_trace(
            go.Scatter(
                x=[dates[-1]] + list(forecast_index),
                y=[daily_prices[-1]] + list(price_paths[:, i]),
                mode="lines",
                line=dict(width=1, color="rgba(255,255,255,0.1)"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=[dates[-1]] + list(forecast_index),
            y=[daily_prices[-1]] + list(mean_price_path),
            mode="lines",
            name="Mean Simulated Price",
            line=dict(color="#ED3500", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[dates[-1]]
            + list(forecast_index)
            + list(forecast_index[::-1])
            + [dates[-1]],
            y=[daily_prices[-1]]
            + list(upper_price_bound)
            + list(lower_price_bound[::-1])
            + [daily_prices[-1]],
            fill="toself",
            fillcolor="rgba(255, 87, 34, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=log_returns,
            mode="lines",
            name="Log Returns",
            line=dict(color=COLORS["primary"]),
        ),
        row=2,
        col=1,
    )

    for i in range(num_ensembles):
        fig.add_trace(
            go.Scatter(
                x=[log_returns.index[-1]] + list(forecast_index),
                y=[log_returns.iloc[-1]] + list(simulations[:, i]),
                mode="lines",
                line=dict(width=1, color="rgba(255,255,255,0.1)"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    lower_return_bound = mean_ensembles - 1.96 * std_ensembles
    upper_return_bound = mean_ensembles + 1.96 * std_ensembles

    fig.add_trace(
        go.Scatter(
            x=[log_returns.index[-1]] + list(forecast_index),
            y=[log_returns.iloc[-1]] + list(mean_ensembles),
            mode="lines",
            name="Mean Simulated Return",
            line=dict(color="#ED3500", width=2),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[log_returns.index[-1]]
            + list(forecast_index)
            + list(forecast_index[::-1])
            + [log_returns.index[-1]],
            y=[log_returns.iloc[-1]]
            + list(upper_return_bound)
            + list(lower_return_bound[::-1])
            + [log_returns.iloc[-1]],
            fill="toself",
            fillcolor="rgba(255, 87, 34, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        title=f"{model_used} forecasts for Portfolio (Risk: {round(risk_return['risk'], 4)*100:.2f}%, Return: {round(risk_return['return'], 4)*100:.2f}%)",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        title_font=dict(color=COLORS["primary"]),
        showlegend=False,
        xaxis2=dict(
            rangeslider=dict(visible=True),
            type="date",
            range=[dates[-1] - timedelta(days=30), forecast_index[-1]],
        ),
    )

    fig.update_yaxes(tickformat=".2%", row=2, col=1)

    return (
        dcc.Graph(
            id=f"{model_used}-simulation-plot",
            figure=fig,
            config={"responsive": True},
            style={"height": "100%", "width": "100%"},
        ),
        {
            "mean_price": mean_price_path[-1],
            "lower_price": lower_price_bound[-1],
            "upper_price": upper_price_bound[-1],
            "mean_return": mean_ensembles[-1],
            "lower_return": lower_return_bound[-1],
            "upper_return": upper_return_bound[-1],
        },
    )


def prediction_summary_table(
    ts_map,
    budget,
    portfolio_value_ts,
    simulation_data,
    chosen_tab,
    COLORS,
):
    tickers = list(ts_map.keys())
    weights = np.array([ts_map[t]["weight"] for t in tickers], dtype=float).reshape(-1)

    def format_delta(value, base, as_pct=False):
        if as_pct:
            # Convert to percentage values
            value_pct = value * 100
            change_abs = (value - base) * 100
            arrow = "↑" if change_abs > 0 else "↓" if change_abs < 0 else "→"
            return f"{value_pct:.2f}% ({change_abs:+.2f}%) {arrow}"
        else:
            # Prices (absolute values)
            change_abs = value - base
            arrow = "↑" if change_abs > 0 else "↓" if change_abs < 0 else "→"
            return f"{value:.2f} ({change_abs:+.2f}) {arrow}"

    rows = []
    last_port_val = portfolio_value_ts.iloc[-1]

    # --- Per ticker rows ---
    for i, ticker in enumerate(tickers):
        alloc = weights[i] * budget
        last_price = ts_map[ticker]["price"].iloc[-1]

        if chosen_tab == "price-summary-simulation":
            # Scale portfolio returns by weight
            mu_i = weights[i] * simulation_data["mean_return"]
            lower_i = weights[i] * simulation_data["lower_return"]
            upper_i = weights[i] * simulation_data["upper_return"]

            mean_price = last_price * np.exp(mu_i)
            lower_price = last_price * np.exp(lower_i)
            upper_price = last_price * np.exp(upper_i)

            row = {
                "Ticker": ticker,
                "Amount Invested ($)": round(alloc, 2),
                "Weight (%)": round(weights[i] * 100, 2),
                "Last Price": round(last_price, 2),
                "Forecast (Mean)": format_delta(mean_price, last_price),
                "95% Lower Bound": format_delta(lower_price, last_price),
                "95% Upper Bound": format_delta(upper_price, last_price),
            }

        else:  # returns-summary-simulation
            mu_i = weights[i] * simulation_data["mean_return"]
            lower_i = weights[i] * simulation_data["lower_return"]
            upper_i = weights[i] * simulation_data["upper_return"]

            row = {
                "Ticker": ticker,
                "Amount Invested ($)": round(alloc, 2),
                "Weight (%)": round(weights[i] * 100, 2),
                "Last Price": round(last_price, 2),
                "Forecast (Mean)": format_delta(mu_i, 0.0, as_pct=True),
                "95% Lower Bound": format_delta(lower_i, 0.0, as_pct=True),
                "95% Upper Bound": format_delta(upper_i, 0.0, as_pct=True),
            }

        rows.append(row)

    # --- Portfolio row ---
    if chosen_tab == "price-summary-simulation":
        mean_text = format_delta(simulation_data["mean_price"], last_port_val)
        lower_text = format_delta(simulation_data["lower_price"], last_port_val)
        upper_text = format_delta(simulation_data["upper_price"], last_port_val)
    else:
        mean_text = format_delta(simulation_data["mean_return"], 0.0, as_pct=True)
        lower_text = format_delta(simulation_data["lower_return"], 0.0, as_pct=True)
        upper_text = format_delta(simulation_data["upper_return"], 0.0, as_pct=True)

    rows.append(
        {
            "Ticker": "PORTFOLIO",
            "Amount Invested ($)": round(budget, 2),
            "Weight (%)": 100.0,
            "Last Price": round(last_port_val, 2),
            "Forecast (Mean)": mean_text,
            "95% Lower Bound": lower_text,
            "95% Upper Bound": upper_text,
        }
    )

    # --- DataTable ---
    return dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in rows[0].keys()],
        data=rows,
        style_table={"maxHeight": "500px", "overflowY": "auto", "overflowX": "auto"},
        style_cell={
            "padding": "6px 10px",
            "backgroundColor": COLORS["background"],
            "color": COLORS["text"],
            "border": "1px solid #444",
            "textAlign": "center",
            "fontSize": "13px",
            "whiteSpace": "normal",
            "height": "auto",
            "minWidth": "90px",
            "maxWidth": "160px",
        },
        style_header={
            "backgroundColor": COLORS["card"],
            "color": COLORS["primary"],
            "fontWeight": "bold",
            "fontSize": "13px",
            "textAlign": "center",
        },
        style_data_conditional=[
            *[
                {
                    "if": {"column_id": col, "filter_query": f"{{{col}}} contains '↑'"},
                    "color": "limegreen",
                    "fontWeight": "bold",
                }
                for col in ["Forecast (Mean)", "95% Lower Bound", "95% Upper Bound"]
            ],
            *[
                {
                    "if": {"column_id": col, "filter_query": f"{{{col}}} contains '↓'"},
                    "color": "red",
                    "fontWeight": "bold",
                }
                for col in ["Forecast (Mean)", "95% Lower Bound", "95% Upper Bound"]
            ],
            *[
                {
                    "if": {"column_id": col, "filter_query": f"{{{col}}} contains '→'"},
                    "color": "gray",
                    "fontWeight": "bold",
                }
                for col in ["Forecast (Mean)", "95% Lower Bound", "95% Upper Bound"]
            ],
            {
                "if": {"filter_query": "{Ticker} = 'PORTFOLIO'"},
                "backgroundColor": "yellow",
                "color": "black",
                "fontWeight": "bold",
            },
        ],
        fixed_rows={"headers": True},
        sort_action="native",
    )
