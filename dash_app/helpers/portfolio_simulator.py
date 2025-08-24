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
from plotly.subplots import make_subplots
from keras.optimizers import Adam  # type: ignore
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential  # type: ignore
from keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras import backend as K  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
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


def summarize_prices_from_cumlog(cum, last_price):

    gross = np.exp(cum)
    median_price = last_price * np.median(gross, axis=1)
    lower_price = last_price * np.quantile(gross, 0.025, axis=1)
    upper_price = last_price * np.quantile(gross, 0.975, axis=1)

    return {
        "median": median_price.tolist(),
        "lower": lower_price.tolist(),
        "upper": upper_price.tolist(),
    }


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
    model_result,
    last_price,
    last_date,
    forecast_until,
    num_ensembles,
    inferred_freq="B",
    progress_cb=None,
):

    forecast_until = pd.to_datetime(forecast_until)
    forecast_index = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        end=forecast_until,
        freq=inferred_freq,
    )
    H = len(forecast_index)
    E = int(num_ensembles)

    price_paths = np.zeros((E, H), dtype=float)

    for i in range(E):
        # simulate log-returns, then cum-sum → gross → price path
        sim_ret = model_result.simulate(nsimulations=H)  # (H,)
        cum = np.cumsum(sim_ret)  # cum log-returns
        price_paths[i, :] = float(last_price) * np.exp(cum)  # level prices
        if progress_cb:
            progress_cb()  # tick once per ensemble

    return forecast_index, price_paths


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
    model_result,
    last_price,
    last_date,
    forecast_until,
    num_ensembles,
    inferred_freq="B",
    burn=250,
    progress_cb=None,
):

    forecast_until = pd.to_datetime(forecast_until)
    forecast_index = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        end=forecast_until,
        freq=inferred_freq,
    )
    H = len(forecast_index)
    E = int(num_ensembles)

    price_paths = np.zeros((E, H), dtype=float)

    for i in range(E):
        sim_data = model_result.model.simulate(
            params=model_result.params,
            nobs=H,
            x=None,
            initial_value=None,
            burn=burn,
        )
        sim_ret = np.asarray(sim_data["data"], dtype=float)  # per-step log-returns
        cum = np.cumsum(sim_ret)  # cumulative log-returns
        price_paths[i, :] = float(last_price) * np.exp(cum)  # level prices

        if progress_cb:
            progress_cb()  # +1 per ensemble

    return forecast_index, price_paths


def simulation_plot(
    dates,
    daily_prices,
    returns_summary,
    cum_returns_summary,
    forecast_index,
    COLORS,
):
    last_price = float(daily_prices[-1])

    # Convert cumulative log-returns summaries → price summaries
    cum_median = np.array(cum_returns_summary["median"])
    cum_low = np.array(cum_returns_summary["lower"])
    cum_high = np.array(cum_returns_summary["upper"])

    median_price = last_price * np.exp(cum_median)
    lower_price = last_price * np.exp(cum_low)
    upper_price = last_price * np.exp(cum_high)

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            f"Forecasted Prices",
            f"Forecasted Log Returns",
        ],
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    # 1) Price panel
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

    # median line
    fig.add_trace(
        go.Scatter(
            x=[dates[-1]] + list(forecast_index),
            y=[last_price] + list(median_price),
            mode="lines",
            name="Median Forecast Price",
            line=dict(color="#ED3500", width=2),
        ),
        row=1,
        col=1,
    )

    # CI band
    fig.add_trace(
        go.Scatter(
            x=[dates[-1]]
            + list(forecast_index)
            + list(forecast_index[::-1])
            + [dates[-1]],
            y=[last_price] + list(upper_price) + list(lower_price[::-1]) + [last_price],
            fill="toself",
            fillcolor="rgba(255, 87, 34, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # -------- Returns panel (no anchoring at dates[-1]) --------
    prices_series = pd.Series(daily_prices, index=dates)
    hist_rets = np.log(prices_series / prices_series.shift(1)).dropna()

    # 1) Historical returns
    fig.add_trace(
        go.Scatter(
            x=hist_rets.index,
            y=hist_rets.values,
            mode="lines",
            name="Historical Log Returns",
            line=dict(color=COLORS["primary"]),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Prepare forecast arrays
    ret_med = np.asarray(returns_summary["median"], dtype=float)
    ret_low = np.asarray(returns_summary["lower"], dtype=float)
    ret_up = np.asarray(returns_summary["upper"], dtype=float)

    # 2) Forecast median returns (START at forecast_index[0])
    fig.add_trace(
        go.Scatter(
            x=list(forecast_index),
            y=ret_med.tolist(),
            mode="lines",
            name="Median Forecast Return",
            line=dict(color="#ED3500", width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 3) 95% band ONLY over forecast_index (no dates[-1] in polygon)
    fig.add_trace(
        go.Scatter(
            x=list(forecast_index) + list(forecast_index[::-1]),
            y=list(ret_up) + list(ret_low[::-1]),
            fill="toself",
            fillcolor="rgba(255, 87, 34, 0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 4) Single red bridge from last historical point to first forecast point
    last_x = hist_rets.index[-1]
    last_y = float(hist_rets.values[-1])
    first_x = forecast_index[0]
    first_y = float(ret_med[0])

    fig.add_trace(
        go.Scatter(
            x=[last_x, first_x],
            y=[last_y, first_y],
            mode="lines",
            line=dict(color="#ED3500", width=2),  # solid red
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        title_font=dict(color=COLORS["primary"]),
        showlegend=False,
        xaxis2=dict(
            rangeslider=dict(visible=True),
            type="date",
            range=[dates[0], forecast_index[-1]],
        ),
    )
    fig.update_yaxes(tickformat=".2%", row=2, col=1)

    # Return a compact payload with end-point summaries for tables
    end_idx = -1
    return (
        dcc.Graph(
            id=f"prediction-simulation-plot",
            figure=fig,
            config={"responsive": True},
            style={"height": "100%", "width": "100%"},
        ),
        {
            "median_price": float(median_price[end_idx]),
            "lower_price": float(lower_price[end_idx]),
            "upper_price": float(upper_price[end_idx]),
            "median_return": float(ret_med[end_idx]),
            "lower_return": float(ret_low[end_idx]),
            "upper_return": float(ret_up[end_idx]),
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
            value_pct = value * 100.0
            change_abs = (value - base) * 100.0
            arrow = "↑" if change_abs > 0 else "↓" if change_abs < 0 else "→"
            return f"{value_pct:.2f}% ({change_abs:+.2f}%) {arrow}"
        else:
            change_abs = value - base
            arrow = "↑" if change_abs > 0 else "↓" if change_abs < 0 else "→"
            return f"{value:.2f} ({change_abs:+.2f}) {arrow}"

    rows = []
    last_port_val = float(portfolio_value_ts.iloc[-1])

    assets_payload = (simulation_data or {}).get("assets", {})
    portfolio_payload = (simulation_data or {}).get("portfolio", {})

    # --- Per‑ticker rows (use per‑asset summaries, not portfolio-scaled) ---
    for i, ticker in enumerate(tickers):
        alloc = float(weights[i] * budget)
        last_price = float(ts_map[ticker]["price"].iloc[-1])

        asset_data = assets_payload.get(ticker)
        if not asset_data:
            # If this asset wasn't simulated (shouldn't happen), skip gracefully
            continue

        if chosen_tab == "price-summary-simulation":
            # Use final horizon price summaries for this ticker
            prices = asset_data["prices"]  # dict with "median"/"lower"/"upper" (lists)
            median_price = float(prices["median"][-1])
            lower_price = float(prices["lower"][-1])
            upper_price = float(prices["upper"][-1])

            row = {
                "Ticker": ticker,
                "Amount Invested ($)": round(alloc, 2),
                "Weight (%)": round(weights[i] * 100.0, 2),
                "Last Price": round(last_price, 2),
                "Forecast (Median)": format_delta(median_price, last_price),
                "Lower Bound": format_delta(lower_price, last_price),
                "Upper Bound": format_delta(upper_price, last_price),
            }

        else:  # "returns-summary-simulation"
            # Use final horizon per‑step return summaries for this ticker
            rets = asset_data["returns"]  # dict with "median"/"lower"/"upper" (lists)
            median_ret = float(rets["median"][-1])
            lower_ret = float(rets["lower"][-1])
            upper_ret = float(rets["upper"][-1])

            row = {
                "Ticker": ticker,
                "Amount Invested ($)": round(alloc, 2),
                "Weight (%)": round(weights[i] * 100.0, 2),
                "Last Price": round(last_price, 2),
                "Forecast (Median)": format_delta(median_ret, 0.0, as_pct=True),
                "Lower Bound": format_delta(lower_ret, 0.0, as_pct=True),
                "Upper Bound": format_delta(upper_ret, 0.0, as_pct=True),
            }

        rows.append(row)

    # If for some reason we have no rows, bail cleanly
    if not rows:
        return dash_table.DataTable(
            columns=[{"name": "Message", "id": "Message"}],
            data=[{"Message": "No simulation data available."}],
            style_table={
                "maxHeight": "500px",
                "overflowY": "auto",
                "overflowX": "auto",
            },
        )

    # --- Portfolio row (use portfolio summaries from payload) ---
    if chosen_tab == "price-summary-simulation":
        median_text = format_delta(
            float(portfolio_payload["median_price"]), last_port_val
        )
        lower_text = format_delta(
            float(portfolio_payload["lower_price"]), last_port_val
        )
        upper_text = format_delta(
            float(portfolio_payload["upper_price"]), last_port_val
        )
    else:
        median_text = format_delta(
            float(portfolio_payload["median_return"]), 0.0, as_pct=True
        )
        lower_text = format_delta(
            float(portfolio_payload["lower_return"]), 0.0, as_pct=True
        )
        upper_text = format_delta(
            float(portfolio_payload["upper_return"]), 0.0, as_pct=True
        )

    rows.append(
        {
            "Ticker": "PORTFOLIO",
            "Amount Invested ($)": round(float(budget), 2),
            "Weight (%)": 100.0,
            "Last Price": round(last_port_val, 2),
            "Forecast (Median)": median_text,
            "Lower Bound": lower_text,
            "Upper Bound": upper_text,
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
                for col in ["Forecast (Median)", "95% Lower Bound", "95% Upper Bound"]
            ],
            *[
                {
                    "if": {"column_id": col, "filter_query": f"{{{col}}} contains '↓'"},
                    "color": "red",
                    "fontWeight": "bold",
                }
                for col in ["Forecast (Median)", "95% Lower Bound", "95% Upper Bound"]
            ],
            *[
                {
                    "if": {"column_id": col, "filter_query": f"{{{col}}} contains '→'"},
                    "color": "gray",
                    "fontWeight": "bold",
                }
                for col in ["Forecast (Median)", "95% Lower Bound", "95% Upper Bound"]
            ],
            {
                "if": {"filter_query": "{Ticker} = 'PORTFOLIO'"},
                "backgroundColor": COLORS["secondary"],
            },
            *[
                {
                    "if": {"filter_query": "{Ticker} = 'PORTFOLIO'", "column_id": col},
                    "color": "black",
                    "fontWeight": "bold",
                }
                for col in ["Ticker", "Amount Invested ($)", "Weight (%)", "Last Price"]
            ],
        ],
        fixed_rows={"headers": True},
        sort_action="native",
    )
