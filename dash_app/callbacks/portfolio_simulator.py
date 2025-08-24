import os
import sys
import dash
import numpy as np

from datetime import datetime, timedelta
from dash import html, Input, Output, State, callback, ctx, dcc, no_update

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.portfolio_simulator import (
    parse_ts_map,
    portfolio_dash_range_selector,
    grid_search_arima_model,
    simulate_arima_paths,
    grid_search_garch_models,
    simulate_garch_paths,
    simulation_plot,
    prediction_summary_table,
)

from helpers.portfolio_exploration import (
    create_historic_plots,
    create_statistics_table,
)

from helpers.button_styles import (
    COLORS,
    verified_button_style,
    unverified_button_style,
    default_style_time_range,
    active_style_time_range,
)


def forecast_plot(
    button_id,
    portfolio_value_ts,
    ts_map,
    num_ensembles,
    criterion_selector,
    forecast_until,
    chosen_quantile,
):

    # --- per-asset ensembles (price paths) ---
    per_asset_price_paths = {}
    per_asset_summaries = {}
    forecast_index = None
    E = int(num_ensembles)

    for tkr, d in ts_map.items():

        px = d["price"]
        rets = np.log(px / px.shift(1)).dropna()

        if button_id == "btn-arima-performance":
            arima_model = grid_search_arima_model(rets, criterion=criterion_selector)
            header_model_spec = f"Model: ARIMA{arima_model['order']} | Criterion: {criterion_selector} | Ensembles: {num_ensembles} | Forecast Date: {forecast_until} | Confidence: {chosen_quantile}"
            fi, price_paths = simulate_arima_paths(
                model_result=arima_model["result"],
                last_price=float(px.iloc[-1]),
                last_date=px.index[-1],
                forecast_until=forecast_until,
                num_ensembles=E,
                inferred_freq="B",
            )

        if button_id == "btn-garch-performance":
            garch_model = grid_search_garch_models(rets, criterion=criterion_selector)
            header_model_spec = f"{garch_model['model_type']}({garch_model['order'][0]},{garch_model['order'][1]}) • {str(garch_model['distribution']).title()} | Criterion: {criterion_selector} | Ensembles: {num_ensembles} | Forecast Date: {forecast_until} | Confidence: {chosen_quantile}"
            fi, price_paths = simulate_garch_paths(
                model_result=garch_model["model"],
                last_price=float(px.iloc[-1]),
                last_date=px.index[-1],
                forecast_until=forecast_until,
                num_ensembles=E,
                inferred_freq="B",
                burn=250,
            )

        if forecast_index is None:
            forecast_index = fi

        per_asset_price_paths[tkr] = price_paths

        # per-asset summaries for your table (median/CI of price; and returns derived from those prices)
        asset_prices = summarize_price_paths(price_paths)  # dict of lists

        # returns per ensemble for this asset
        prev = float(px.iloc[-1])
        prev_mat = np.full((E, 1), prev)
        asset_ret_paths = np.log(
            price_paths / np.concatenate([prev_mat, price_paths[:, :-1]], axis=1)
        )
        asset_returns = {
            "median": np.median(asset_ret_paths, axis=0).tolist(),
            "lower": np.quantile(asset_ret_paths, 0.025, axis=0).tolist(),
            "upper": np.quantile(asset_ret_paths, 0.975, axis=0).tolist(),
        }
        per_asset_summaries[tkr] = {
            "prices": asset_prices,
            "returns": asset_returns,
            "last_price": prev,
            "shares": float(d["shares"]),
            "weight": float(d["weight"]),
        }

    # --- aggregate to portfolio per ensemble ---
    H = len(forecast_index)
    port_price_paths = np.zeros((E, H), dtype=float)  # (E, H)

    for tkr, data in per_asset_summaries.items():
        sh = data["shares"]
        port_price_paths += sh * per_asset_price_paths[tkr]  # broadcast ok: (E,H)

    last_port_val = float(portfolio_value_ts.iloc[-1])
    prev_port = np.full((E, 1), last_port_val)
    port_ret_paths = np.log(
        port_price_paths / np.concatenate([prev_port, port_price_paths[:, :-1]], axis=1)
    )

    # --- summarize portfolio prices & returns (accurate quantiles from ensembles) ---
    port_prices = {
        "median": np.median(port_price_paths, axis=0).tolist(),
        "lower": np.quantile(
            port_price_paths, (1 - chosen_quantile) / 2, axis=0
        ).tolist(),
        "upper": np.quantile(
            port_price_paths, (1 + chosen_quantile) / 2, axis=0
        ).tolist(),
    }
    port_returns = {
        "median": np.median(port_ret_paths, axis=0).tolist(),
        "lower": np.quantile(
            port_ret_paths, (1 - chosen_quantile) / 2, axis=0
        ).tolist(),
        "upper": np.quantile(
            port_ret_paths, (1 + chosen_quantile) / 2, axis=0
        ).tolist(),
    }

    # (Optional) clear heavy arrays now to free RAM
    per_asset_price_paths = None
    port_price_paths = None
    port_ret_paths = None

    # --- build cum log-returns summaries from portfolio *price* summaries (for the price panel) ---
    port_cum = {
        "median": (np.log(np.array(port_prices["median"]) / last_port_val)).tolist(),
        "lower": (
            np.log(np.maximum(np.array(port_prices["lower"]), 1e-12) / last_port_val)
        ).tolist(),
        "upper": (
            np.log(np.maximum(np.array(port_prices["upper"]), 1e-12) / last_port_val)
        ).tolist(),
    }

    # --- plot: summary-only (CI for both price and returns, now accurate) ---
    forecast_plot, _ = simulation_plot(
        dates=portfolio_value_ts.index,
        daily_prices=portfolio_value_ts.values,
        returns_summary=port_returns,
        cum_returns_summary=port_cum,
        forecast_index=forecast_index,
        COLORS=COLORS,
    )

    # --- payload for tables (portfolio + per-asset) ---
    simulation_payload = {
        "scope": "portfolio_and_assets",
        "forecast_index": [str(d.date()) for d in forecast_index],
        "portfolio": {
            "median_price": float(np.array(port_prices["median"])[-1]),
            "lower_price": float(np.array(port_prices["lower"])[-1]),
            "upper_price": float(np.array(port_prices["upper"])[-1]),
            "median_return": float(np.array(port_returns["median"])[-1]),
            "lower_return": float(np.array(port_returns["lower"])[-1]),
            "upper_return": float(np.array(port_returns["upper"])[-1]),
        },
        "assets": {
            tkr: {
                "last_price": data["last_price"],
                "shares": data["shares"],
                "weight": data["weight"],
                "returns": data["returns"],
                "prices": data["prices"],
            }
            for tkr, data in per_asset_summaries.items()
        },
    }

    return (
        html.Div(
            id="portfolio-plot-container",
            style={
                "display": "flex",
                "flexDirection": "column",
                "flex": "1",
                "overflow": "hidden",
                "height": "100%",
                "width": "100%",
            },
            children=[
                html.Div(
                    header_model_spec,
                    style={
                        "fontSize": "1.05rem",
                        "fontWeight": 700,
                        "color": COLORS["primary"],
                    },
                ),
                forecast_plot,
            ],
        ),
        simulation_payload,
    )


def step_log_returns(prev_level, path):
    prev = np.concatenate([[prev_level], path[:-1]])
    return np.log(np.array(path) / prev)


def summarize_price_paths(price_paths):
    """
    price_paths: (E, H)
    returns dict of arrays length H
    """
    return {
        "median": np.median(price_paths, axis=0).tolist(),
        "lower": np.quantile(price_paths, 0.025, axis=0).tolist(),
        "upper": np.quantile(price_paths, 0.975, axis=0).tolist(),
    }


# Get next business day
def next_business_day(date_str):
    if date_str is None:
        return

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    next_day = date_obj + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_day += timedelta(days=1)
    return next_day.strftime("%Y-%m-%d")  # Return as string if needed


# Get exact one year business day
def max_forecast_date(latest_date_str):
    if latest_date_str is None:
        return

    date_obj = datetime.strptime(latest_date_str, "%Y-%m-%d")
    one_year_later = date_obj + timedelta(days=365)
    return one_year_later.strftime("%Y-%m-%d")


# Switch states after user confirms a portfolio
@callback(
    [
        Output("btn-portfolio-performance", "disabled"),
        Output("date-chooser-simulation", "disabled"),
        Output("date-chooser-simulation", "min_date_allowed"),
        Output("date-chooser-simulation", "max_date_allowed"),
        Output("num-ensemble-slider", "disabled"),
        Output("selected-portfolio-simulation", "children"),
        Output("selected-range-simulation", "children"),
        Output("latest-date-simulation", "children"),
        Output("model-selection-criterion", "options"),
        Output("ci-level-selector", "options"),
    ],
    [Input("latest-date", "data"), Input("confirmed-portfolio-details", "data")],
    State("portfolio-confirmed", "data"),
)
def change_states_upon_portfolio_confirmation(
    latest_date, portfolio_details, confirmed_portfolio
):

    if not confirmed_portfolio:
        raise dash.exceptions.PreventUpdate

    confirmed_risk = portfolio_details["risk"] * 100
    confirmed_return = portfolio_details["return"] * 100

    # Map selected_range codes to readable text
    confirmed_range = portfolio_details["selected_range"]
    range_labels = {
        "1M": "past 1 month",
        "3M": "past 3 months",
        "6M": "past 6 months",
        "1Y": "past 1 year",
        "2Y": "past 2 years",
    }
    period_text = range_labels.get(confirmed_range, f"past {confirmed_range}")

    return (
        False,
        False,
        next_business_day(latest_date),
        max_forecast_date(latest_date),
        False,
        f"Risk : {confirmed_risk:.2f}% | Return : {confirmed_return:.2f}%",
        f"{period_text}",
        f"Last Updated: {latest_date}",
        [
            {
                "label": "Akaike Information Criterion",
                "value": "aic",
                "disabled": False,
            },
            {
                "label": "Bayesian Information Criterion",
                "value": "bic",
                "disabled": False,
            },
            {
                "label": "LogLikelihood",
                "value": "loglikelihood",
                "disabled": False,
            },
        ],
        [
            {"label": "90%", "value": 0.90, "disabled": False},
            {"label": "95%", "value": 0.95, "disabled": False},
            {"label": "99%", "value": 0.99, "disabled": False},
        ],
    )


# Activate ARIMA and GARCH button after user inputs date, valid budget,
# and preferred criterion information.
@callback(
    [
        Output("btn-arima-performance", "disabled"),
        Output("btn-arima-performance", "style"),
        Output("btn-arima-performance", "className"),
        Output("btn-garch-performance", "disabled"),
        Output("btn-garch-performance", "style"),
        Output("btn-garch-performance", "className"),
    ],
    [
        Input("date-chooser-simulation", "date"),
        Input("model-selection-criterion", "value"),
    ],
    prevent_initial_call=True,
)
def activate_arima_garch_buttons(selected_date, selected_criterion_information):

    if selected_date and selected_criterion_information:
        return (
            False,
            verified_button_style,
            "simple",
            False,
            verified_button_style,
            "simple",
        )

    else:
        return (
            True,
            unverified_button_style,
            "",
            True,
            unverified_button_style,
            "",
        )


# Show historical performance of portfolio
@callback(
    [
        Output("portfolio-simulator-main-content", "children"),
    ],
    [
        Input("btn-portfolio-performance", "n_clicks"),
    ],
    [
        State("portfolio-confirmed", "data"),
        State("portfolio-store", "data"),
        State("dropdown-ticker-selection", "value"),
        State("confirmed-weights-store", "data"),
        State("budget-value", "data"),
    ],
    prevent_initial_call=True,
)
def Show_portfolio_performance(
    _,
    confirmed_portfolio,
    portfolio_store,
    selected_tickers,
    weights,
    budget,
):

    if not confirmed_portfolio:
        raise dash.exceptions.PreventUpdate

    _, portfolio_value_ts = parse_ts_map(
        selected_tickers=selected_tickers,
        portfolio_weights=weights,
        portfolio_store=portfolio_store,
        budget=budget,
    )

    if ctx.triggered_id == "btn-portfolio-performance":
        return (
            html.Div(
                id="simulator-main-panel",
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "height": "100%",
                    "width": "100%",
                    "overflow": "hidden",
                },
                children=[
                    portfolio_dash_range_selector(
                        default_style=default_style_time_range
                    ),
                    dcc.Tabs(
                        id="portfolio-performance-tabs",
                        value="portfolio-tab-plot",
                        style={
                            "marginTop": "10px",
                            "backgroundColor": COLORS["card"],
                            "color": COLORS["text"],
                            "height": "42px",
                            "borderRadius": "5px",
                            "overflow": "hidden",
                        },
                        colors={
                            "border": COLORS["background"],
                            "primary": COLORS["primary"],
                            "background": COLORS["card"],
                        },
                        children=[
                            dcc.Tab(
                                label="Price & Returns Plot",
                                value="portfolio-tab-plot",
                                style={
                                    "backgroundColor": COLORS["background"],
                                    "color": COLORS["text"],
                                    "padding": "6px 18px",
                                    "fontSize": "14px",
                                    "fontWeight": "bold",
                                    "border": "none",
                                    "borderBottom": f"2px solid transparent",
                                },
                                selected_style={
                                    "backgroundColor": COLORS["background"],
                                    "color": COLORS["primary"],
                                    "padding": "6px 18px",
                                    "fontSize": "14px",
                                    "fontWeight": "bold",
                                    "border": "none",
                                    "borderBottom": f"2px solid {COLORS['primary']}",
                                },
                            ),
                            dcc.Tab(
                                label="Statistical Summary",
                                value="portfolio-tab-stats",
                                style={
                                    "backgroundColor": COLORS["background"],
                                    "color": COLORS["text"],
                                    "padding": "6px 18px",
                                    "fontWeight": "bold",
                                    "fontSize": "14px",
                                    "border": "none",
                                    "borderBottom": f"2px solid transparent",
                                },
                                selected_style={
                                    "backgroundColor": COLORS["background"],
                                    "color": COLORS["primary"],
                                    "padding": "6px 18px",
                                    "fontWeight": "bold",
                                    "fontSize": "14px",
                                    "border": "none",
                                    "borderBottom": f"2px solid {COLORS['primary']}",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        id="portfolio-plot-container",
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "flex": "1",
                            "overflow": "hidden",
                            "height": "100%",
                            "width": "100%",
                        },
                    ),
                ],
            ),
        )


@callback(
    [
        Output("portfolio-simulator-main-content", "children", allow_duplicate=True),
        Output("simulated-forecasts", "data"),
    ],
    [
        Input("btn-arima-performance", "n_clicks"),
        Input("btn-garch-performance", "n_clicks"),
    ],
    [
        State("model-selection-criterion", "value"),
        State("portfolio-confirmed", "data"),
        State("portfolio-store", "data"),
        State("dropdown-ticker-selection", "value"),
        State("confirmed-weights-store", "data"),
        State("budget-value", "data"),
        State("confirmed-portfolio-details", "data"),
        State("date-chooser-simulation", "date"),
        State("num-ensemble-slider", "value"),
        State("ci-level-selector", "value"),
    ],
    prevent_initial_call=True,
)
def update_portfolio_simulator_main_plot(
    _,
    __,
    criterion_selector,
    confirmed_portfolio,
    portfolio_store,
    selected_tickers,
    weights,
    budget,
    risk_return,
    forecast_until,
    num_ensembles,
    chosen_quantile,
):

    if not confirmed_portfolio:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered_id

    ts_map, portfolio_value_ts = parse_ts_map(
        selected_tickers=selected_tickers,
        portfolio_weights=weights,
        portfolio_store=portfolio_store,
        budget=budget,
    )

    return forecast_plot(
        button_id,
        portfolio_value_ts,
        ts_map,
        num_ensembles,
        criterion_selector,
        forecast_until,
        chosen_quantile,
    )


# Highlight buttons based on time-range selected
@callback(
    [
        Output("portfolio-range-1M", "style"),
        Output("portfolio-range-3M", "style"),
        Output("portfolio-range-6M", "style"),
        Output("portfolio-range-1Y", "style"),
        Output("portfolio-range-2Y", "style"),
        Output("portfolio-selected-range", "data"),
    ],
    [
        Input("portfolio-range-1M", "n_clicks"),
        Input("portfolio-range-3M", "n_clicks"),
        Input("portfolio-range-6M", "n_clicks"),
        Input("portfolio-range-1Y", "n_clicks"),
        Input("portfolio-range-2Y", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def update_portfolio_range_styles(*btn_clicks):
    button_ids = [
        "portfolio-range-1M",
        "portfolio-range-3M",
        "portfolio-range-6M",
        "portfolio-range-1Y",
        "portfolio-range-2Y",
    ]

    # If no button has been clicked yet, fall back to "All"
    if not any(click and click > 0 for click in btn_clicks):
        selected = "portfolio-range-2Y"
    else:
        selected = ctx.triggered_id or "portfolio-range-2Y"

    style_map = {btn_id: default_style_time_range for btn_id in button_ids}
    style_map[selected] = active_style_time_range

    return (
        style_map["portfolio-range-1M"],
        style_map["portfolio-range-3M"],
        style_map["portfolio-range-6M"],
        style_map["portfolio-range-1Y"],
        style_map["portfolio-range-2Y"],
        selected.split("-")[-1],
    )


# Update historic daily plot based on range selected
@callback(
    [
        Output("portfolio-plot-container", "children"),
    ],
    [
        Input("portfolio-performance-tabs", "value"),
        Input("portfolio-selected-range", "data"),
    ],
    [
        State("budget-value", "data"),
        State("portfolio-store", "data"),
        State("dropdown-ticker-selection", "value"),
        State("confirmed-weights-store", "data"),
        State("confirmed-portfolio-details", "data"),
    ],
    prevent_initial_call=True,
)
def update_plot_on_range_change(
    active_tab,
    selected_range,
    budget,
    portfolio_store,
    selected_tickers,
    weights,
    risk_return,
):

    # Compute budget-adjusted time series
    _, portfolio_value_ts = parse_ts_map(
        selected_tickers=selected_tickers,
        portfolio_weights=weights,
        portfolio_store=portfolio_store,
        budget=budget,
    )

    today = portfolio_value_ts.index[-1]
    range_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": None}

    selected_range = (selected_range or "2Y").upper()
    if selected_range not in range_days:
        selected_range = "2Y"

    if range_days[selected_range] is not None:
        cutoff = today - timedelta(days=range_days[selected_range])
        portfolio_value_ts = portfolio_value_ts[portfolio_value_ts.index >= cutoff]

    portfolio_log_returns = np.asarray(
        np.log(1 + portfolio_value_ts.pct_change().dropna())
    )

    if active_tab == "portfolio-tab-plot":
        return (
            create_historic_plots(
                full_name=f"Portfolio (Risk: {risk_return['risk']*100:.2f}% | Return: {risk_return['return']*100:.2f}%)",
                dates=portfolio_value_ts.index,
                daily_prices=portfolio_value_ts.values,
                daily_log_returns=portfolio_log_returns,
                selected_range=selected_range,
                COLORS=COLORS,
            ),
        )

    elif active_tab == "portfolio-tab-stats":
        return (
            create_statistics_table(
                dates=portfolio_value_ts.index,
                daily_prices=portfolio_value_ts.values,
                daily_log_returns=portfolio_log_returns,
                selected_range=selected_range,
                COLORS=COLORS,
            ),
        )


# Show summary table for forecasts
@callback(
    Output("summary-forecast-simulator", "children"),
    Input("simulated-forecasts", "data"),
    [
        State("portfolio-store", "data"),
        State("dropdown-ticker-selection", "value"),
        State("confirmed-weights-store", "data"),
        State("budget-value", "data"),
    ],
    prevent_initial_call=True,
)
def display_forecast_summary_table(
    simulation_data, portfolio_store, selected_tickers, weights, budget
):

    if not simulation_data:
        return no_update

    ts_map, portfolio_value_ts = parse_ts_map(
        selected_tickers=selected_tickers,
        portfolio_weights=weights,
        portfolio_store=portfolio_store,
        budget=budget,
    )

    return (
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "column",
                "minHeight": 0,
                "height": "100%",
                "width": "100%",
                "overflow": "hidden",
            },
            children=[
                dcc.Tabs(
                    id="simulation-summary-tabs",
                    value="price-summary-simulation",
                    style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["text"],
                        "borderRadius": "5px",
                        "overflow": "hidden",
                    },
                    colors={
                        "border": COLORS["background"],
                        "primary": COLORS["primary"],
                        "background": COLORS["card"],
                    },
                    children=[
                        dcc.Tab(
                            label="Price Summary Simulation",
                            value="price-summary-simulation",
                            style={
                                "backgroundColor": COLORS["background"],
                                "color": COLORS["text"],
                                "padding": "6px 18px",
                                "fontSize": "14px",
                                "fontWeight": "bold",
                                "border": "none",
                                "borderBottom": f"2px solid transparent",
                            },
                            selected_style={
                                "backgroundColor": COLORS["background"],
                                "color": COLORS["primary"],
                                "padding": "6px 18px",
                                "fontSize": "14px",
                                "fontWeight": "bold",
                                "border": "none",
                                "borderBottom": f"2px solid {COLORS['primary']}",
                            },
                            children=[
                                prediction_summary_table(
                                    ts_map=ts_map,
                                    budget=budget,
                                    portfolio_value_ts=portfolio_value_ts,
                                    simulation_data=simulation_data,
                                    chosen_tab="price-summary-simulation",
                                    COLORS=COLORS,
                                )
                            ],
                        ),
                        dcc.Tab(
                            label="Returns Summary Simulation",
                            value="returns-summary-simulation",
                            style={
                                "backgroundColor": COLORS["background"],
                                "color": COLORS["text"],
                                "padding": "6px 18px",
                                "fontWeight": "bold",
                                "fontSize": "14px",
                                "border": "none",
                                "borderBottom": f"2px solid transparent",
                            },
                            selected_style={
                                "backgroundColor": COLORS["background"],
                                "color": COLORS["primary"],
                                "padding": "6px 18px",
                                "fontWeight": "bold",
                                "fontSize": "14px",
                                "border": "none",
                                "borderBottom": f"2px solid {COLORS['primary']}",
                            },
                            children=[
                                prediction_summary_table(
                                    ts_map=ts_map,
                                    budget=budget,
                                    portfolio_value_ts=portfolio_value_ts,
                                    simulation_data=simulation_data,
                                    chosen_tab="returns-summary-simulation",
                                    COLORS=COLORS,
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
    )
