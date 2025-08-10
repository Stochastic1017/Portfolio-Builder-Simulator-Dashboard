import os
import re
import sys
import uuid
import dash
import numpy as np
import dash_bootstrap_components as dbc

from datetime import datetime, timedelta
from dash import html, Input, Output, State, ALL, MATCH, callback, ctx, dcc, no_update

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.portfolio_simulator import (
    parse_ts_map,
    portfolio_dash_range_selector,
    grid_search_arima_model,
    simulate_arima_paths,
    grid_search_garch_models,
    simulate_garch_paths,
    train_lstm_model,
    simulate_lstm_paths,
    train_gbm_sde_model,
    simulate_gbm_sde_paths,
    simulation_plot,
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
    active_labelStyle_radioitems,
    active_inputStyle_radioitems,
    active_style_radioitems,
)


# Stock ticker validation procedure
def validate_budget(budget):

    if budget is None or budget == "":
        return {"valid": False, "value": None, "error": "Budget cannot be empty."}

    # Remove leading/trailing whitespace
    budget = budget.strip()

    # Basic sanity check for allowed characters
    if not re.fullmatch(r"[0-9,]*\.?[0-9]*", budget):
        return {
            "valid": False,
            "value": None,
            "error": "Budget contains invalid characters.",
        }

    # Check multiple decimals
    if budget.count(".") > 1:
        return {
            "valid": False,
            "value": None,
            "error": "Budget has multiple decimal points.",
        }

    # Validate comma placement using regex
    if "," in budget:
        # Regex to match correct comma placement: 1,000 or 12,345.67
        if not re.fullmatch(r"(?:\d{1,3})(?:,\d{3})*(?:\.\d{1,2})?", budget):
            return {
                "valid": False,
                "value": None,
                "error": "Commas are placed incorrectly.",
            }

    # Remove commas for numeric conversion
    numeric_str = budget.replace(",", "")

    try:

        value = float(numeric_str)
        if value < 0:
            return {
                "valid": False,
                "value": None,
                "error": "Budget cannot be negative.",
            }

        return {"valid": True, "value": value, "error": None}

    except ValueError:
        return {"valid": False, "value": None, "error": "Could not parse budget value."}


# Get next business day
def next_business_day(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    next_day = date_obj + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_day += timedelta(days=1)
    return next_day.strftime("%Y-%m-%d")  # Return as string if needed


# Get exact one year business day
def max_forecast_date(latest_date_str):
    date_obj = datetime.strptime(latest_date_str, "%Y-%m-%d")
    one_year_later = date_obj + timedelta(days=365)
    return one_year_later.strftime("%Y-%m-%d")


# Ensure budget formatting is consistent for monetary inputs
@callback(
    Output("inp-budget", "value"),
    Input("inp-budget", "value"),
    prevent_initial_call=True,
)
def format_budget_input(value):
    if not value:
        return ""

    # Remove everything except digits and decimal point
    raw_value = re.sub(r"[^\d.]", "", value)

    try:
        # Format number with commas, preserve decimal if present
        if "." in raw_value:
            number = float(raw_value)
            formatted = f"{number:,.2f}"
        else:
            number = int(raw_value)
            formatted = f"{number:,}"
        return formatted
    except:
        return value


# Verify budget and reset status if budget changes
@callback(
    [
        Output("verify-budget", "data"),
        Output("budget-value", "data"),
    ],
    [Input("btn-verify-budget", "n_clicks")],
    [State("inp-budget", "value")],
    prevent_initial_call=True,
)
def handle_verify_budget(_, budget_input):
    trigger_id = ctx.triggered_id

    if trigger_id == "btn-verify-budget":
        result = validate_budget(budget_input)

        if result["valid"]:
            return {"verified": True}, result["value"]
        else:
            return {"verified": False, "error": result["error"]}, None

    elif trigger_id == "inp-budget":
        return {"verified": False}, None

    return no_update


# Show validation symbol upon successful verification
@callback(
    [
        Output("inp-budget", "valid"),
        Output("inp-budget", "invalid"),
        Output("inp-budget", "key"),
    ],
    Input("verify-budget", "data"),
    prevent_initial_call=True,
)
def set_budget_validation(verify_budget):

    # Defensive fallback key
    dynamic_key = f"key-{uuid.uuid4()}"

    is_verified = verify_budget.get("verified", False)

    if not is_verified:
        return (False, True, dynamic_key)

    return (True, False, dynamic_key)


# Toggle portfolio performance and date chooser after verification of budget
@callback(
    [
        Output("btn-portfolio-performance", "disabled"),
        Output("btn-portfolio-performance", "style"),
        Output("btn-portfolio-performance", "className"),
        Output("date-chooser-simulation", "disabled"),
        Output("date-chooser-simulation", "min_date_allowed"),
        Output("date-chooser-simulation", "max_date_allowed"),
        Output("num-ensemble-slider", "disabled"),
        Output("model-selection-criterion", "options"),
        Output("model-selection-criterion", "labelStyle"),
        Output("model-selection-criterion", "inputStyle"),
        Output("model-selection-criterion", "style"),
    ],
    Input("verify-budget", "data"),
    State("latest-date", "data"),
)
def enable_initial_controls(verify_budget, latest_date):
    is_verified = verify_budget.get("verified", False)

    if is_verified:
        return (
            False,
            verified_button_style,
            "simple",
            False,
            next_business_day(latest_date),
            max_forecast_date(latest_date),
            False,
            [
                {"label": "AIC (Akaike)", "value": "aic", "disabled": False},
                {"label": "BIC (Bayesian)", "value": "bic", "disabled": False},
                {"label": "LogLikelihood", "value": "loglikelihood", "disabled": False},
            ],
            active_labelStyle_radioitems,
            active_inputStyle_radioitems,
            active_style_radioitems,
        )

    else:
        return (
            True,
            unverified_button_style,
            "",
            True,
            next_business_day(latest_date),
            max_forecast_date(latest_date),
            True,
            [
                {"label": "AIC (Akaike)", "value": "aic", "disabled": True},
                {"label": "BIC (Bayesian)", "value": "bic", "disabled": True},
                {"label": "LogLikelihood", "value": "loglikelihood", "disabled": True},
            ],
            active_labelStyle_radioitems,
            active_inputStyle_radioitems,
            active_style_radioitems,
        )


# Activate LSTM button after user inputs date and a valid budget
@callback(
    [
        Output("btn-lstm-performance", "disabled"),
        Output("btn-lstm-performance", "style"),
        Output("btn-lstm-performance", "className"),
        Output("btn-gbm-performance", "disabled"),
        Output("btn-gbm-performance", "style"),
        Output("btn-gbm-performance", "className"),
    ],
    Input("date-chooser-simulation", "date"),
)
def activate_lstm_nnsde_button(selected_date):

    if selected_date:
        return (
            False,
            verified_button_style,
            "simple",
            False,
            verified_button_style,
            "simple",
        )

    else:
        return (True, unverified_button_style, "", True, unverified_button_style, "")


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


@callback(
    [
        Output("portfolio-simulator-main-content", "children"),
    ],
    [
        Input("btn-portfolio-performance", "n_clicks"),
        Input("btn-arima-performance", "n_clicks"),
        Input("btn-garch-performance", "n_clicks"),
        Input("btn-lstm-performance", "n_clicks"),
        Input("btn-gbm-performance", "n_clicks"),
        Input("model-selection-criterion", "value"),
    ],
    [
        State("verify-budget", "data"),
        State("portfolio-store", "data"),
        State("selected-tickers-store", "data"),
        State("confirmed-weights-store", "data"),
        State("budget-value", "data"),
        State("portfolio-risk-return", "data"),
        State("date-chooser-simulation", "date"),
        State("num-ensemble-slider", "value"),
    ],
    prevent_initial_call=True,
)
def update_portfolio_simulator_main_plot(
    _,
    __,
    ___,
    ____,
    _____,
    criterion_selector,
    verify_budget,
    portfolio_store,
    selected_tickers,
    weights,
    budget,
    risk_return,
    forecast_until,
    num_ensembles,
):
    button_id = ctx.triggered_id

    if not verify_budget.get("verified", False):
        raise dash.exceptions.PreventUpdate

    _, portfolio_value_ts = parse_ts_map(
        selected_tickers=selected_tickers,
        portfolio_weights=weights,
        portfolio_store=portfolio_store,
        budget=budget,
    )

    log_returns = (
        np.log(portfolio_value_ts / portfolio_value_ts.shift(1)).shift(-1).dropna()
    )

    if button_id == "btn-portfolio-performance":
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

    elif button_id == "btn-arima-performance":
        arima_model = grid_search_arima_model(log_returns, criterion=criterion_selector)

        simulations, forecast_index, mean_ensembles, std_ensembles = (
            simulate_arima_paths(
                model_result=arima_model["result"],
                last_date=portfolio_value_ts.index[-1],
                forecast_until=forecast_until,
                num_ensembles=num_ensembles,
                inferred_freq="B",
            )
        )

        return (
            (
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
                        simulation_plot(
                            model_used="ARIMA",
                            risk_return=risk_return,
                            dates=portfolio_value_ts.index,
                            daily_prices=portfolio_value_ts.values,
                            log_returns=log_returns,
                            simulations=simulations,
                            forecast_index=forecast_index,
                            mean_ensembles=mean_ensembles,
                            std_ensembles=std_ensembles,
                            COLORS=COLORS,
                        )
                    ],
                )
            ),
        )

    elif button_id == "btn-garch-performance":

        garch_model = grid_search_garch_models(
            log_returns, criterion=criterion_selector
        )

        simulations, forecast_index, mean_returns, std_returns = simulate_garch_paths(
            model_result=garch_model["model"],
            last_date=portfolio_value_ts.index[-1],
            forecast_until=forecast_until,
            num_ensembles=num_ensembles,
            inferred_freq="B",
        )

        return (
            (
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
                        simulation_plot(
                            model_used="GARCH",
                            risk_return=risk_return,
                            dates=portfolio_value_ts.index,
                            daily_prices=portfolio_value_ts.values,
                            log_returns=log_returns,
                            simulations=simulations,
                            forecast_index=forecast_index,
                            mean_ensembles=mean_returns,
                            std_ensembles=std_returns,
                            COLORS=COLORS,
                        )
                    ],
                )
            ),
        )

    elif button_id == "btn-gbm-performance":

        gbm_model = train_gbm_sde_model(log_returns, lookback=10)

        simulations, forecast_index, mean_returns, std_returns = simulate_gbm_sde_paths(
            model_result=gbm_model,
            last_value=log_returns.iloc[-1],
            last_date=portfolio_value_ts.index[-1],
            forecast_until=forecast_until,
            num_ensembles=num_ensembles,
            historical_returns=log_returns,
            inferred_freq="B",
        )

        return (
            (
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
                        simulation_plot(
                            model_used="GBM",
                            risk_return=risk_return,
                            dates=portfolio_value_ts.index,
                            daily_prices=portfolio_value_ts.values,
                            log_returns=log_returns,
                            simulations=simulations,
                            forecast_index=forecast_index,
                            mean_ensembles=mean_returns,
                            std_ensembles=std_returns,
                            COLORS=COLORS,
                        )
                    ],
                )
            ),
        )

    elif button_id == "btn-lstm-performance":

        lstm_model = train_lstm_model(log_returns)

        simulations, forecast_index, mean_returns, std_returns = simulate_lstm_paths(
            model_result=lstm_model,
            last_date=portfolio_value_ts.index[-1],
            forecast_until=forecast_until,
            num_ensembles=num_ensembles,
            historical_returns=log_returns,
            inferred_freq="B",
        )

        return (
            (
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
                        simulation_plot(
                            model_used="LSTM",
                            risk_return=risk_return,
                            dates=portfolio_value_ts.index,
                            daily_prices=portfolio_value_ts.values,
                            log_returns=log_returns,
                            simulations=simulations,
                            forecast_index=forecast_index,
                            mean_ensembles=mean_returns,
                            std_ensembles=std_returns,
                            COLORS=COLORS,
                        )
                    ],
                )
            ),
        )

    return no_update


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
        State("selected-tickers-store", "data"),
        State("confirmed-weights-store", "data"),
        State("portfolio-risk-return", "data"),
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
                full_name=f"Portfolio (Risk: {round(risk_return['risk'], 4)*100:.2f}%, Return: {round(risk_return['return'], 4)*100:.2f}%)",
                dates=portfolio_value_ts.index,
                daily_prices=portfolio_value_ts.values,
                daily_log_returns=portfolio_log_returns,
                COLORS=COLORS,
            ),
        )

    elif active_tab == "portfolio-tab-stats":
        return (
            create_statistics_table(
                dates=portfolio_value_ts.index,
                daily_prices=portfolio_value_ts.values,
                daily_log_returns=portfolio_log_returns,
                COLORS=COLORS,
            ),
        )
