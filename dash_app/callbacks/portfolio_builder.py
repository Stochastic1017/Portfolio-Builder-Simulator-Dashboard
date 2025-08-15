import os
import re
import sys
import json
import dash
import uuid
import numpy as np
import pandas as pd

from io import StringIO
from datetime import datetime
from datetime import timedelta
from dash import html, Input, Output, State, callback, ctx, no_update


# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from helpers.portfolio_builder import (
    efficient_frontier_dash_range_selector,
    portfolio_optimization,
    plot_efficient_frontier,
    plot_single_ticker,
    summary_table,
)

from helpers.button_styles import (
    COLORS,
    verified_button_portfolio,
    unverified_button_portfolio,
    verified_toggle_button,
    unverified_toggle_button,
    default_style_time_range,
    active_style_time_range,
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


# Populate dropdown with tickers
@callback(
    [
        Output("dropdown-ticker-selection", "options"),
        Output("dropdown-ticker-selection", "value"),
    ],
    [Input("portfolio-store", "data")],
    prevent_initial_call=True,
)
def populate_dropdown_options(data):

    options = [{"label": entry["ticker"], "value": entry["ticker"]} for entry in data]
    default_selected = [entry["ticker"] for entry in data]

    return options, default_selected


@callback(
    [
        Output("portfolio-builder-main-content", "children"),
        Output("max-sharpe-button", "disabled"),
        Output("toggle-max-sharpe", "style"),
        Output("max-diversification-button", "disabled"),
        Output("toggle-max-diversification", "style"),
        Output("min-variance-button", "disabled"),
        Output("toggle-min-variance", "style"),
        Output("equal-weights-button", "disabled"),
        Output("toggle-equal-weights", "style"),
    ],
    [
        Input("btn-verify-budget", "n_clicks"),
        Input("dropdown-ticker-selection", "value"),
    ],
    [State("verify-budget", "data")],
    prevent_initial_call=True,
)
def update_portfolio_builder_main_output(_, selected_tickers, verify_budget):

    if not _:
        raise dash.exceptions.PreventUpdate

    if (
        (not ctx.triggered_id == "btn-verify-budget")
        or (not selected_tickers)
        or (not verify_budget)
    ):
        return (
            no_update,
            True,
            unverified_toggle_button,
            True,
            unverified_toggle_button,
            True,
            unverified_toggle_button,
            True,
            unverified_toggle_button,
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
                efficient_frontier_dash_range_selector(
                    default_style=default_style_time_range
                ),
                html.Div(
                    id="efficient-frontier-plot-container",
                    style={"flex": "1", "overflow": "hidden"},
                ),
            ],
        ),
        False,
        verified_toggle_button,
        False,
        verified_toggle_button,
        False,
        verified_toggle_button,
        False,
        verified_toggle_button,
    )


# Highlight buttons based on time-range selected
@callback(
    [
        Output("efficient-frontier-range-1M", "style"),
        Output("efficient-frontier-range-3M", "style"),
        Output("efficient-frontier-range-6M", "style"),
        Output("efficient-frontier-range-1Y", "style"),
        Output("efficient-frontier-range-2Y", "style"),
        Output("efficient-frontier-selected-range", "data"),
    ],
    [
        Input("efficient-frontier-range-1M", "n_clicks"),
        Input("efficient-frontier-range-3M", "n_clicks"),
        Input("efficient-frontier-range-6M", "n_clicks"),
        Input("efficient-frontier-range-1Y", "n_clicks"),
        Input("efficient-frontier-range-2Y", "n_clicks"),
    ],
)
def update_range_styles(*btn_clicks):
    button_ids = [
        "efficient-frontier-range-1M",
        "efficient-frontier-range-3M",
        "efficient-frontier-range-6M",
        "efficient-frontier-range-1Y",
        "efficient-frontier-range-2Y",
    ]

    # If no button has been clicked yet, fall back to "2Y"
    if not any(click and click > 0 for click in btn_clicks):
        selected = "efficient-frontier-range-2Y"
    else:
        selected = ctx.triggered_id or "efficient-frontier-range-2Y"

    style_map = {btn_id: default_style_time_range for btn_id in button_ids}
    style_map[selected] = active_style_time_range

    # return the last token (e.g. '1M', '3M', '2Y')
    selected_range_value = selected.rsplit("-", 1)[-1]

    return (
        style_map["efficient-frontier-range-1M"],
        style_map["efficient-frontier-range-3M"],
        style_map["efficient-frontier-range-6M"],
        style_map["efficient-frontier-range-1Y"],
        style_map["efficient-frontier-range-2Y"],
        selected_range_value,
    )


@callback(
    [
        Output("efficient-frontier-plot-container", "children"),
    ],
    [
        Input("max-sharpe-button", "value"),
        Input("max-diversification-button", "value"),
        Input("min-variance-button", "value"),
        Input("equal-weights-button", "value"),
        Input("efficient-frontier-selected-range", "data"),
    ],
    [
        State("portfolio-store", "data"),
        State("dropdown-ticker-selection", "value"),
    ],
    prevent_initial_call=True,
)
def update_efficient_frontier_on_range_change(
    max_sharpe_on,
    max_diversification_on,
    min_variance_on,
    equal_weights_on,
    selected_range,
    cache_data,
    selected_tickers,
):

    if len(selected_tickers) > 1:

        filtered_data = [
            entry
            for entry in cache_data
            if entry["ticker"] in [ticker for ticker in selected_tickers]
        ]

        optimization_dict = portfolio_optimization(filtered_data, selected_range)

        return (
            plot_efficient_frontier(
                max_sharpe_on,
                min_variance_on,
                max_diversification_on,
                equal_weights_on,
                optimization_dict,
                COLORS,
            ),
        )

    else:

        historical_df = pd.read_json(
            StringIO(cache_data[0]["historical_json"]), orient="records"
        )
        historical_df["date"] = pd.to_datetime(historical_df["date"])

        today = pd.Timestamp.today()
        time_deltas = {
            "1M": timedelta(days=30),
            "3M": timedelta(days=90),
            "6M": timedelta(days=180),
            "1Y": timedelta(days=365),
            "2Y": None,
        }

        cutoff = time_deltas.get(selected_range.upper(), None)

        if cutoff:
            start_date = today - cutoff
            filtered_df = historical_df[historical_df["date"] >= start_date]

        else:
            filtered_df = historical_df

        # Extract relevant metrics (filtered)
        daily_log_returns = np.asarray(
            np.log(1 + filtered_df["close"].pct_change().dropna())
        )

        return (
            plot_single_ticker(
                max_sharpe_on,
                min_variance_on,
                max_diversification_on,
                equal_weights_on,
                daily_log_returns,
                COLORS,
            ),
        )


# Summary table for chosen portfolio
@callback(
    [
        Output("summary-table-container", "children"),
        Output("btn-confirm-portfolio", "disabled"),
        Output("btn-confirm-portfolio", "style"),
        Output("btn-confirm-portfolio", "className"),
        Output("confirmed-weights-store", "data"),
        Output("portfolio-clicked-risk-return", "data"),
    ],
    [Input("efficient-frontier-graph", "clickData")],
    [
        State("portfolio-store", "data"),
        State("dropdown-ticker-selection", "value"),
        State("budget-value", "data"),
    ],
    prevent_initial_call=True,
)
def display_summary_table(clickData, cache_data, selected_tickers, budget):

    if not clickData:
        return no_update

    point = clickData["points"][0]
    weights = point["customdata"]
    risk = point["x"]
    ret = point["y"]

    # Format risk/return info
    header_text = f"Portfolio Details for Risk: {risk:.2%} | Return: {ret:.2%}"

    # Filter portfolio data using selected tickers
    filtered_data = [
        entry
        for entry in cache_data
        if entry["ticker"] in [ticker for ticker in selected_tickers]
    ]

    return (
        [
            html.H4(
                header_text,
                style={
                    "color": COLORS["primary"],
                    "textAlign": "center",
                    "marginBottom": "20px",
                },
            ),
            summary_table(filtered_data, COLORS, weights=weights, budget=budget),
        ],
        False,
        verified_button_portfolio,
        "special",
        weights,
        {"risk": risk, "return": ret},
    )


# Confirm portfolio and display appropriate message
@callback(
    [
        Output("portfolio-builder-toast", "is_open"),
        Output("portfolio-builder-toast", "children"),
        Output("portfolio-builder-toast", "header"),
        Output("portfolio-builder-toast", "icon"),
        Output("portfolio-builder-toast", "style"),
        Output("portfolio-risk-return", "data"),
        Output("latest-date", "data"),
    ],
    [Input("btn-confirm-portfolio", "n_clicks")],
    [
        State("portfolio-clicked-risk-return", "data"),
        State("portfolio-store", "data"),
    ],
    prevent_initial_call=True,
)
def confirm_portfolio(_, risk_return, cache_data):

    latest_date_str = None

    for asset in cache_data:
        try:
            history = json.loads(asset["historical_json"])
            latest_ts = max(entry["date"] for entry in history if "date" in entry)
            latest_date_str = datetime.fromtimestamp(latest_ts / 1000).strftime(
                "%Y-%m-%d"
            )
            break  # stop after first successful parse
        except Exception:
            continue

    if not risk_return or "risk" not in risk_return or "return" not in risk_return:
        raise dash.exceptions.PreventUpdate

    return (
        True,
        "Portfolio confirmed! You're ready to simulate.",
        "Success",
        "success",  # bootstrap icons style
        {
            "position": "fixed",
            "top": "70px",
            "right": "30px",
            "zIndex": 9999,
            "backgroundColor": COLORS["card"],
            "color": COLORS["text"],
            "borderLeft": f"6px solid green",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
            "padding": "12px 16px",
            "borderRadius": "6px",
            "width": "350px",
        },
        {
            "risk": risk_return["risk"],
            "return": risk_return["return"],
        },
        latest_date_str,
    )


# Allow users to navigate to portfolio simulator page
@callback(
    [
        Output("btn-portfolio-simulator", "disabled"),
        Output("btn-portfolio-simulator", "style"),
        Output("btn-portfolio-simulator", "className"),
    ],
    [Input("btn-confirm-portfolio", "n_clicks")],
    prevent_initial_call=True,
)
def update_portfolio_analytics_button(n_clicks):

    if n_clicks:
        return False, verified_button_portfolio, "special"

    return True, unverified_button_portfolio, ""
