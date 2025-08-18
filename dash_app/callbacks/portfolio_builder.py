import os
import re
import sys
import json
import dash
import uuid
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from io import StringIO
from datetime import datetime
from datetime import timedelta
from dash import html, Input, Output, State, callback, ctx, dcc, no_update


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
    verified_button_style,
    unverified_button_style,
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


@callback(
    [
        Output("verify-budget", "data"),
        Output("budget-value", "data"),
    ],
    Input("btn-verify-budget", "n_clicks"),
    State("inp-budget", "value"),
    prevent_initial_call=True,
)
def handle_verify_budget_store(n_clicks, budget_input):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    result = validate_budget(budget_input)
    if result["valid"]:
        return (
            {"verified": True},
            result["value"],
        )
    else:
        return (
            {"verified": False, "error": result["error"]},
            None,
        )


# Verify budget and reset status if budget changes
@callback(
    [
        Output("portfolio-builder-toast", "is_open", allow_duplicate=True),
        Output("portfolio-builder-toast", "children", allow_duplicate=True),
        Output("portfolio-builder-toast", "header", allow_duplicate=True),
        Output("portfolio-builder-toast", "icon", allow_duplicate=True),
        Output("portfolio-builder-toast", "style", allow_duplicate=True),
    ],
    Input("verify-budget", "data"),
    prevent_initial_call=True,
)
def show_budget_toast(verify_budget):
    if verify_budget.get("verified"):
        return (
            True,
            "Budget verified successfully!",
            "Success",
            "success",
            {
                "position": "fixed",
                "top": "70px",
                "right": "30px",
                "zIndex": 9999,
                "backgroundColor": COLORS["card"],
                "color": COLORS["text"],
                "borderLeft": "6px solid green",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
                "padding": "12px 16px",
                "borderRadius": "6px",
                "width": "350px",
            },
        )
    else:
        return (
            True,
            verify_budget.get("error", "Unknown error"),
            "Error",
            "danger",
            {
                "position": "fixed",
                "top": "70px",
                "right": "30px",
                "zIndex": 9999,
                "backgroundColor": COLORS["card"],
                "color": COLORS["text"],
                "borderLeft": "6px solid red",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
                "padding": "12px 16px",
                "borderRadius": "6px",
                "width": "350px",
            },
        )


# Toggle styles between enabled/disabled status
@callback(
    [
        Output("btn-efficient-frontier", "disabled"),
        Output("btn-efficient-frontier", "style"),
        Output("btn-efficient-frontier", "className"),
    ],
    Input("verify-budget", "data"),
    prevent_initial_call=True,
)
def toggle_button_states(verify_budget):

    if verify_budget.get("verified"):

        return (
            False,
            verified_button_style,
            "simple",
        )

    else:
        return (
            True,
            unverified_button_style,
            "",
        )


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
        Input("btn-efficient-frontier", "n_clicks"),
    ],
    State("budget-value", "data"),
    prevent_initial_call=True,
)
def update_portfolio_builder_main_output(_, budget_value):

    if budget_value is None:
        raise dash.exceptions.PreventUpdate

    if ctx.triggered_id == "dropdown-ticker-selection":
        return (
            None,
            True,
            unverified_toggle_button,
            True,
            unverified_toggle_button,
            True,
            unverified_toggle_button,
            True,
            unverified_toggle_button,
        )

    if ctx.triggered_id == "btn-efficient-frontier":

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


@callback(
    [
        Output("btn-efficient-frontier", "disabled", allow_duplicate=True),
        Output("btn-efficient-frontier", "style", allow_duplicate=True),
        Output("btn-efficient-frontier", "className", allow_duplicate=True),
    ],
    Input("inp-budget", "value"),
    prevent_initial_call=True,
)
def reset_buttons_on_budget_change(_):
    # Any change means budget must be re-verified
    return True, unverified_button_style, ""


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


# Update efficient frontier plot upon range change
@callback(
    [
        Output("efficient-frontier-plot-container", "children"),
    ],
    [
        Input("btn-efficient-frontier", "n_clicks"),
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
    _,
    max_sharpe_on,
    max_diversification_on,
    min_variance_on,
    equal_weights_on,
    selected_range,
    cache_data,
    selected_tickers,
):

    if (not cache_data) or (not selected_tickers):
        default_plot = go.Figure()
        default_plot.update_layout(
            template="plotly_dark",
            title="Efficient Frontier (Risk vs Return MPT Framework)",
            title_font=dict(color=COLORS["primary"]),
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font=dict(color=COLORS["text"]),
            showlegend=False,
            autosize=True,
            height=None,
        )
        return (
            dcc.Graph(
                id="efficient-frontier-graph",
                figure=default_plot,
                config={"responsive": True},
                style={"height": "100%", "width": "100%"},
            ),
        )

    if len(selected_tickers) == 1:
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

        mean_log_returns = daily_log_returns.mean()
        std_log_returns = daily_log_returns.std()

        return (
            plot_single_ticker(
                mean_log_returns,
                std_log_returns,
                max_sharpe_on,
                min_variance_on,
                max_diversification_on,
                equal_weights_on,
                COLORS,
            ),
        )

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


# Reset efficient frontier plot upon changing dropdown ticker selection or budget
@callback(
    [
        Output("portfolio-builder-main-content", "children", allow_duplicate=True),
        Output("max-sharpe-button", "disabled", allow_duplicate=True),
        Output("toggle-max-sharpe", "style", allow_duplicate=True),
        Output("max-diversification-button", "disabled", allow_duplicate=True),
        Output("toggle-max-diversification", "style", allow_duplicate=True),
        Output("min-variance-button", "disabled", allow_duplicate=True),
        Output("toggle-min-variance", "style", allow_duplicate=True),
        Output("equal-weights-button", "disabled", allow_duplicate=True),
        Output("toggle-equal-weights", "style", allow_duplicate=True),
    ],
    Input("dropdown-ticker-selection", "value"),
    prevent_initial_call=True,
)
def reset_on_ticker_change(_):
    return (
        None,
        True,
        unverified_toggle_button,
        True,
        unverified_toggle_button,
        True,
        unverified_toggle_button,
        True,
        unverified_toggle_button,
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
    Input("efficient-frontier-graph", "clickData"),
    [
        State("efficient-frontier-selected-range", "data"),
        State("portfolio-store", "data"),
        State("dropdown-ticker-selection", "value"),
        State("budget-value", "data"),
    ],
    prevent_initial_call=True,
)
def display_summary_table(
    clickData, selected_range, cache_data, selected_tickers, budget
):
    if not clickData:
        raise dash.exceptions.PreventUpdate
    
    # Map selected_range codes to readable text
    range_labels = {
        "1M": "past 1 month",
        "3M": "past 3 months",
        "6M": "past 6 months",
        "1Y": "past 1 year",
        "2Y": "past 2 years",
    }
    period_text = range_labels.get(selected_range, f"past {selected_range}")

    point = clickData["points"][0]
    weights = point["customdata"]
    risk = point["x"]
    ret = point["y"]

    header_text = (
        f"Portfolio Details for Risk: {risk:.2%} | Return: {ret:.2%} | {period_text}"
    )

    filtered_data = [
        entry for entry in cache_data if entry["ticker"] in selected_tickers
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


# Reset summary table upon changing dropdown ticker selection or budget
@callback(
    [
        Output("summary-table-container", "children", allow_duplicate=True),
        Output("btn-confirm-portfolio", "disabled", allow_duplicate=True),
        Output("btn-confirm-portfolio", "style", allow_duplicate=True),
        Output("btn-confirm-portfolio", "className", allow_duplicate=True),
        Output("confirmed-weights-store", "data", allow_duplicate=True),
        Output("portfolio-clicked-risk-return", "data", allow_duplicate=True),
    ],
    [
        Input("dropdown-ticker-selection", "value"),
        Input("inp-budget", "value"),
    ],
    prevent_initial_call=True,
)
def reset_summary_on_changes(_, __):
    return (
        None,
        True,
        unverified_button_portfolio,
        "",
        None,
        None,
    )


# Reset summary table on range
@callback(
    [
        Output("summary-table-container", "children", allow_duplicate=True),
        Output("btn-confirm-portfolio", "disabled", allow_duplicate=True),
        Output("btn-confirm-portfolio", "style", allow_duplicate=True),
        Output("btn-confirm-portfolio", "className", allow_duplicate=True),
        Output("confirmed-weights-store", "data", allow_duplicate=True),
        Output("portfolio-clicked-risk-return", "data", allow_duplicate=True),
    ],
    Input("efficient-frontier-selected-range", "data"),
    prevent_initial_call=True,
)
def reset_summary_on_range_change(_):
    return (
        None,
        True,
        unverified_button_portfolio,
        "",
        None,
        None,
    )


# Confirm portfolio and display appropriate message
@callback(
    [
        Output("portfolio-builder-toast", "is_open", allow_duplicate=True),
        Output("portfolio-builder-toast", "children", allow_duplicate=True),
        Output("portfolio-builder-toast", "header", allow_duplicate=True),
        Output("portfolio-builder-toast", "icon", allow_duplicate=True),
        Output("portfolio-builder-toast", "style", allow_duplicate=True),
        Output("confirmed-portfolio-details", "data"),
        Output("latest-date", "data"),
        Output("portfolio-confirmed", "data")
    ],
    [Input("btn-confirm-portfolio", "n_clicks")],
    [
        State("portfolio-clicked-risk-return", "data"),
        State("portfolio-store", "data"),
        State("efficient-frontier-selected-range", "data"),
    ],
    prevent_initial_call=True,
)
def confirm_portfolio(_, risk_return, cache_data, selected_range):

    latest_date_str = None

    for asset in cache_data:
        try:
            history = json.loads(asset["historical_json"])
            latest_ts = max(entry["date"] for entry in history if "date" in entry)
            latest_date_str = datetime.fromtimestamp(latest_ts / 1000).strftime(
                "%Y-%m-%d"
            )
            break
        except Exception:
            continue

    if (not risk_return) or ("risk" not in risk_return) or ("return" not in risk_return):
        raise dash.exceptions.PreventUpdate

    return (
        True,
        "Portfolio confirmed! You're ready to simulate.",
        "Success",
        "success",
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
            "selected_range": selected_range
        },
        latest_date_str,
        True
    )
