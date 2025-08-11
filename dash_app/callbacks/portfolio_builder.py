import os
import sys
import json
import dash
import dash_daq as daq
import dash_bootstrap_components as dbc

from datetime import datetime
from scipy.special import comb
from dash import html, Input, Output, State, ALL, MATCH, callback, ctx, dcc, no_update


# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from helpers.portfolio_builder import (
    efficient_frontier_dash_range_selector,
    portfolio_optimization,
    plot_efficient_frontier,
    summary_table,
)

from helpers.button_styles import (
    COLORS,
    verified_button_portfolio,
    unverified_button_portfolio,
    verified_button_style,
    unverified_button_style,
    verified_toggle_button,
    unverified_toggle_button,
    default_style_time_range,
    active_style_time_range,
)


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


# Track selected tickers
@callback(
    [
        Output("dropdown-ticker-selection", "options", allow_duplicate=True),
        Output("dropdown-ticker-selection", "value", allow_duplicate=True),
        Output("selected-ticker-card", "children", allow_duplicate=True),
        Output("selected-tickers-store", "data", allow_duplicate=True),
        Output("portfolio-builder-main-content", "children", allow_duplicate=True),
        Output("max-sharpe-button", "disabled", allow_duplicate=True),
        Output("toggle-max-sharpe", "style", allow_duplicate=True),
        Output("max-diversification-button", "disabled", allow_duplicate=True),
        Output("toggle-max-diversification", "style", allow_duplicate=True),
        Output("min-variance-button", "disabled", allow_duplicate=True),
        Output("toggle-min-variance", "style", allow_duplicate=True),
        Output("equal-weights-button", "disabled", allow_duplicate=True),
        Output("toggle-equal-weights", "style", allow_duplicate=True),
        Output("summary-table-container", "children", allow_duplicate=True),
    ],
    [
        Input({"type": "remove-ticker", "index": ALL}, "n_clicks"),
        Input("dropdown-ticker-selection", "value"),
    ],
    [
        State("portfolio-store", "data"),
        State("dropdown-ticker-selection", "options"),
        State("selected-ticker-card", "children"),
    ],
    prevent_initial_call=True,
)
def update_ticker_selection(
    _, dropdown_value, portfolio_data, current_options, current_children
):

    trigger = ctx.triggered_id

    full_tickers = [entry["ticker"] for entry in portfolio_data]

    # Reconstruct current selected list from tag buttons
    current_selected = (
        [child["props"]["id"]["index"] for child in current_children]
        if current_children
        else full_tickers
    )

    # 1. Remove ticker if "x" clicked
    if isinstance(trigger, dict) and trigger.get("type") == "remove-ticker":
        ticker_to_remove = trigger["index"]

        # Prevent removal if it would leave fewer than 2
        if len(current_selected) <= 2:
            return no_update, no_update, no_update

        current_selected = [t for t in current_selected if t != ticker_to_remove]
        dropdown_value = None

    # 2. Add from dropdown
    elif isinstance(trigger, str) and dropdown_value:
        if dropdown_value not in current_selected:
            current_selected.append(dropdown_value)
        dropdown_value = None  # clear dropdown after adding

    # Filter dropdown options
    new_options = [
        {"label": t, "value": t} for t in full_tickers if t not in current_selected
    ]

    # Build card tags
    selected_tags = [
        html.Div(
            [
                html.Span(
                    ticker,
                    style={
                        "marginRight": "6px",
                        "fontSize": "0.8em",
                        "fontWeight": "500",
                        "color": COLORS["background"],
                    },
                ),
                html.Button(
                    "×",
                    id={"type": "remove-ticker", "index": ticker},
                    n_clicks=0,
                    style={
                        "border": "none",
                        "background": "transparent",
                        "color": COLORS["background"],
                        "fontSize": "0.8em",
                        "cursor": "pointer",
                        "padding": "0",
                        "lineHeight": "1",
                    },
                ),
            ],
            style={
                "padding": "4px 10px",
                "backgroundColor": COLORS["primary"],
                "borderRadius": "14px",
                "display": "flex",
                "alignItems": "center",
                "gap": "4px",
                "height": "28px",
                "overflow": "hidden",
            },
            id={"type": "ticker-tag", "index": ticker},
        )
        for ticker in current_selected
    ]

    return (
        new_options,
        dropdown_value,
        selected_tags,
        [{"value": t, "label": t} for t in current_selected],
        no_update,
        True,
        unverified_toggle_button,
        True,
        unverified_toggle_button,
        True,
        unverified_toggle_button,
        True,
        unverified_toggle_button,
        no_update,
    )


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
        Output("efficient-frontier-clicked", "data"),
    ],
    Input("btn-efficient-frontier", "n_clicks"),
    prevent_initial_call=True,
)
def update_portfolio_builder_main_output(
    _,
):

    if not _:
        raise dash.exceptions.PreventUpdate

    if not ctx.triggered_id == "btn-efficient-frontier":
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
            False,
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
        True,
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
        State("selected-tickers-store", "data"),
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

    filtered_data = [
        entry
        for entry in cache_data
        if entry["ticker"] in [item["value"] for item in selected_tickers]
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
    [State("portfolio-store", "data"), State("selected-tickers-store", "data")],
    prevent_initial_call=True,
)
def display_summary_table(clickData, cache_data, selected_tickers):

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
        if entry["ticker"] in [item["value"] for item in selected_tickers]
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
            summary_table(filtered_data, COLORS, weights=weights),
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
    Input("btn-confirm-portfolio", "n_clicks"),
    prevent_initial_call=True,
)
def update_portfolio_analytics_button(n_clicks):

    if n_clicks:
        return False, verified_button_portfolio, "special"

    return True, unverified_button_portfolio, ""
