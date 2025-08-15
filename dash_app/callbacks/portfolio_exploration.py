import os
import uuid
import sys
import dash
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc

from io import StringIO
from datetime import timedelta
from dotenv import load_dotenv
from dash import html, dcc, Input, Output, State, callback, ctx, no_update

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load .env to fetch api key
load_dotenv()
api_key = os.getenv("POLYGON_API_KEY")

from helpers.portfolio_exploration import (
    StockTickerInformation,
    validate_stock_ticker,
    dash_range_selector,
    create_historic_plots,
    create_statistics_table,
    company_metadata_layout,
    news_article_card_layout,
)

from helpers.button_styles import (
    COLORS,
    verified_button_portfolio,
    unverified_button_portfolio,
    verified_button_style,
    unverified_button_style,
    default_style_time_range,
    active_style_time_range,
)


# Function to mask api_key in error toast
def mask_api_key_in_url(url):
    if not url:
        return url
    if "apiKey=" in url:
        base, key = url.split("apiKey=", 1)
        masked_key = "*" * len(key)
        return f"{base}apiKey={masked_key}"
    return url


# Upon successful verification, display metadata
@callback(
    [
        Output("verify-ticker", "data"),
        Output("portfolio-exploration-toast", "is_open"),
        Output("portfolio-exploration-toast", "children"),
        Output("portfolio-exploration-toast", "header"),
        Output("portfolio-exploration-toast", "icon"),
        Output("portfolio-exploration-toast", "style"),
        Output("portfolio-exploration-main-content", "children"),
    ],
    [
        Input("btn-verify-ticker", "n_clicks"),
        Input("inp-ticker", "value"),
    ],
    prevent_initial_call=True,
)
def verify_and_display(_, ticker):
    trigger_id = ctx.triggered_id

    # User is typing → reset verification
    if trigger_id == "inp-ticker":
        return ({"verified": False}, False, "", "", "", no_update, no_update)

    # User clicked "Verify Ticker"
    if trigger_id == "btn-verify-ticker":
        result = validate_stock_ticker(ticker, api_key)

        # If error → show red toast
        if "error" in result:
            return (
                {"verified": False},
                True,
                f"{mask_api_key_in_url(result['error'])}",
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
                no_update,
            )

        # Success → green success toast + metadata layout
        company_info = result["company_info"]
        address = result["address"]

        return (
            {"verified": True, **result},
            True,
            "Stock ticker verification successful!",
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
            company_metadata_layout(company_info, address, COLORS),
        )

    return no_update


# Change validation symbol upon successful verification
@callback(
    [
        Output("inp-ticker", "valid"),
        Output("inp-ticker", "invalid"),
        Output("inp-ticker", "key"),
    ],
    Input("verify-ticker", "data"),
    prevent_initial_call=True,
)
def set_ticker_validation(verify_ticker):
    is_verified = verify_ticker.get("verified", False)

    # Change the key so Dash forces a component refresh
    dynamic_key = f"key-{uuid.uuid4()}"

    return is_verified, not is_verified, dynamic_key


# Toggle styles between enabled/disabled status
@callback(
    [
        Output("btn-news", "disabled"),
        Output("btn-news", "style"),
        Output("btn-news", "className"),
        Output("btn-performance", "disabled"),
        Output("btn-performance", "style"),
        Output("btn-performance", "className"),
        Output("btn-add", "disabled"),
        Output("btn-add", "style"),
        Output("btn-add", "className"),
    ],
    Input("verify-ticker", "data"),
)
def toggle_button_states(verify_status):
    is_verified = verify_status.get("verified", False)

    if is_verified:
        return (
            False,
            verified_button_style,
            "simple",
            False,
            verified_button_style,
            "simple",
            False,
            verified_button_portfolio,
            "special",
        )

    else:
        return (
            True,
            unverified_button_style,
            "",
            True,
            unverified_button_style,
            "",
            True,
            unverified_button_portfolio,
            "",
        )


# Update main output section depending on what is chosen by user
@callback(
    [Output("portfolio-exploration-main-content", "children", allow_duplicate=True)],
    [
        Input("btn-news", "n_clicks"),
        Input("btn-performance", "n_clicks"),
        Input("btn-add", "n_clicks"),
        Input("selected-range", "data"),
    ],
    [State("verify-ticker", "data")],
    prevent_initial_call=True,
)
def update_main_output(_, __, ___, ____, cache_data):

    # Recovering cached data from API call
    company_info = cache_data["company_info"]
    news_articles = cache_data["news"]["results"]

    # Button clicked by user, one of the fllowing:
    button_id = ctx.triggered_id
    if button_id == "selected-range":
        raise dash.exceptions.PreventUpdate  # Prevent callback overlap

    # 1. Check Latest News
    if button_id == "btn-news":
        return (
            dbc.Container(
                children=[
                    html.H2(f"News Feed for {company_info['name']}", className="my-4"),
                    dbc.Container(
                        children=[
                            html.Div(
                                style={
                                    "maxHeight": "80vh",
                                    "overflowY": "scroll",
                                    "paddingRight": "10px",
                                },
                                children=[
                                    news_article_card_layout(article, COLORS)
                                    for article in news_articles
                                ],
                            )
                        ]
                    ),
                ],
                fluid=True,
            ),
        )

    # 2. Check Historic Performance
    elif button_id == "btn-performance":
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
                    dash_range_selector(default_style=default_style_time_range),
                    dcc.Tabs(
                        id="stock-performance-tabs",
                        value="tab-plot",
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
                                value="tab-plot",
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
                                value="tab-stats",
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
                        id="historical-plot-container",
                        style={"flex": "1", "overflow": "hidden"},
                    ),
                ],
            ),
        )

    # 3. Add to Portfolio
    elif button_id == "btn-add":
        return no_update

    return no_update


# Highlight buttons based on time-range selected
@callback(
    [
        Output("range-1M", "style"),
        Output("range-3M", "style"),
        Output("range-6M", "style"),
        Output("range-1Y", "style"),
        Output("range-2Y", "style"),
        Output("selected-range", "data"),
    ],
    [
        Input("range-1M", "n_clicks"),
        Input("range-3M", "n_clicks"),
        Input("range-6M", "n_clicks"),
        Input("range-1Y", "n_clicks"),
        Input("range-2Y", "n_clicks"),
    ],
)
def update_range_styles(*btn_clicks):
    button_ids = ["range-1M", "range-3M", "range-6M", "range-1Y", "range-2Y"]

    # If no button has been clicked yet, fall back to "All"
    if not any(click and click > 0 for click in btn_clicks):
        selected = "range-2Y"
    else:
        selected = ctx.triggered_id or "range-2Y"

    style_map = {btn_id: default_style_time_range for btn_id in button_ids}
    style_map[selected] = active_style_time_range

    return (
        style_map["range-1M"],
        style_map["range-3M"],
        style_map["range-6M"],
        style_map["range-1Y"],
        style_map["range-2Y"],
        selected.split("-")[1],
    )


# Update historic daily plot based on range selected
@callback(
    Output("historical-plot-container", "children"),
    [Input("stock-performance-tabs", "value"), Input("selected-range", "data")],
    State("verify-ticker", "data"),
    prevent_initial_call=True,
)
def update_plot_on_range_change(active_tab, selected_range, data):

    historical_df = pd.read_json(StringIO(data["historical_json"]), orient="records")
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
    dates = np.asarray(filtered_df["date"])
    daily_prices = np.asarray(filtered_df["close"])
    daily_log_returns = np.asarray(
        np.log(1 + filtered_df["close"].pct_change().dropna())
    )

    if active_tab == "tab-plot":
        return create_historic_plots(
            data["company_info"]["name"], dates, daily_prices, daily_log_returns, COLORS
        )

    elif active_tab == "tab-stats":
        return create_statistics_table(dates, daily_prices, daily_log_returns, COLORS)


# Upon "add to portfolio" click, append to table and cache data
@callback(
    Output("portfolio-store", "data"),
    Input("btn-add", "n_clicks"),
    [
        State("verify-ticker", "data"),
        State("portfolio-store", "data"),
    ],
    prevent_initial_call=True,
)
def add_to_portfolio(_, verify_data, portfolio_data):

    if not verify_data.get("verified"):
        return portfolio_data

    new_entry = {
        "ticker": verify_data["company_info"]["ticker"],
        "fullname": verify_data["company_info"]["name"],
        "sic_description": verify_data["company_info"]["sic_description"],
        "market_cap": verify_data["company_info"]["market_cap"],
        "historical_json": verify_data["historical_json"],
    }

    # Avoid duplicates
    if new_entry not in portfolio_data:
        portfolio_data.append(new_entry)

    return portfolio_data


# Update table visual
@callback(Output("portfolio-table", "data"), Input("portfolio-store", "data"))
def update_portfolio_table(data):
    return data


# Allow users to navigate to portfolio builder page
# Provided at least two tickers were selected
@callback(
    [
        Output("btn-portfolio-builder", "disabled"),
        Output("btn-portfolio-builder", "style"),
        Output("btn-portfolio-builder", "className"),
    ],
    Input("portfolio-store", "data"),
)
def update_portfolio_analytics_button(tickers):
    if tickers is None or len(tickers) < 2:
        return True, unverified_button_portfolio, ""

    return False, verified_button_portfolio, "special"
