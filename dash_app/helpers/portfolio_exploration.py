import os
import sys
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

from scipy import stats
from dash import dcc, html, dash_table
from plotly.subplots import make_subplots

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class StockTickerInformation:

    def __init__(self, ticker, api_key):
        """
        Initializing function.

        Args:
            ticker: unique abbreviation of publicly traded stock ticker.
            api_key: Polygon.io api key
        """

        self.ticker = ticker
        self.api_key = api_key

    def get_metadata(self):
        """
        Retrieve detailed reference information for a specific stock ticker from the Polygon.io API.

        Returns:
            dict: A dictionary containing metadata about the specified stock ticker.

        Raises:
            requests.RequestException: If the HTTP request fails.
        """

        try:
            url = f"https://api.polygon.io/v3/reference/tickers/{self.ticker}?apiKey={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:
            raise Exception(f"HTTP request failed: {error}")

    def get_news(self):

        try:
            url = f"https://api.polygon.io/v2/reference/news?ticker={self.ticker}&apiKey={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:
            raise Exception(f"HTTP request failed: {error}")

    def get_all_data(self):
        """
        Fetches historical daily aggregated stock data for a given ticker symbol within a specified date range
        using the Polygon.io API.

        Parameters:
            start_date (str): The start date of the data range in 'YYYY-MM-DD' format.
            end_date (str): The end date of the data range in 'YYYY-MM-DD' format.

        Returns:
            Pandas DataFrame containing daily stock data with the date, open, high, low, close, and volume.

        Raises:
            Exception: If the API response does not contain the expected 'results' key or if another error occurs.
        """

        start_date = "2000-01-01"
        end_date = datetime.today().strftime("%Y-%m-%d")
        url = f"https://api.polygon.io/v2/aggs/ticker/{self.ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if "results" not in data:
                raise Exception(f"Polygon API error: {data}")

            df = pd.DataFrame(data["results"])
            df["t"] = pd.to_datetime(df["t"], unit="ms")  # timestamp to datetime
            df = df.rename(
                columns={
                    "t": "date",
                    "c": "close",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "v": "volume",
                }
            )

            return df.sort_values(by="date")

        except requests.RequestException as error:
            raise Exception(f"HTTP request failed: {error}")


# Stock ticker validation procedure
def validate_stock_ticker(ticker, api_key):

    if not ticker or not ticker.strip():
        return {"error": "Ticker is empty or invalid."}

    try:
        polygon_api = StockTickerInformation(
            ticker=ticker.upper().strip(), api_key=api_key
        )

        # Metadata
        metadata = polygon_api.get_metadata()
        company_info = metadata.get("results", {}) if metadata else {}

        # Address
        address = company_info.get("address", {})

        # News
        news = polygon_api.get_news() or []

        # Historical performance (hard fail if missing essential fields)
        historical_df = polygon_api.get_all_data()
        if historical_df is None or historical_df.empty:
            return {"error": "Historical data missing."}

        if not {"close", "date"}.issubset(historical_df.columns):
            return {"error": "Historical data incomplete (missing close or date)."}

        return {
            "verified": True,
            "ticker": ticker,
            "company_info": company_info or None,
            "address": address or None,
            "news": news or None,
            "historical_json": historical_df.to_json(orient="records"),
        }

    except Exception as e:
        return {"error": f"Ticker verification failed: {str(e)}"}


def company_metadata_layout(company_info, address, COLORS):
    # Placeholder helper
    def placeholder(text):
        return html.Span(
            text,
            style={
                "color": COLORS.get("warning", "#FF5733"),  # orange-ish by default
                "fontStyle": "italic",
            },
        )

    return html.Div(
        style={
            "padding": "30px",
            "height": "100%",
            "width": "100%",
            "boxSizing": "border-box",
            "overflowY": "auto",
            "display": "flex",
            "flexDirection": "column",
            "gap": "20px",
        },
        children=[
            # Company name and logo
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "20px"},
                children=[
                    html.H2(
                        company_info.get("name", "Name Not Found"),
                        style={"margin": 0, "color": COLORS["primary"]},
                    ),
                ],
            ),
            # Description
            html.P(
                company_info.get("description", "Description Not Found"),
                style={
                    "fontSize": "1.1em",
                    "textAlign": "justify",
                    "textAlignLast": "left",
                    "fontStyle": (
                        "italic" if not company_info.get("description") else "normal"
                    ),
                    "color": (
                        "#666" if not company_info.get("description") else "inherit"
                    ),
                },
            ),
            # Basic info section
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "15px",
                },
                children=[
                    html.Div(
                        [
                            html.Strong("Ticker: "),
                            company_info.get("ticker") or placeholder("Not Found"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Market: "),
                            company_info.get("market") or placeholder("Not Found"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Exchange: "),
                            company_info.get("primary_exchange")
                            or placeholder("Not Found"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Locale: "),
                            company_info.get("locale", "").upper()
                            or placeholder("Not Found"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Type: "),
                            company_info.get("type") or placeholder("Not Found"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Active: "),
                            (
                                str(company_info.get("active"))
                                if "active" in company_info
                                else placeholder("Not Found")
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Currency: "),
                            company_info.get("currency_name", "").upper()
                            or placeholder("Not Found"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("SIC: "),
                            (
                                f"{company_info.get('sic_code', '')} – {company_info.get('sic_description', '')}"
                                if company_info.get("sic_code")
                                else placeholder("Not Found")
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Market Cap: "),
                            (
                                f"${company_info.get('market_cap', 0):,.0f}"
                                if company_info.get("market_cap")
                                else placeholder("Not Found")
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Employees: "),
                            (
                                f"{company_info.get('total_employees', 'N/A'):,}"
                                if company_info.get("total_employees")
                                else placeholder("Not Found")
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("List Date: "),
                            company_info.get("list_date") or placeholder("Not Found"),
                        ]
                    ),
                ],
            ),
            # Address and Contact
            html.Div(
                [
                    html.H4(
                        "Contact",
                        style={"color": COLORS["primary"], "marginTop": "20px"},
                    ),
                    html.P(
                        company_info.get("phone_number")
                        or placeholder("Phone Not Found")
                    ),
                    html.P(
                        f"{address.get('address1', '')}, {address.get('city', '')}, {address.get('state', '')} {address.get('postal_code', '')}"
                        if address
                        else placeholder("Address Not Found")
                    ),
                    html.A(
                        (
                            "Visit Website"
                            if company_info.get("homepage_url")
                            else "Website Not Found"
                        ),
                        href=company_info.get("homepage_url") or "#",
                        target="_blank",
                        style={
                            "color": (
                                COLORS["primary"]
                                if company_info.get("homepage_url")
                                else "#999"
                            ),
                            "textDecoration": (
                                "underline"
                                if company_info.get("homepage_url")
                                else "none"
                            ),
                            "pointerEvents": (
                                "auto" if company_info.get("homepage_url") else "none"
                            ),
                        },
                    ),
                ]
            ),
        ],
    )


def dash_range_selector(default_style):

    return html.Div(
        id="range-selector-container",
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
                        id="range-1M",
                        n_clicks=0,
                        style=default_style,
                        className="simple",
                    ),
                    html.Button(
                        "3M",
                        id="range-3M",
                        n_clicks=0,
                        style=default_style,
                        className="simple",
                    ),
                    html.Button(
                        "6M",
                        id="range-6M",
                        n_clicks=0,
                        style=default_style,
                        className="simple",
                    ),
                    html.Button(
                        "1Y",
                        id="range-1Y",
                        n_clicks=0,
                        style=default_style,
                        className="simple",
                    ),
                    html.Button(
                        "2Y",
                        id="range-2Y",
                        n_clicks=0,
                        style=default_style,
                        className="simple",
                    ),
                ],
            )
        ],
    )


def create_historic_plots(
    full_name, dates, daily_prices, daily_log_returns, selected_range, COLORS
):

    # Map selected_range codes to readable text
    range_labels = {
        "1M": "past 1 month",
        "3M": "past 3 months",
        "6M": "past 6 months",
        "1Y": "past 1 year",
        "2Y": "past 2 years",
    }
    period_text = range_labels.get(selected_range, f"past {selected_range}")

    ######################
    ### Defining Subplots
    ######################

    # Create subplots
    historical_daily_plot = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=[
            "Daily Prices",
            "Daily Log Returns with 95% CI",
            "Histogram of Daily Log Returns with 95% CI",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        shared_xaxes=False,
        shared_yaxes=False,
        column_widths=[0.7, 0.3],
    )

    ################
    ### Price Plot
    ################

    # Add traces for closing prices
    historical_daily_plot.add_trace(
        go.Scatter(
            x=dates,
            y=daily_prices,
            mode="lines",
            name="Daily Closing Prices",
            line=dict(color=COLORS["primary"]),
        ),
        row=1,
        col=1,
    )

    #################################
    ### Line Plot for Daily Returns
    ### with 95% Confidence Interval
    #################################

    # Calculate the two-sided 95% confidence bounds
    mean_return = np.mean(daily_log_returns)
    std_return = np.std(daily_log_returns)
    lower_bound = mean_return - 1.96 * std_return
    upper_bound = mean_return + 1.96 * std_return

    # Add the line plot for daily returns vs day
    historical_daily_plot.add_trace(
        go.Scatter(
            x=dates,
            y=daily_log_returns,
            mode="lines",
            name="Daily Log Returns",
            line=dict(color=COLORS["primary"]),
        ),
        row=2,
        col=1,
    )

    # Add horizontal line for the upper bound
    historical_daily_plot.add_trace(
        go.Scatter(
            x=dates,
            y=[upper_bound] * len(dates),
            mode="lines",
            name="Upper 95% Confidence Bound",
            line=dict(color="rgba(255,255,255,0.5)"),
        ),
        row=2,
        col=1,
    )

    # Add horizontal line for the lower bound
    historical_daily_plot.add_trace(
        go.Scatter(
            x=dates,
            y=[lower_bound] * len(dates),
            mode="lines",
            name="Upper 95% Confidence Bound",
            line=dict(color="rgba(255,255,255,0.5)"),
            fill="tonexty",
            fillcolor="rgba(255,255,255,0.05)",
        ),
        row=2,
        col=1,
    )

    ##############################
    ### Histogram + Gaussian Fit
    ##############################

    # Combined range to align both histogram and normal PDF
    return_range = np.linspace(daily_log_returns.min(), daily_log_returns.max(), 500)
    hist_counts, _ = np.histogram(daily_log_returns, bins=50, density=True)

    # Histogram of daily returns
    historical_daily_plot.add_trace(
        go.Histogram(
            y=daily_log_returns,
            name="Daily Log Returns Histogram",
            marker_color=COLORS["primary"],
            nbinsy=50,
            orientation="h",
            opacity=0.6,
            histnorm="probability density",
        ),
        row=2,
        col=2,
    )

    # Gaussian fit using sample estimates
    historical_daily_plot.add_trace(
        go.Scatter(
            x=stats.norm.pdf(return_range, mean_return, std_return),
            y=return_range,
            mode="lines",
            name=f"Gaussian fit with sample estimates",
            line=dict(color="#E0E0E0", width=4),
            showlegend=True,
        ),
        row=2,
        col=2,
    )

    # Horizontal line for upper bound of 95% confidence
    historical_daily_plot.add_trace(
        go.Scatter(
            x=[0, hist_counts.max()],
            y=[upper_bound] * len(dates),
            mode="lines",
            name="Upper 95% Bound",
            line=dict(color="rgba(255,255,255,0.5)"),
            showlegend=True,
        ),
        row=2,
        col=2,
    )

    # Horizontal line for lower bound of 95% confidence
    historical_daily_plot.add_trace(
        go.Scatter(
            x=[0, hist_counts.max()],
            y=[lower_bound] * len(dates),
            mode="lines",
            name="Lower 95% Bound",
            line=dict(color="rgba(255,255,255,0.5)"),
            fill="tonexty",
            fillcolor="rgba(255,255,255,0.05)",
            showlegend=True,
        ),
        row=2,
        col=2,
    )

    # Calculate last change
    last_price = daily_prices[-1]
    first_price = daily_prices[1]
    change_abs = last_price - first_price
    change_pct = (change_abs / first_price) * 100
    arrow = "↑" if change_abs > 0 else "↓" if change_abs < 0 else "→"

    # Format with sign
    change_text = f"{change_abs:+.2f} ({change_pct:+.2f}%) {arrow} {period_text}"
    change_color = (
        "limegreen" if change_abs > 0 else "red" if change_abs < 0 else "gray"
    )

    historical_daily_plot.update_layout(
        template="plotly_dark",
        title=f"Historical Daily Performance Analysis for {full_name}",
        title_font=dict(color=COLORS["primary"]),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        showlegend=False,
        autosize=True,
        height=None,
        annotations=[
            dict(
                text=change_text,
                xref="paper",
                yref="paper",
                x=1,  
                y=1.15,  
                showarrow=False,
                font=dict(color=change_color, size=16, family="Arial Black"),
                xanchor="right",
            )
        ],
    )

    # Format y-axes as percentage
    historical_daily_plot.update_yaxes(
        tickformat=".2%", row=2, col=1
    )  # Returns line plot
    historical_daily_plot.update_yaxes(
        tickformat=".2%", row=2, col=2
    )  # Histogram of returns

    return dcc.Graph(
        id="historic-performance-plot",
        figure=historical_daily_plot,
        config={"responsive": True},
        style={"height": "100%", "width": "100%"},
    )


def summarize_daily_returns(dates, daily_prices, daily_log_returns, selected_range):

    # Map selected_range codes to readable text
    range_labels = {
        "1M": "past 1 month",
        "3M": "past 3 months",
        "6M": "past 6 months",
        "1Y": "past 1 year",
        "2Y": "past 2 years",
    }
    period_text = range_labels.get(selected_range, f"past {selected_range}")

    dates = pd.Series(dates).dropna()
    daily_prices = pd.Series(daily_prices).dropna()
    daily_log_returns = pd.Series(daily_log_returns).dropna()
    change_abs = daily_prices.iloc[-1] - daily_prices.iloc[1]
    change_pct = (change_abs / daily_prices[1]) * 100
    arrow = "↑" if change_abs > 0 else "↓" if change_abs < 0 else "→"
    change_text = f"{change_abs:+.2f} ({change_pct:+.2f}%) {arrow} {period_text}"

    # Basic stats
    latest_date = dates.iloc[-1]
    latest_price = daily_prices.iloc[-1]
    latest_returns = daily_log_returns.iloc[-1]
    mean = daily_log_returns.mean()
    median = daily_log_returns.median()
    std = daily_log_returns.std()
    var = daily_log_returns.var()
    skew = daily_log_returns.skew()
    kurt = daily_log_returns.kurt()
    min_val = daily_log_returns.min()
    max_val = daily_log_returns.max()

    # Hypothesis test: mean = 0
    t_stat, p_val = stats.ttest_1samp(daily_log_returns, popmean=0)

    # Normality test (Shapiro-Wilk is good for <5000 samples)
    normality_test = stats.shapiro(daily_log_returns)
    normal_stat, normal_pval = normality_test

    return {
        "Latest Date": latest_date.strftime("%A, %d %B %Y"),
        "Latest Price": latest_price,
        "Latest Log Return": latest_returns,
        "Price / Percentage Change": change_text,
        "ChangeAbs": change_abs,
        "Mean Log Returns": mean,
        "Median Log Returns": median,
        "Standard Deviation Log Returns": std,
        "Variance Log Returns": var,
        "Skewness Log Returns": skew,
        "Kurtosis Log Returns": kurt,
        "Minimum Log Returns": min_val,
        "Maximum Log Returns": max_val,
        "t-Statistic (Mean Log Returns = 0)": t_stat,
        "p-Value (Mean Log Returns = 0)": p_val,
        f"Shapiro-Wilk Test Statistic": normal_stat,
        f"Shapiro-Wilk Test p-value": normal_pval,
    }


def create_statistics_table(
    dates, daily_prices, daily_log_returns, selected_range, COLORS
):

    stats_dict = summarize_daily_returns(
        dates, daily_prices, daily_log_returns, selected_range
    )

    sections = {
        "Current Price and Returns": [
            "Latest Date",
            "Latest Price",
            "Latest Log Return",
            "Price / Percentage Change",
        ],
        "Sample Statistics": [
            "Mean Log Returns",
            "Median Log Returns",
            "Standard Deviation Log Returns",
            "Variance Log Returns",
            "Skewness Log Returns",
            "Kurtosis Log Returns",
            "Minimum Log Returns",
            "Maximum Log Returns",
        ],
        "Hypothesis Test (Mean Log Returns = 0)": [
            "t-Statistic (Mean Log Returns = 0)",
            "p-Value (Mean Log Returns = 0)",
        ],
        "Normality Log Returns Test": [
            key for key in stats_dict.keys() if "Test" in key
        ],
    }

    table_rows = []
    for section, metrics in sections.items():
        table_rows.append({"Metric": section, "Value": "", "ChangeAbs": None})
        for metric in metrics:
            val = stats_dict.get(metric, "")
            change_abs_val = (
                stats_dict.get("ChangeAbs", None)
                if metric == "Price / Percentage Change"
                else None
            )
            table_rows.append(
                {"Metric": metric, "Value": val, "ChangeAbs": change_abs_val}
            )

    return html.Div(
        style={
            "flex": "1",
            "display": "flex",
            "flexDirection": "column",
            "minHeight": 0,
            "maxHeight": "500px",
        },
        children=[
            dash_table.DataTable(
                data=table_rows,
                columns=[
                    {"name": "Metric", "id": "Metric"},
                    {"name": "Value", "id": "Value"},
                ],
                style_cell={
                    "textAlign": "left",
                    "padding": "8px",
                    "color": "white",
                    "border": "none",
                    "backgroundColor": COLORS["background"],
                    "fontFamily": "monospace",
                },
                style_data_conditional=[
                    {
                        "if": {"state": "selected"},
                        "backgroundColor": "inherit !important",
                        "border": "inherit !important",
                    },
                    {
                        "if": {"filter_query": '{Value} = ""'},
                        "fontWeight": "bold",
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["primary"],
                    },
                    {
                        "if": {
                            "filter_query": '{Metric} = "Price / Percentage Change" && {ChangeAbs} > 0',
                            "column_id": "Value",
                        },
                        "color": "limegreen",
                        "fontWeight": "bold",
                    },
                    {
                        "if": {
                            "filter_query": '{Metric} = "Price / Percentage Change" && {ChangeAbs} < 0',
                            "column_id": "Value",
                        },
                        "color": "red",
                        "fontWeight": "bold",
                    },
                ],
                style_header={"display": "none"},
                style_table={
                    "flex": "1",
                    "overflowY": "auto",
                    "maxHeight": "75vh",
                    "marginTop": "10px",
                    "fontSize": "15px",
                    "marginBottom": "10px",
                    "padding": "10px 20px",
                },
                editable=False,
                row_selectable=False,
                selected_rows=[],
                active_cell=None,
                cell_selectable=False,
                sort_action="none",
                filter_action="none",
                page_action="none",
                style_as_list_view=True,
            )
        ],
    )


def news_article_card_layout(article, COLORS):

    sentiment_color = {
        "positive": "success",
        "neutral": "secondary",
        "negative": "danger",
    }
    sentiment = article["insights"][0]["sentiment"]

    return dbc.Card(
        [
            dbc.CardHeader(
                style={
                    "backgroundColor": COLORS["card"],
                    "color": COLORS["text"],
                    "borderBottom": "1px solid #444",
                },
                children=[
                    ### Publisher Details
                    # Publisher logo image
                    html.Img(
                        src=article["publisher"]["favicon_url"],
                        height="20px",
                        style={"marginRight": "10px"},
                    ),
                    # Publisher name
                    html.A(
                        article["publisher"]["name"],
                        href=article["publisher"]["homepage_url"],
                        target="_blank",
                        className="me-2",
                        style={"color": COLORS["primary"]},
                    ),
                    # Sentiment of news article
                    dbc.Badge(
                        sentiment.capitalize(),
                        color=sentiment_color.get(sentiment, "secondary"),
                        className="float-end",
                    ),
                ],
            ),
            # Article title
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Span(article["title"], className="card-title"),
                            html.A(
                                " 🔗",
                                href=article["article_url"],
                                target="_blank",
                                style={"textDecoration": "none"},
                            ),
                        ],
                        style={
                            "color": COLORS["primary"],
                            "textAlign": "center",
                            "display": "flex",
                            "fontWeight": "bold",
                            "fontSize": 25,
                            "justifyContent": "center",
                            "alignItems": "center",
                            "gap": "5px",
                        },
                    ),
                    html.P(
                        f"By {article['author']} | Published: {article['published_utc'][:10]}",
                        style={
                            "color": "white",
                            "textAlign": "center",
                            "display": "block",
                            "width": "100%",
                        },
                    ),
                    html.Img(
                        src=article["image_url"],
                        style={
                            "width": "100%",
                            "marginTop": "10px",
                            "marginBottom": "10px",
                        },
                    ),
                    html.P(article["description"], style={"color": COLORS["text"]}),
                    html.Div(
                        [
                            dbc.Badge(
                                keyword,
                                color="warning",
                                className="me-1 mb-1",
                                style={
                                    "backgroundColor": COLORS["primary"],
                                    "color": COLORS["background"],
                                },
                            )
                            for keyword in article["keywords"]
                        ]
                    ),
                ],
                style={"backgroundColor": COLORS["card"], "color": COLORS["text"]},
            ),
        ],
        className="mb-4 shadow-sm",
        style={"backgroundColor": COLORS["card"], "border": "none"},
    )
