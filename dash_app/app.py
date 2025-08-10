import os
import sys
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    use_pages=True,  # Enables multi-page support, automatically reads from pages/
    suppress_callback_exceptions=True,
)

##################
### Dev Only !!!
##################

if os.getenv("DEV_ONLY", "0").strip().lower() in {"1", "true", "yes"}:
    import json

    try:
        with open("temp.json") as f:
            temp = json.load(f)
    except FileNotFoundError:
        temp = []
else:
    temp = []

##################
### Dev Only !!!
##################

# App layout
app.layout = html.Div(
    children=[
        ################
        ### Cache store
        ################
        ### Stock exploration dashboard page
        dcc.Store(id="verify-ticker", data={"verified": False}),  # verify ticker
        dcc.Store(id="selected-range", data="2Y"),  # selected range
        dcc.Store(id="portfolio-store", data=temp),  # portfolio table
        ### Portfolio builder page
        dcc.Store(id="selected-tickers-store"),  # subset of tickers chosen
        dcc.Store(
            id="efficient-frontier-clicked", data=False
        ),  # efficient frontier clicked or not
        dcc.Store(id="portfolio-weights-store"),  # for all optimizations
        dcc.Store(
            id="confirmed-weights-store"
        ),  # for the selected/confirmed portfolio weights
        dcc.Store(
            id="portfolio-clicked-risk-return"
        ),  # temporary placeholder for risk/return
        dcc.Store(id="portfolio-risk-return"),  # risk/return for confirmed portfolio
        dcc.Store(id="latest-date"),  # latest date (last row value)
        ### Portfolio simulation page
        dcc.Store(id="verify-budget", data={"verified": False}),  # verify budget
        dcc.Store(id="budget-value"),  # budget (in $) input
        dcc.Store(
            id="portfolio-selected-range", data="portfolio-range-all"
        ),  # portfolio selected range
        #################
        ### Landing Page
        #################
        dcc.Location(id="url", refresh=True, pathname="/pages/portfolio-exploration"),
        dash.page_container,
    ]
)

server = app.server

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
