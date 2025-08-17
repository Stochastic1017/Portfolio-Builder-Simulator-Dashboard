import os
import sys
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import callbacks.portfolio_exploration
import callbacks.portfolio_builder
import callbacks.portfolio_simulator

# Create the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css",
    ],
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
            location="main-app"
    except FileNotFoundError:
        temp = []
        location="main-app"
else:
    temp = []
    location="landing-page"

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
        dcc.Store(id="selected-range", data="range-2Y"),  # selected range
        dcc.Store(id="portfolio-store", data=temp),  # portfolio table
        ### Portfolio builder page
        dcc.Store(id="verify-budget", data={"verified": False}),  # verify budget
        dcc.Store(id="budget-value"),  # budget (in $) input
        dcc.Store(id="selected-tickers-store"),  # subset of tickers chosen
        dcc.Store(id="portfolio-weights-store"),  # for all optimizations
        dcc.Store(
            id="confirmed-weights-store"
        ),  # for the selected/confirmed portfolio weights
        dcc.Store(
            id="portfolio-clicked-risk-return"
        ),  # temporary placeholder for risk/return
        dcc.Store(id="portfolio-risk-return"),  # risk/return for confirmed portfolio
        dcc.Store(id="latest-date"),  # latest date (last row value)
        dcc.Store(
            id="efficient-frontier-selected-range", data="efficient-frontier-range-2Y"
        ),
        ### Portfolio simulation page
        dcc.Store(
            id="portfolio-selected-range", data="portfolio-range-2Y"
        ),  # portfolio selected range
        #################
        ### Landing Page
        #################
        dcc.Location(id="url", refresh=True, pathname=f"/pages/{location}"),
        dash.page_container,
    ]
)

server = app.server

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
