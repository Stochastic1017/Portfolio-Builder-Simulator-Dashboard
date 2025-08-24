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


# App layout
app.layout = html.Div(
    children=[
        ################
        ### Cache store
        ################
        ### Stock exploration dashboard page
        dcc.Store(id="verify-ticker", data={"verified": False}),
        dcc.Store(id="selected-range", data="range-2Y"),
        dcc.Store(id="portfolio-store", data=[]),
        ### Portfolio builder page
        dcc.Store(id="verify-budget", data={"verified": False}),
        dcc.Store(id="budget-value"),
        dcc.Store(id="selected-tickers-store"),
        dcc.Store(id="portfolio-weights-store"),
        dcc.Store(
            id="efficient-frontier-selected-range", data="efficient-frontier-range-2Y"
        ),
        dcc.Store(id="confirmed-weights-store"),
        dcc.Store(id="portfolio-clicked-risk-return"),
        dcc.Store(id="confirmed-portfolio-details"),
        dcc.Store(id="latest-date"),
        dcc.Store(id="portfolio-confirmed", data=False),
        ### Portfolio simulation page
        dcc.Store(
            id="portfolio-selected-range", data="portfolio-range-2Y"
        ),  # portfolio selected range
        dcc.Store(id="simulated-forecasts", data=None),
        #################
        ### Landing Page
        #################
        dcc.Location(id="url", refresh=True, pathname=f"/pages/landing-page"),
        dash.page_container,
    ]
)

server = app.server

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True, threaded=True)
