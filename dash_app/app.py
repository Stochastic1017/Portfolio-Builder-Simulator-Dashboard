
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
    suppress_callback_exceptions=True
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

        # Stock exploration dashboard page
        dcc.Store(id="verify-status", data={"verified": False}), # verify status 
        dcc.Store(id="selected-range", data="range-all"),        # selected range
        dcc.Store(id='portfolio-store', data=temp),              # portfolio table
                    
        #################
        ### Landing Page
        #################

        dcc.Location(id="url", refresh=True, pathname="/pages/stock-exploration"),
        dash.page_container
    ]
)

server = app.server

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8050, debug=True)
