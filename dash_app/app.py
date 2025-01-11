
import os
import sys
import dash

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dash import dcc, html

# Create the Dash app
app = dash.Dash(
    __name__,
    use_pages=True,  # Enables multi-page support, automatically reads from pages/
    suppress_callback_exceptions=True
)

# App layout
app.layout = html.Div(
    children=[
        # Default URL path set to "/portfolio"
        dcc.Location(id="url", refresh=True, pathname="/portfolio"),
        dash.page_container
    ]
)

server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
