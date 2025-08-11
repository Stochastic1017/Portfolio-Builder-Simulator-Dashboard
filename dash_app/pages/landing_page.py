import os
import sys
import dash

from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.button_styles import COLORS, verified_button_style

# Register the page
dash.register_page(__name__, path="/pages/landing-page")

layout = dbc.Container(
    [
        html.Div(
            [
                html.H1(
                    "Welcome to My App!",
                    className="animate__animated animate__fadeInDown typewriter",
                    style={
                        "textAlign": "center",
                        "marginTop": "15%",
                        "color": COLORS["primary"],
                        "fontWeight": "bold",
                    },
                ),
                html.Br(),
                html.Div(
                    [
                        html.P(
                            "This application was built by a single individual with a passion for coding, finance, mathematics, and statistics. "
                            "It is a personal project and does not represent the views or interests of any company or institution.",
                            className="animate__animated animate__fadeInLeft animate__delay-1s",
                            style={
                                "color": COLORS["text"],
                                "fontSize": "1rem",
                                "marginBottom": "0.5rem",
                            },
                        ),
                        html.Br(),
                        html.P(
                            "The insights provided are grounded in classical statistical theory, but rest on strong assumptions that often do not reflect real-world markets. "
                            "Please exercise caution and do not rely solely on this tool for financial decisions.",
                            className="animate__animated animate__fadeInRight animate__delay-2s",
                            style={
                                "color": COLORS["text"],
                                "fontSize": "1rem",
                                "marginBottom": "0.5rem",
                            },
                        ),
                        html.Br(),
                        html.P(
                            "All market data is sourced from the Polygon.io API. "
                            "The free tier limits requests to 10 API calls per minute, which allows approximately 3 tickers to be analyzed every 60 seconds.",
                            className="animate__animated animate__fadeInUp animate__delay-3s",
                            style={
                                "color": COLORS["text"],
                                "fontSize": "1rem",
                                "marginBottom": "0.5rem",
                            },
                        ),
                    ],
                    style={"maxWidth": "700px", "margin": "0 auto"},
                ),
                html.Br(),
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "marginTop": "2rem",
                    },
                    children=[
                        # Go to portfolio builder page
                        dcc.Link(
                            html.Button(
                                "Continue",
                                id="btn-continue",
                                n_clicks=0,
                                style=verified_button_style,
                                className="special animate__animated animate__fadeInUp animate__delay-5s",
                            ),
                            href="/pages/main-app",
                            refresh=False,  # Set to True if you want full page reload
                        )
                    ],
                ),
            ],
        ),
    ]
)
