import os
import re
import sys
import uuid
import dash
import numpy as np
import dash_bootstrap_components as dbc

from datetime import datetime, timedelta
from dash import html, Input, Output, State, ALL, MATCH, callback, ctx, dcc, no_update

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from helpers.button_styles import COLORS, verified_button_style

dash.register_page(__name__, path="/pages/portfolio-simulator")

layout = html.Div(
    style={
        "background": COLORS["background"],
        "minHeight": "100vh",
        "padding": "20px",
        "color": COLORS["text"],
        "fontFamily": '"Inter", system-ui, -apple-system, sans-serif',
    },
    children=[
        #################
        ### Pop-up Toast
        #################
        # Toast messages to inform users of success/failures
        dbc.Toast(
            id="portfolio-simulator-toast",
            header="Success",
            icon="success",
            is_open=False,
            duration=4000,
            dismissable=True,
        ),
        #####################
        ### Left console
        ### Right content
        #####################
        # Left console + Right content
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "320px 1fr",
                "gap": "20px",
                "height": "100vh",
                "padding": "20px",
                "boxSizing": "border-box",
            },
            children=[
                # Left console
                html.Div(
                    id="portfolio-simulator-console",
                    style={
                        "backgroundColor": COLORS["card"],
                        "borderRadius": "10px",
                        "padding": "20px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "15px",
                        "height": "100%",
                        "boxSizing": "border-box",
                        "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.1)",
                        "overflow": "hidden",
                    },
                    children=[
                        # Budget (in $) input + verify budget button
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "alignItems": "center",
                            },
                            children=[
                                # Input for budget (in $)
                                dbc.Input(
                                    id="inp-budget",
                                    type="text",
                                    debounce=True,
                                    valid=False,
                                    invalid=False,
                                    key="input-key",
                                    placeholder="Enter Budget (in $)",
                                    className="custom-input",
                                    style={
                                        "width": "200px",
                                        "padding": "10px",
                                        "backgroundColor": COLORS["background"],
                                        "border": f"1px solid {COLORS['primary']}",
                                        "borderRadius": "5px",
                                        "color": COLORS["text"],
                                        "fontSize": "1em",
                                        "marginRight": "10px",
                                    },
                                ),
                                # A stylized button to verify if user input budget is correct
                                html.Button(
                                    "Verify Budget",
                                    id="btn-verify-budget",
                                    n_clicks=0,
                                    disabled=False,
                                    className="special",
                                    style={
                                        "padding": "6px 12px",
                                        "backgroundColor": COLORS["primary"],
                                        "color": "black",
                                        "border": "1px solid #9370DB",
                                        "borderRadius": "20px",
                                        "fontWeight": "bold",
                                        "fontSize": "0.75em",
                                        "cursor": "pointer",
                                        "alignSelf": "flex-start",
                                        "transition": "all 0.2s ease-in-out",
                                    },
                                ),
                            ],
                        ),
                        # Buttons to explore portfolio weights and performance
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "10px",
                            },
                            children=[
                                # Button for user to explore past performance of portfolio
                                html.Button(
                                    "Evaluate Past Performance",
                                    id="btn-portfolio-performance",
                                    style=verified_button_style,
                                    disabled=False,
                                    className="simple",
                                ),
                            ],
                        ),
                        # Date picker and Ensemble Generator for prediction
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "10px",
                                "width": "100%",
                            },
                            children=[
                                html.Label(
                                    "Choose a future date for prediction:",
                                    style={
                                        "color": COLORS["primary"],
                                        "fontWeight": "bold",
                                        "fontSize": "1rem",
                                    },
                                ),
                                dcc.DatePickerSingle(
                                    id="date-chooser-simulation",
                                    disabled=True,
                                    with_portal=True,
                                    display_format="MMM Do, YY",
                                ),
                                html.Br(),
                                html.Label(
                                    "Number of ensembles to generate:",
                                    style={
                                        "color": COLORS["primary"],
                                        "fontWeight": "bold",
                                        "marginBottom": "10px",
                                        "fontSize": "1rem",
                                    },
                                ),
                                # Slider to choose number of ensembles to generate
                                dcc.Slider(
                                    id="num-ensemble-slider",
                                    min=10,
                                    max=500,
                                    value=100,
                                    disabled=True,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                    },
                                ),
                            ],
                        ),
                        # Criterion selection and ARIMA/GARCH modeling
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "10px",
                            },
                            children=[
                                html.Label(
                                    "Classical Forecasting:",
                                    style={
                                        "color": COLORS["primary"],
                                        "fontWeight": "bold",
                                        "fontSize": "1rem",
                                    },
                                ),
                                dcc.RadioItems(
                                    id="model-selection-criterion",
                                    options=[
                                        {
                                            "label": "AIC (Akaike)",
                                            "value": "aic",
                                            "disabled": True,
                                        },
                                        {
                                            "label": "BIC (Bayesian)",
                                            "value": "bic",
                                            "disabled": True,
                                        },
                                        {
                                            "label": "LogLikelihood",
                                            "value": "loglikelihood",
                                            "disabled": True,
                                        },
                                    ],
                                ),
                                html.Br(),
                                # ARIMA model
                                html.Button(
                                    "ARIMA Forecast",
                                    id="btn-arima-performance",
                                    style=verified_button_style,
                                    disabled=False,
                                    className="simple",
                                ),
                                # GARCH model
                                html.Button(
                                    "GARCH Forecast",
                                    id="btn-garch-performance",
                                    style=verified_button_style,
                                    disabled=False,
                                    className="simple",
                                ),
                                html.Br(),
                            ],
                        ),
                        # Buttons to generate LSTM and GBM predictions
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "10px",
                            },
                            children=[
                                html.Label(
                                    "Machine Learning Forecasting:",
                                    style={
                                        "color": COLORS["primary"],
                                        "fontWeight": "bold",
                                        "fontSize": "1rem",
                                    },
                                ),
                                html.Br(),
                                # Button for user to start monte carlo exploration
                                html.Button(
                                    "Gradient Boosting Forecast",
                                    id="btn-gbm-performance",
                                    style=verified_button_style,
                                    disabled=False,
                                    className="simple",
                                ),
                                # Button for user to start monte carlo exploration
                                html.Button(
                                    "LSTM Forecast",
                                    id="btn-lstm-performance",
                                    style=verified_button_style,
                                    disabled=False,
                                    className="simple",
                                ),
                            ],
                        ),
                    ],
                ),
                # Right content
                html.Div(
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "height": "100%",
                        "minHeight": 0,
                    },
                    children=[
                        dcc.Loading(
                            id="portfolio-exploration-loading",
                            type="cube",
                            color=COLORS["primary"],
                            style={
                                "height": "50%",
                                "width": "100%",
                                "display": "flex",
                                "flexDirection": "column",
                                "flex": "1",
                                "minHeight": 0,
                            },
                            children=[
                                html.Div(
                                    id="portfolio-simulator-main-content",
                                    style={
                                        "backgroundColor": COLORS["background"],
                                        "borderRadius": "10px",
                                        "height": "100vh",
                                        "width": "100%",
                                        "boxSizing": "border-box",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "justifyContent": "center",
                                        "alignItems": "center",
                                        "textAlign": "center",
                                        "padding": "2rem",
                                        "overflow": "hidden",
                                    },
                                    children=[
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Welcome to Portfolio Simulator Page!",
                                                    style={
                                                        "color": COLORS["primary"],
                                                        "marginBottom": "1rem",
                                                    },
                                                ),
                                                html.Br(),
                                                html.Div(
                                                    [
                                                        html.P(
                                                            "To simulate the portfolio, please follow the steps below:",
                                                            style={
                                                                "color": COLORS["text"],
                                                                "fontSize": "1.1rem",
                                                            },
                                                        ),
                                                        html.Ol(
                                                            [
                                                                html.Li(
                                                                    "Input a budget (in $) and click 'Verify Budget'.",
                                                                    style={
                                                                        "color": COLORS[
                                                                            "text"
                                                                        ]
                                                                    },
                                                                ),
                                                                html.Li(
                                                                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                                                                    style={
                                                                        "color": COLORS[
                                                                            "text"
                                                                        ]
                                                                    },
                                                                ),
                                                                html.Li(
                                                                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                                                                    style={
                                                                        "color": COLORS[
                                                                            "text"
                                                                        ]
                                                                    },
                                                                ),
                                                                html.Li(
                                                                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                                                                    style={
                                                                        "color": COLORS[
                                                                            "text"
                                                                        ]
                                                                    },
                                                                ),
                                                            ],
                                                            style={
                                                                "textAlign": "left",
                                                                "color": COLORS["text"],
                                                                "maxWidth": "600px",
                                                                "margin": "1rem auto",
                                                            },
                                                        ),
                                                    ],
                                                    style={"maxWidth": "700px"},
                                                ),
                                            ]
                                        )
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
        ################
        ### Page Footer
        ################
        html.Footer(
            style={
                "backgroundColor": COLORS["background"],
                "padding": "8px 12px",
                "textAlign": "center",
                "fontsize": "0.8em",
                "borderRadius": "8px",
                "marginTop": "12px",
                "lineheight": "1.2",
            },
            children=[
                html.P(
                    "Developed by Shrivats Sudhir | Contact: shrivats.sudhir@gmail.com"
                ),
                html.P(
                    [
                        "GitHub Repository: ",
                        html.A(
                            "Portfolio Optimization and Visualization Dashboard",
                            href="https://github.com/Stochastic1017/Portfolio-Analysis-Dashboard",
                            target="_blank",
                            style={
                                "color": COLORS["primary"],
                                "textDecoration": "none",
                            },
                        ),
                    ]
                ),
            ],
        ),
    ],
)
