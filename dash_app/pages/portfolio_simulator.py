import os
import sys
import dash
import dash_bootstrap_components as dbc

from dash import html, dcc

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from helpers.button_styles import (
    COLORS,
    unverified_button_style,
    active_labelStyle_radioitems,
    active_inputStyle_radioitems,
    active_style_radioitems,
)

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
                        "overflow": "hidden",
                    },
                    children=[
                        # Buttons to explore portfolio weights and performance
                        html.Button(
                            id="btn-portfolio-performance",
                            disabled=True,
                            n_clicks=0,
                            style={
                                "all": "unset",  # removes default button styles
                                "cursor": "pointer",
                                "width": "100%",
                                "height": "12.5vh",
                            },
                            children=[
                                dbc.Card(
                                    style={
                                        "backgroundColor": COLORS["background"],
                                        "borderRadius": "10px",
                                        "padding": "12px 16px",
                                        "marginBottom": "16px",
                                        "width": "100%",
                                        "height": "12.5vh",
                                        "color": COLORS["text"],
                                        "fontSize": "0.9rem",
                                        "gap": "6px",
                                    },
                                    children=[
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "flexDirection": "column",
                                                "gap": "4px",
                                            },
                                            children=[
                                                html.Span(
                                                    "",
                                                    id="selected-portfolio-simulation",
                                                    style={
                                                        "fontWeight": "600",
                                                        "color": COLORS["primary"],
                                                        "fontSize": "0.95rem",
                                                    },
                                                ),
                                                html.Span(
                                                    "",
                                                    id="selected-range-simulation",
                                                    style={
                                                        "fontSize": "0.9rem",
                                                        "color": COLORS["primary"],
                                                    },
                                                ),
                                                html.Span(
                                                    "",
                                                    id="latest-date-simulation",
                                                    style={
                                                        "fontSize": "0.8rem",
                                                        "color": COLORS["secondary"],
                                                    },
                                                ),
                                            ],
                                        )
                                    ],
                                )
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
                                            "label": "Akaike Information Criterion",
                                            "value": "aic",
                                            "disabled": True,
                                        },
                                        {
                                            "label": "Bayesian Information Criterion",
                                            "value": "bic",
                                            "disabled": True,
                                        },
                                        {
                                            "label": "LogLikelihood",
                                            "value": "loglikelihood",
                                            "disabled": True,
                                        },
                                    ],
                                    labelStyle=active_labelStyle_radioitems,
                                    inputStyle=active_inputStyle_radioitems,
                                    style=active_style_radioitems,
                                ),
                                # ARIMA model
                                html.Button(
                                    "ARIMA Forecast",
                                    id="btn-arima-performance",
                                    style=unverified_button_style,
                                    disabled=True,
                                ),
                                # GARCH model
                                html.Button(
                                    "GARCH Forecast",
                                    id="btn-garch-performance",
                                    style=unverified_button_style,
                                    disabled=True,
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
                                # Button for user to start monte carlo exploration
                                html.Button(
                                    "Gradient Boosting Forecast",
                                    id="btn-gbm-performance",
                                    style=unverified_button_style,
                                    disabled=True,
                                ),
                                # Button for user to start monte carlo exploration
                                html.Button(
                                    "LSTM Forecast",
                                    id="btn-lstm-performance",
                                    style=unverified_button_style,
                                    disabled=True,
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
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
        ##################
        ### Summary Table
        ##################
        html.Div(
            id="summary-forecast-simulator",
            style={
                "marginTop": "20px",
                "padding": "20px",
                "backgroundColor": COLORS["background"],
                "borderRadius": "10px",
                "marginLeft": "auto",
                "marginRight": "auto",
                "justifyContent": "center",
                "alignItems": "center",
            },
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
