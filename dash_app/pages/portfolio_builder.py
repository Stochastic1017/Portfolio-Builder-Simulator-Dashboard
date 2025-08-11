import os
import sys
import json
import dash
import dash_daq as daq
import dash_bootstrap_components as dbc

from datetime import datetime
from scipy.special import comb
from dash import html, Input, Output, State, ALL, MATCH, callback, ctx, dcc, no_update


# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from helpers.button_styles import (
    COLORS,
    verified_button_style,
    unverified_toggle_button,
)


dash.register_page(__name__, path="/pages/portfolio-builder")


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
        # Add at the end of layout (outside left console/right panel)
        dbc.Toast(
            id="portfolio-builder-toast",
            header="Info",
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
                    id="portfolio-builder-console",
                    style={
                        "backgroundColor": COLORS["card"],
                        "borderRadius": "10px",
                        "padding": "20px 20px 40px 20px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "12px",
                        "height": "100%",
                        "boxSizing": "border-box",
                        "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.1)",
                        "overflow": "hidden",
                    },
                    children=[
                        # Selected Tickers and Dropdown
                        html.Div(
                            [
                                html.Label(
                                    "Add Tickers to Portfolio",
                                    style={
                                        "color": COLORS["primary"],
                                        "fontSize": "1em",
                                        "marginBottom": "4px",
                                        "fontWeight": "bold",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="dropdown-ticker-selection",
                                    placeholder="Select tickers...",
                                    multi=False,
                                    clearable=True,
                                    searchable=True,
                                    className="custom-dropdown",
                                    style={
                                        "backgroundColor": COLORS["background"],
                                        "color": COLORS["primary"],
                                        "borderRadius": "6px",
                                    },
                                ),
                                html.Br(),
                                html.Div(
                                    id="selected-ticker-card",
                                    style={
                                        "backgroundColor": COLORS["background"],
                                        "padding": "10px",
                                        "borderRadius": "6px",
                                        "minHeight": "200px",
                                        "maxHeight": "200px",
                                        "overflowY": "auto",
                                        "display": "flex",
                                        "flexWrap": "wrap",
                                        "gap": "10px",
                                        "border": f'1px solid {COLORS["primary"]}',
                                    },
                                ),
                            ]
                        ),
                        # Buttons to explore portfolio weights and performance
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "10px",
                            },
                            children=[
                                # Button for user to explore weights via MPT
                                html.Button(
                                    "Explore Efficient Frontier",
                                    id="btn-efficient-frontier",
                                    style=verified_button_style,
                                    disabled=False,
                                    className="simple",
                                ),
                            ],
                        ),
                        # Four toggle buttons to highlight important portfolio
                        html.Div(
                            style={
                                "width": "100%",
                                "maxWidth": "400px",
                                "padding": "10px",
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "10px",
                            },
                            children=[
                                html.Label(
                                    "Highlight Portfolios",
                                    style={
                                        "color": COLORS["primary"],
                                        "fontWeight": "bold",
                                        "marginBottom": "10px",
                                        "fontSize": "1rem",
                                    },
                                ),
                                # Maximum Sharpe Ratio toggle
                                html.Div(
                                    id="toggle-max-sharpe",
                                    style=unverified_toggle_button,
                                    children=[
                                        html.Span(
                                            "Maximum Sharpe:",
                                            style={
                                                "color": COLORS["text"],
                                                "fontWeight": "400",
                                                "fontSize": "0.8rem",
                                            },
                                        ),
                                        daq.ToggleSwitch(
                                            id="max-sharpe-button",
                                            value=False,
                                            size=40,
                                            style={"marginLeft": "auto"},
                                            color=COLORS["primary"],
                                            disabled=True,
                                        ),
                                    ],
                                ),
                                # Maximum Diversification Ratio toggle
                                html.Div(
                                    id="toggle-max-diversification",
                                    style=unverified_toggle_button,
                                    children=[
                                        html.Span(
                                            "Maximum Diversification:",
                                            style={
                                                "color": COLORS["text"],
                                                "fontWeight": "400",
                                                "fontSize": "0.8rem",
                                            },
                                        ),
                                        daq.ToggleSwitch(
                                            id="max-diversification-button",
                                            value=False,
                                            size=40,
                                            style={"marginLeft": "auto"},
                                            color=COLORS["primary"],
                                            disabled=True,
                                        ),
                                    ],
                                ),
                                # Minimum Variance toggle
                                html.Div(
                                    id="toggle-min-variance",
                                    style=unverified_toggle_button,
                                    children=[
                                        html.Span(
                                            "Minimum Variance:",
                                            style={
                                                "color": COLORS["text"],
                                                "fontWeight": "400",
                                                "fontSize": "0.8rem",
                                            },
                                        ),
                                        daq.ToggleSwitch(
                                            id="min-variance-button",
                                            value=False,
                                            size=40,
                                            color=COLORS["primary"],
                                            disabled=True,
                                        ),
                                    ],
                                ),
                                # Minimum Variance toggle
                                html.Div(
                                    id="toggle-equal-weights",
                                    style=unverified_toggle_button,
                                    children=[
                                        html.Span(
                                            "Equal Weights:",
                                            style={
                                                "color": COLORS["text"],
                                                "fontWeight": "400",
                                                "fontSize": "0.8rem",
                                            },
                                        ),
                                        daq.ToggleSwitch(
                                            id="equal-weights-button",
                                            value=False,
                                            size=40,
                                            color=COLORS["primary"],
                                            disabled=True,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # A stylized button for users to add stock ticker to portfolio
                        html.Button(
                            "Confirm Portfolio",
                            id="btn-confirm-portfolio",
                            n_clicks=0,
                            disabled=True,
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
                                    id="portfolio-builder-main-content",
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
                                        html.H3(
                                            "Welcome to Portfolio Builder Page!",
                                            style={
                                                "color": COLORS["primary"],
                                                "marginBottom": "1rem",
                                            },
                                        ),
                                        html.Br(),
                                        html.Div(
                                            [
                                                html.P(
                                                    "To find the appropriate portfolio, please follow the steps below:",
                                                    style={
                                                        "color": COLORS["text"],
                                                        "fontSize": "1.1rem",
                                                    },
                                                ),
                                                html.Ol(
                                                    [
                                                        html.Li(
                                                            "Choose a subset of tickers you wish to have in the portfolio.",
                                                            style={
                                                                "color": COLORS["text"]
                                                            },
                                                        ),
                                                        html.Li(
                                                            "Explore efficient frontier to find the optimum risk/return ratio.",
                                                            style={
                                                                "color": COLORS["text"]
                                                            },
                                                        ),
                                                        html.Li(
                                                            "Confirm the portfolio weights and proceed.",
                                                            style={
                                                                "color": COLORS["text"]
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
                                    ],
                                )
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
            id="summary-table-container",
            style={
                "marginTop": "40px",
                "padding": "20px",
                "backgroundColor": COLORS["card"],
                "borderRadius": "10px",
                "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.1)",
                "maxWidth": "1200px",
                "marginLeft": "auto",
                "marginRight": "auto",
                "justifyContent": "center",
                "alignItems": "center",
            },
            children=[
                html.P(
                    "Open the Efficient Frontier and choose a portfolio to view its details.",
                    style={
                        "fontSize": "25px",
                        "fontWeight": "750",
                        "color": COLORS["text"],
                        "textAlign": "center",
                        "marginBottom": "20px",
                    },
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
