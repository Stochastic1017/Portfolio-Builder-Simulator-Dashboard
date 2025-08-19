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
            children="",
            icon="primary",
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
                        # Selected Tickers and Dropdown
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "10px",
                            },
                            children=[
                                html.Label(
                                    "Add Tickers to Portfolio",
                                    style={
                                        "color": COLORS["primary"],
                                        "fontSize": "1em",
                                        "marginBottom": "10px",
                                        "fontWeight": "bold",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="dropdown-ticker-selection",
                                    placeholder="Select tickers...",
                                    multi=True,
                                    clearable=True,
                                    searchable=True,
                                    className="custom-dropdown",
                                    style={
                                        "backgroundColor": COLORS["background"],
                                        "borderRadius": "6px",
                                        "height": "25vh",
                                        "overflowY": "auto",
                                    },
                                ),
                            ],
                        ),
                        # Button to plot efficient frontier
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "10px",
                            },
                            children=[
                                # Button to explore Efficient Frontier
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "gap": "10px",
                                    },
                                    children=[
                                        # Button for user to check latest news on stock ticker
                                        html.Button(
                                            "Plot Efficient Frontier",
                                            id="btn-efficient-frontier",
                                            disabled=True,
                                            style={"fontWeight": "bold"},
                                        ),
                                    ],
                                ),
                                # Label to find optimized portfolios
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
                            style={"fontWeight": "bold"},
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
