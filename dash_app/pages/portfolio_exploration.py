import os
import sys
import dash
import dash.dash_table as dt
import dash_bootstrap_components as dbc

from dash import html, dcc

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.button_styles import COLORS

# Register the page
dash.register_page(__name__, path="/pages/portfolio-exploration")

layout = html.Div(
    style={
        "background": COLORS["background"],
        "minHeight": "100vh",
        "padding": "20px",
        "boxSizing": "border-box",
        "color": COLORS["text"],
        "fontFamily": '"Inter", system-ui, -apple-system, sans-serif',
    },
    children=[
        #################
        ### Pop-up Toast
        #################
        # Toast messages to inform users of success/failures
        dbc.Toast(
            id="portfolio-exploration-toast",
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
                    id="stock-exploration-console",
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
                        # Stock ticker input + verify ticker button
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "alignItems": "center",
                            },
                            children=[
                                # Input for stock ticker
                                dbc.Input(
                                    id="inp-ticker",
                                    type="text",
                                    debounce=True,
                                    valid=False,
                                    invalid=False,
                                    key="input-key",
                                    placeholder="Enter Stock Ticker",
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
                                # A stylized button to verify if user input ticker is correct
                                html.Button(
                                    "Verify Ticker",
                                    id="btn-verify-ticker",
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
                        # Two buttons to explore stock ticker
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "10px",
                            },
                            children=[
                                # Button for user to check latest news on stock ticker
                                html.Button(
                                    "Check Latest News",
                                    id="btn-news",
                                    disabled=True,
                                    style={"fontWeight": "bold"},
                                ),
                                # Button for user to check historic performance of stock ticker
                                html.Button(
                                    "Check Historic Performance",
                                    id="btn-performance",
                                    disabled=True,
                                    style={"fontWeight": "bold"},
                                ),
                            ],
                        ),
                        # A stylized button for users to add stock ticker to portfolio
                        html.Button(
                            "Add to Portfolio",
                            id="btn-add",
                            n_clicks=0,
                            disabled=True,
                        ),
                        # Scrollable portfolio table
                        html.Div(
                            style={
                                "flex": "1 1 auto",
                                "display": "flex",
                                "minHeight": 0,
                                "marginTop": "10px",
                            },
                            children=[
                                dt.DataTable(
                                    id="portfolio-table",
                                    columns=[
                                        {"id": "ticker", "name": "Ticker"},
                                        {"id": "fullname", "name": "Company Name"},
                                    ],
                                    style_table={
                                        "overflowY": "auto",
                                        "overflowX": "auto",
                                        "height": "100%",
                                        "border": "1px solid #ccc",
                                        "borderRadius": "10px",
                                        "marginBottom": "20px",
                                    },
                                    style_cell={
                                        "padding": "10px",
                                        "textAlign": "left",
                                        "backgroundColor": "#f9f9f9",
                                        "color": "#333",
                                        "fontFamily": "Arial",
                                        "minWidth": "100px",
                                        "maxWidth": "250px",
                                        "whiteSpace": "normal",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                    },
                                    style_data_conditional=[
                                        {
                                            "if": {"state": "selected"},
                                            "backgroundColor": "inherit !important",
                                            "border": "inherit !important",
                                        }
                                    ],
                                    style_header={
                                        "backgroundColor": "#0E1117",
                                        "color": "white",
                                        "fontWeight": "bold",
                                        "border": "1px solid #ccc",
                                    },
                                    editable=False,  # Disable editing
                                    row_deletable=True,  # Enable row deletion
                                    sort_action="none",  # Disable sorting
                                    filter_action="none",  # Disable filtering
                                    page_action="none",  # Disable pagination
                                    style_as_list_view=True,  # Remove any default interactivity styling
                                    selected_rows=[],  # Prevent row selection
                                    active_cell=None,  # Prevent active cell highlighting
                                    row_selectable=None,  # Disable row selection
                                    data=[],
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
                                    id="portfolio-exploration-main-content",
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
