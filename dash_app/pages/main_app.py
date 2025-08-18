import os
import sys
import dash
import dash_bootstrap_components as dbc

from dash import html, dcc


# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from pages.portfolio_exploration import layout as portfolio_exploration_layout
from pages.portfolio_builder import layout as portfolio_builder_layout
from pages.portfolio_simulator import layout as portfolio_simulator_layout

from helpers.button_styles import COLORS


dash.register_page(__name__, path="/pages/main-app")

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
        dcc.Tabs(
            id="main-app-tabs",
            value="main-tab-explore",
            children=[
                dcc.Tab(
                    label="Portfolio Exploration",
                    value="main-tab-explore",
                    children=[portfolio_exploration_layout],
                    style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["text"],
                        "padding": "10px 20px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": f"2px solid transparent",
                        "transition": "color 0.3s ease",
                    },
                    selected_style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["primary"],
                        "padding": "6px 18px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": f"2px solid {COLORS['primary']}",
                    },
                ),
                dcc.Tab(
                    label="Portfolio Builder",
                    value="main-tab-builder",
                    children=[portfolio_builder_layout],
                    style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["text"],
                        "padding": "10px 20px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": f"2px solid transparent",
                        "transition": "color 0.3s ease",
                    },
                    selected_style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["primary"],
                        "padding": "6px 18px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": f"2px solid {COLORS['primary']}",
                    },
                ),
                dcc.Tab(
                    label="Portfolio Simulator",
                    value="main-tab-simulator",
                    children=[portfolio_simulator_layout],
                    style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["text"],
                        "padding": "10px 20px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": f"2px solid transparent",
                        "transition": "color 0.3s ease",
                    },
                    selected_style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["primary"],
                        "padding": "6px 18px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": f"2px solid {COLORS['primary']}",
                    },
                ),
            ],
        )
    ],
)
