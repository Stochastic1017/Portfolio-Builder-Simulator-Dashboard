
import os
import sys
import dash
import numpy as np
import dash.dash_table as dt
import plotly.graph_objects as go

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dash import html, dcc, Input, Output, State, callback
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, norm
from helpers.polygon_stock_api import StockTickerInformation
from helpers.polygon_stock_historic_plots import empty_placeholder_figure

# Register the page
dash.register_page(__name__, path="/pages/stock-dashboard-api")

# Define color constants
COLORS = {
    'primary': '#FFD700',      # Golden Yellow
    'secondary': '#FFF4B8',    # Light Yellow
    'background': '#1A1A1A',   # Dark Background
    'card': '#2D2D2D',         # Card Background
    'text': '#FFFFFF'          # White Text
}

style_defaults_for_tab = {
    'position': 'absolute',
    'top': '20px',
    'left': '0px',
    'padding': '10px 12px',
    'backgroundColor': COLORS['primary'],
    'color': COLORS['background'],
    'borderTopRightRadius': '8px',
    'borderBottomRightRadius': '8px',
    'fontWeight': 'bold',
    'cursor': 'pointer',
    'zIndex': '999',  # lower than sidebar
    'transition': 'opacity 0.3s ease',
}

layout = html.Div(
    
    style={
        'background': COLORS['background'],
        'minHeight': '100vh',
        'padding': '20px',
        'color': COLORS['text'],
        'fontFamily': '"Inter", system-ui, -apple-system, sans-serif'
    },

    children=[

        ################
        ### Page Header
        ################

        html.Div(
            style={
                'marginBottom': '30px',
                'textAlign': 'center',
                'borderBottom': f'2px solid {COLORS["primary"]}',
                'paddingBottom': '20px'
            },

            children=[

                # Header Title
                html.H1(
                    "Stock Exploration",
                    style={
                        'color': COLORS['primary'],
                        'fontSize': '2.5em',
                        'marginBottom': '10px'
                    }
                )
            ]
        ),

        ###################
        ### Grid + Content
        ###################

        html.Div(
            style={
                'display': 'grid',
                'gridTemplateColumns': '320px 1fr',
                'gap': '20px',
                'height': '100vh',
                'padding': '20px',
                'boxSizing': 'border-box',
            },
            
            children=[

            # left console
            html.Div(
                style={
                    'backgroundColor': COLORS['card'],
                    'borderRadius': '10px',  # rounded edges
                    'padding': '20px',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'gap': '15px',
                    'height': '100%',
                    'boxSizing': 'border-box',
                    'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)',  # subtle shadow
                    'overflow': 'hidden', # Prevent layout overflow
                },

                children=[

                    # Takes user input for stock ticker
                    dcc.Input(
                        id="inp-ticker",
                        type="text",
                        placeholder="Enter stock ticker...",
                        style={
                            'width': '92%',
                            'padding': '10px',
                            'backgroundColor': COLORS['background'],
                            'border': f'1px solid {COLORS["primary"]}',
                            'borderRadius': '5px',
                            'color': COLORS['text'],
                            'fontSize': '1em'
                        }
                    ),

                    # Button for user to check metadata of stock ticker
                    html.Button("Check Metadata", 
                        id="btn-metadata", 
                        n_clicks=0, 
                        style={
                            'padding': '10px',
                            'backgroundColor': COLORS['text'],
                            'border': 'none',
                            'borderRadius': '5px',
                            'color': COLORS['background'],
                            'fontWeight': 'bold',
                            'cursor': 'pointer'
                        }
                    ),

                    # Button for user to check latest news on stock ticker
                    html.Button("Check Latest News", 
                        id="btn-news", 
                        n_clicks=0, 
                        style={
                            'padding': '10px',
                            'backgroundColor': COLORS['text'],
                            'border': 'none',
                            'borderRadius': '5px',
                            'color': COLORS['background'],
                            'fontWeight': 'bold',
                            'cursor': 'pointer'
                        }
                    ),

                    # Button for user to check historic performance of stock ticker
                    html.Button("Check Historic Performance", 
                        id="btn-performance", 
                        n_clicks=0,
                        style={
                            'padding': '10px',
                            'backgroundColor': COLORS['text'],
                            'border': 'none',
                            'borderRadius': '5px',
                            'color': COLORS['background'],
                            'fontWeight': 'bold',
                            'cursor': 'pointer'
                        }
                    ),

                    # A stylized button for users to add stock ticker to portfolio
                    html.Button("Add to Portfolio", 
                        id="btn-add", 
                        n_clicks=0, 
                        style={
                            'padding': '12px',
                            'backgroundColor': COLORS['primary'],
                            'border': 'none',
                            'borderRadius': '8px',
                            'color': '#000000',
                            'fontWeight': 'bold',
                            'fontSize': '1em',
                            'cursor': 'pointer',
                            'marginTop': '10px'
                        }
                    ),
                
                    # Scrollable portfolio table
                    html.Div(
                        style={
                            'flexGrow': 1,
                            'overflowY': 'auto',
                            'minHeight': 0,
                            'marginTop': '10px',
                            'marginBottom': '10px',
                        },
                        children=[
                            dt.DataTable(
                                id='portfolio-table',
                                columns=[
                                    {'id': 'ticker', 'name': 'fullname'}
                                ],
                                data=[],
                                style_table={'overflowY': 'auto', 'maxHeight': '100%'},
                                style_cell={'textAlign': 'left'},
                            )
                        ]
                    ),
                ]
            ),
                
            # Right main content
            html.Div(
                id="output-section",
                style={
                    'backgroundColor': COLORS['card'],
                    'borderRadius': '10px',  # rounded edges
                    'height': '100%',
                    'width': '100%',
                    'boxSizing': 'border-box',
                    'overflow': 'hidden' # Prevent layout overflow
                },
                children=[

                    # Placeholder empty graph
                    dcc.Graph(
                        id="main-output-graph",
                        style={
                            'height': '100%',
                            'width': '100%'
                        },
                        config={
                            'responsive': True
                        },
                        figure=empty_placeholder_figure())
                    ]   
                ),

            ]
        ),

        # Go to Portfolio Analytics button
        html.Div(
            style={
        'display': 'flex',
        'justifyContent': 'flex-end',
        'marginTop': '20px',
        'marginLeft': '320px',  # matches width of left console
        'maxWidth': 'calc(100% - 320px)',  # align with right section
        'paddingRight': '20px'
    },
            children=[
                html.Button(
                    "Go to Portfolio Analytics ➡️",
                    id="btn-portfolio-analytics",
                    n_clicks=0,
                    style={
                        'padding': '12px 15px',
                        'backgroundColor': COLORS['primary'],
                        'border': 'none',
                        'borderRadius': '8px',
                        'color': '#000000',
                        'fontWeight': 'bold',
                        'fontSize': '1.1em',
                        'cursor': 'pointer',
                        'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)',
                    }
                )
            ]
        ),

        ################
        ### Page Footer
        ################

        html.Footer(
            style={
                'backgroundColor': COLORS['background'],
                'padding': '8px 12px',
                'textAlign': 'center',
                'fontsize': '0.8em',
                'borderRadius': '8px',
                'marginTop': '12px',
                'lineheight': '1.2'
            },

            children=[
                html.P("Developed by Shrivats Sudhir | Contact: shrivats.sudhir@gmail.com"),
                html.P(["GitHub Repository: ",
                html.A("Portfolio Optimization and Visualization Dashboard",
                    href="https://github.com/Stochastic1017/Portfolio-Analysis-Dashboard",
                    target="_blank",
                    style={'color': COLORS['primary'], 'textDecoration': 'none'}),
                ]),
            ]
        )
    ]
)
