
import os
import sys
import dash
import pandas as pd
import dash.dash_table as dt
import dash_bootstrap_components as dbc

from io import StringIO
from dash import (html, dcc, Input, Output, State, callback, ctx, no_update)

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.markowitz_portfolio_theory import plot_efficient_frontier, summary_table
from helpers.button_styles import (COLORS, 
                                   verified_button_portfolio, unverified_button_portfolio,
                                   verified_button_style, unverified_button_style, 
                                   default_style_time_range, active_style_time_range)

dash.register_page(__name__, path="/pages/portfolio-builder")

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
                    "Portfolio Builder",
                    style={
                        'color': COLORS['primary'],
                        'fontSize': '2.5em',
                        'marginBottom': '10px'
                    }
                )
            ]
        ),

        #####################
        ### Left console
        ### Right content
        #####################

        # Left console + Right content
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

            # Left console
            html.Div(
                id="portfolio-builder-console",
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
                    
                    # Budget (in $) input
                    html.Div(
                        style={
                            'display': 'flex',
                            'flexDirection': 'row',
                            'alignItems': 'center'
                        },
                        
                        children=[

                            html.Div(
                                style={
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                    'gap': '5px',
                                    'marginTop': '10px'
                                },
                                
                                children=[
                                    html.Label("Enter Budget ($)", style={
                                        'color': COLORS['primary'],
                                        'fontWeight': '600',
                                        'fontSize': '0.9rem',
                                    }),
                                    dcc.Input(
                                        id="budget-input",
                                        type="number",
                                        min=1000,           # Optional: minimum value ($1k)
                                        max=100_000_000,    # Cap at $100 million
                                        step=1000,          # Increment in thousands
                                        placeholder="e.g., 50000",
                                        debounce=True,
                                        style={
                                            'width': '200px',
                                            'padding': '10px',
                                            'backgroundColor': COLORS['background'],
                                            'border': f'1px solid {COLORS["primary"]}',
                                            'borderRadius': '6px',
                                            'color': COLORS['text'],
                                            'fontSize': '1em',
                                            'textAlign': 'right',
                                            'boxShadow': '0 1px 3px rgba(0,0,0,0.2)'
                                        }
                                    ),
                                    html.Div(id='budget-error-message', style={
                                        'color': 'red',
                                        'fontSize': '0.85rem',
                                        'marginTop': '4px',
                                        'display': 'none'
                                    })
                                ]
                            ),

                        ]
                    ),

                    # Two buttons to explore stock ticker
                    html.Div(
                        style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'gap': '10px'  
                        },
                        
                        children=[
                            
                            # Button for user to check latest news on stock ticker
                            html.Button("Get Portfolio Summary", 
                                id="portfolio-summary", 
                                disabled=False, 
                            ),                    

                            # Button for user to check historic performance of stock ticker
                            html.Button("Get Efficient Frontier", 
                                id="efficient-frontier", 
                                disabled=False,
                            ),
                        
                        ],                    
                    ),

                ]
            ),
                
            # Right content
            html.Div(
                id="portfolio-builder-main-content",
                style={
                    'backgroundColor': COLORS['background'],
                    'borderRadius': '10px',
                    'height': '100%',
                    'width': '100%',
                    'boxSizing': 'border-box',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'justifyContent': "center",
                    'alignItems': 'center',
                    'textAlign': 'center',
                    'padding': '2rem',
                    'overflow': 'hidden'
                },
                
                children=[
                    html.Div([
                        html.H3("Welcome to Portfolio Builder!", style={'color': COLORS['primary'], 'marginBottom': '1rem'}),

                        html.Br(),

                        html.Div([

                        html.P("Given list of stock tickers chosen in Stock Exploration page:", 
                               style={'color': COLORS['text'], 
                                      'fontSize': '1.1rem'}),

                        html.Ol([
                            html.Li("Input Budget (in USD).", 
                                    style={'color': COLORS['text']}),
                            html.Li("Get a summary table of portfolio.", 
                                    style={'color': COLORS['text']}),
                            html.Li("Get efficient frontier to find build portfolio according to specified risk/return ratios.", 
                                    style={'color': COLORS['text']}),
                            ], 
                            style={'textAlign': 'left', 
                                   'color': COLORS['text'], 
                                   'maxWidth': '600px', 
                                   'margin': 
                                   '1rem auto'}),
                            ], style={'maxWidth': '700px'})
                        ])
                    ]
                )

            ]
        ),

    ]

)

@callback(
    Output("budget-error-message", "children"),
    Output("budget-error-message", "style"),
    Input("budget-input", "value")
)
def validate_budget(budget):
    if budget is None:
        return "", {'display': 'none'}

    if not isinstance(budget, (int, float)):
        return "Please enter a numeric value.", {'display': 'block'}
    
    if budget <= 0:
        return "Budget must be greater than $0.", {'display': 'block'}

    if budget > 100_000_000:
        return "Budget cannot exceed $100 million.", {'display': 'block'}

    return "", {'display': 'none'}

@callback(
    Output("portfolio-builder-main-content", "children"),
    [
        Input("portfolio-summary", "n_clicks"),
        Input("efficient-frontier", "n_clicks")
    ],
    State("portfolio-store", "data"),
    prevent_initial_call=True
)
def update_main_content(n_summary, n_frontier, data):
    if not data:
        return html.Div("No portfolio data found.", style={'color': 'red'})

    triggered_id = ctx.triggered_id if ctx.triggered_id else None

    if triggered_id == "portfolio-summary":
        return html.Div([
            html.H3(f"{len(data)} stock(s) loaded into portfolio.", style={'color': COLORS['primary']}),
            html.Div(summary_table(data, COLORS), 
                     id="portfolio-table-output", 
                     style={'marginTop': '20px'})
        ])
    
    elif triggered_id == "efficient-frontier":
        return html.Div(
                id="portfolio-builder-main-content",
                style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'height': '100%',
                    'width': '100%',
                    'overflow': 'hidden',
                },

                children=[
                    html.Div(
                        style={
                            "flex": "1", 
                            "overflow": "hidden"
                        },
                        children=[plot_efficient_frontier(data, COLORS)]
                    ),
                          
                ]
        )
