
import os
import re
import sys
import uuid
import dash
import pandas as pd
import dash.dash_table as dt
import dash_bootstrap_components as dbc

from dash import (html, Input, Output, State, callback, ctx, no_update)

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.markowitz_portfolio_theory import plot_efficient_frontier, summary_table
from helpers.button_styles import (COLORS, 
                                   verified_button_portfolio, unverified_button_portfolio,
                                   verified_button_style, unverified_button_style, 
                                   default_style_time_range, active_style_time_range)

dash.register_page(__name__, path="/pages/portfolio-builder")

# Stock ticker validation procedure
def validate_budget(budget):
    
    """
    Function to validate budget input by user. 
    """

    if budget is None or budget == "":
        return {"valid": False, "value": None, "error": "Budget cannot be empty."}

    # Remove leading/trailing whitespace
    budget = budget.strip()

    # Basic sanity check for allowed characters
    if not re.fullmatch(r"[0-9,]*\.?[0-9]*", budget):
        return {"valid": False, "value": None, "error": "Budget contains invalid characters."}

    # Check multiple decimals
    if budget.count('.') > 1:
        return {"valid": False, "value": None, "error": "Budget has multiple decimal points."}

    # Validate comma placement using regex
    if ',' in budget:
        # Regex to match correct comma placement: 1,000 or 12,345.67
        if not re.fullmatch(r"(?:\d{1,3})(?:,\d{3})*(?:\.\d{1,2})?", budget):
            return {"valid": False, "value": None, "error": "Commas are placed incorrectly."}

    # Remove commas for numeric conversion
    numeric_str = budget.replace(",", "")

    try:
    
        value = float(numeric_str)
        if value < 0:
            return {"valid": False, "value": None, "error": "Budget cannot be negative."}
        return {"valid": True, "value": value, "error": None}
    
    except ValueError:
        return {"valid": False, "value": None, "error": "Could not parse budget value."}

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
                    
                    # Budget (in $) input + verify budget button
                    html.Div(
                        style={
                            'display': 'flex',
                            'flexDirection': 'row',
                            'alignItems': 'center'
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
                                    'width': '200px',
                                    'padding': '10px',
                                    'backgroundColor': COLORS['background'],
                                    'border': f'1px solid {COLORS['primary']}',
                                    'borderRadius': '5px',
                                    'color': COLORS['text'],
                                    'fontSize': '1em',
                                    'marginRight': '10px'
                                },
                            ),

                            # A stylized button to verify if user input budget is correct
                            html.Button("Verify Budget",
                                id="btn-verify-budget",
                                n_clicks=0,
                                disabled=False,
                                className='special',
                                style={
                                    'padding': '6px 12px',
                                    'backgroundColor': COLORS['primary'],
                                    'color': 'black',
                                    'border': '1px solid #9370DB',
                                    'borderRadius': '20px',
                                    'fontWeight': 'bold',
                                    'fontSize': '0.75em',
                                    'cursor': 'pointer',
                                    'alignSelf': 'flex-start',
                                    'transition': 'all 0.2s ease-in-out'
                                }
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
                            
                            # Button for user to check historic performance of stock ticker
                            html.Button("Select Portfolio", 
                                id="btn-portfolio-choose", 
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

# Ensure budget formatting is consistent for monetary inputs
@callback(
    Output("inp-budget", "value"),
    Input("inp-budget", "value"),
    prevent_initial_call=True
)
def format_budget_input(value):
    if not value:
        return ""

    # Remove everything except digits and decimal point
    raw_value = re.sub(r"[^\d.]", "", value)

    try:
        # Format number with commas, preserve decimal if present
        if "." in raw_value:
            number = float(raw_value)
            formatted = f"{number:,.2f}"
        else:
            number = int(raw_value)
            formatted = f"{number:,}"
        return formatted
    except:
        return value  # return unformatted if there's an issue

# Verify budget and reset status if budget changes
@callback(
    [
        Output("verify-budget", "data"),
        Output("budget-value", "data"),
    ],
    Input("btn-verify-budget", "n_clicks"),
    State("inp-budget", "value"),
    prevent_initial_call=True
)
def handle_verify_budget(_, budget_input):
    trigger_id = ctx.triggered_id

    if trigger_id == "btn-verify-budget":
        result = validate_budget(budget_input)

        if result["valid"]:
            return {"verified": True}, result["value"]
        else:
            return {"verified": False, "error": result["error"]}, None

    elif trigger_id == "inp-budget":
        return {"verified": False}, None

    return no_update

# Change validation symbol upon successful verification
@callback(
    Output("inp-budget", "valid"),
    Output("inp-budget", "invalid"),
    Output("inp-budget", "key"),
    Input("verify-budget", "data"),
    prevent_initial_call=True
)
def set_budget_validation(verify_budget):
    is_verified = verify_budget.get("verified", False)

    # Change the key so Dash forces a component refresh
    dynamic_key = f"key-{uuid.uuid4()}"
    
    return is_verified, not is_verified, dynamic_key

# Toggle styles between enabled/disabled status
@callback(
    [
        Output("btn-portfolio-choose", "disabled"),
        Output("btn-portfolio-choose", "style"),
        Output("btn-portfolio-choose", "className"),
    ],
    Input("verify-budget", "data")
)
def toggle_button_states(verify_budget):
    is_verified = verify_budget.get("verified", False)

    if is_verified:
        return (
            False, verified_button_style, "simple",
        )
    
    else:
        return (
            True, unverified_button_style, "",
        )

# Update main output section depending on what is chosen by user
@callback(
    Output("portfolio-builder-main-content", "children"),
    [
        Input("btn-portfolio-choose", "n_clicks"),
    ],
    State("portfolio-store", "data"),
    prevent_initial_call=True
)
def update_main_content(_, data):

    triggered_id = ctx.triggered_id if ctx.triggered_id else None

    if triggered_id == "btn-portfolio-choose":
        return plot_efficient_frontier(data, COLORS)

"""
# Gets back weights of portfolio chosen by user on the efficient frontier
@callback(
    Output("portfolio-builder-main-content", "children", allow_duplicate=True),
    Input("efficient-frontier-graph", "clickData"),
    State("budget-value", "data"),
    State("portfolio-store", "data"),
    prevent_initial_call=True
)
def update_table_on_point_click(click_data, budget, cache_data):
    if not click_data or not budget:
        return html.Div("Click a point and enter a budget to get allocation.", style={'color': 'red'})

    weights = click_data["points"][0]["customdata"]
    return html.Div([
        html.H3("Portfolio Allocation for Selected Risk/Return Point", style={'color': COLORS['primary']}),
        summary_table(cache_data, COLORS, weights, budget)
    ])
"""