
import os
import re
import sys
import uuid
import dash
import pandas as pd
import dash.dash_table as dt
import dash_bootstrap_components as dbc

from dash import (html, Input, Output, State, ALL, MATCH, callback, ctx, dcc, no_update)

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.portfolio_builder.markowitz_portfolio_theory import (
    plot_efficient_frontier, 
    summary_table
)

from helpers.styles.button_styles import (
    COLORS, 
    verified_button_portfolio, unverified_button_portfolio,
    verified_button_style, unverified_button_style, 
    default_style_time_range, active_style_time_range
)

dash.register_page(__name__, path="/pages/portfolio-builder")

# Stock ticker validation procedure
def validate_budget(budget):
    
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
                    'borderRadius': '10px',
                    'padding': '20px',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'gap': '15px',
                    'height': '100%',
                    'boxSizing': 'border-box',
                    'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)',  
                    'overflow': 'hidden', 
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

                    # Selected Tickers and Dropdown
                    html.Div([
                        html.Label("Add Tickers to Portfolio", style={
                            'color': COLORS['primary'],
                            'fontSize': '1em',
                            'marginBottom': '4px',
                            'fontWeight': 'bold'
                        }),

                        dcc.Dropdown(
                            id="dropdown-ticker-selection",
                            placeholder="Select tickers...",
                            multi=False,
                            clearable=True,
                            searchable=True,
                            className="custom-dropdown",
                            style={
                                'backgroundColor': COLORS['background'],
                                'color': COLORS['primary'],
                                'borderRadius': '6px',
                            },
                        ),

                        html.Br(),

                        html.Div(
                            id="selected-ticker-card",
                            style={
                                'backgroundColor': COLORS['background'],
                                'padding': '10px',
                                'borderRadius': '6px',
                                'minHeight': '200px',
                                'maxHeight': '200px',
                                'overflowY': 'auto',
                                'display': 'flex',
                                'flexWrap': 'wrap',
                                'gap': '10px',
                                'border': f'1px solid {COLORS["primary"]}'
                            }
                        )
                    ]),

                    # Buttons to explore portfolio weights and performance


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
    [
        Input("btn-verify-budget", "n_clicks")
    ],
    [
        State("inp-budget", "value")
    ],
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

@callback(
    [
        Output("portfolio-builder-main-content", "children"),
        Output("inp-budget", "valid"),
        Output("inp-budget", "invalid"),
        Output("inp-budget", "key")
    ],
    [
        Input("verify-budget", "data")
    ],
    [
        State("portfolio-store", "data")
    ],
    prevent_initial_call=True
)
def set_budget_validation(verify_budget, cache_data):
    # Defensive fallback key
    dynamic_key = f"key-{uuid.uuid4()}"

    if not verify_budget:
        return no_update, False, True, dynamic_key

    is_verified = verify_budget.get("verified", False)

    if not is_verified:
        return no_update, False, True, dynamic_key

    else:
        return (
            no_update,
            True,
            False,
            dynamic_key
        )

@callback(
    Output("dropdown-ticker-selection", "options"),
    Output("dropdown-ticker-selection", "value"),
    Input("portfolio-store", "data"),
)
def populate_dropdown_options(data):

    options = [{"label": entry["ticker"], "value": entry["ticker"]} for entry in data]
    default_selected = [entry["ticker"] for entry in data]
    
    return options, default_selected

# Track selected tickers
@callback(
    Output("dropdown-ticker-selection", "options", allow_duplicate=True),
    Output("dropdown-ticker-selection", "value", allow_duplicate=True),
    Output("selected-ticker-card", "children", allow_duplicate=True),
    Input("dropdown-ticker-selection", "value"),
    Input({"type": "remove-ticker", "index": ALL}, "n_clicks"),
    State("portfolio-store", "data"),
    State("dropdown-ticker-selection", "options"),
    State("selected-ticker-card", "children"),
    prevent_initial_call=True
)
def update_ticker_selection(dropdown_value, remove_clicks, portfolio_data, current_options, current_children):
    
    trigger = ctx.triggered_id

    full_tickers = [entry["ticker"] for entry in portfolio_data]

    # Reconstruct current selected list from tag buttons
    current_selected = [child["props"]["id"]["index"] for child in current_children] if current_children else full_tickers

    # 1. Remove ticker if "x" clicked
    if isinstance(trigger, dict) and trigger.get("type") == "remove-ticker":
        ticker_to_remove = trigger["index"]
        current_selected = [t for t in current_selected if t != ticker_to_remove]
        dropdown_value = None  # reset dropdown

    # 2. Add from dropdown
    elif isinstance(trigger, str) and dropdown_value:
        if dropdown_value not in current_selected:
            current_selected.append(dropdown_value)
        dropdown_value = None  # clear dropdown after adding

    # Filter dropdown options
    new_options = [
        {"label": t, "value": t}
        for t in full_tickers
        if t not in current_selected
    ]

    # Build card tags
    selected_tags = [
        html.Div([
            html.Span(ticker, style={
                'marginRight': '6px',
                'fontSize': '0.8em',
                'fontWeight': '500',
                'color': COLORS['background'],
            }),
            html.Button("×", id={"type": "remove-ticker", "index": ticker}, n_clicks=0,
                style={
                    'border': 'none',
                    'background': 'transparent',
                    'color': COLORS['background'],
                    'fontSize': '0.8em',
                    'cursor': 'pointer',
                    'padding': '0',
                    'lineHeight': '1'
                })
        ],
        style={
            'padding': '4px 10px',
            'backgroundColor': COLORS['primary'],
            'borderRadius': '14px',
            'display': 'flex',
            'alignItems': 'center',
            'gap': '4px',
            'height': '28px',
            'overflow': 'hidden'
        },
        id={"type": "ticker-tag", "index": ticker})
        for ticker in current_selected
    ]

    return new_options, dropdown_value, selected_tags

"""
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

@callback(
    Output("portfolio-builder-main-content", "children", allow_duplicate=True),
    Input("btn-portfolio-choose", "n_clicks"),  # or any trigger to load graph
    State("portfolio-store", "data"),
    prevent_initial_call=True
)
def show_frontier_with_summary(n_clicks, cache_data):

    efficient_frontier, min_max_returns, min_max_stddevs = plot_efficient_frontier(cache_data, COLORS),

    return efficient_frontier


@callback(
    Output("summary-table-container", "children"),
    Input("efficient-frontier-graph", "clickData"),
    State("budget-value", "data"),
    State("portfolio-store", "data"),
    prevent_initial_call=True
)
def update_summary_table(click_data, budget, cache_data):
    return summary_table(cache_data, COLORS, weights=click_data["points"][0].get("customdata"), budget=budget)
"""