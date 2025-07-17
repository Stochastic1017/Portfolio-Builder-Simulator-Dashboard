
import os
import re
import sys
import uuid
import dash
import dash_daq as daq
import dash_bootstrap_components as dbc

from scipy.special import comb
from dash import (html, Input, Output, State, ALL, MATCH, callback, ctx, dcc, no_update)

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.portfolio_builder import (
    portfolio_optimization,
    plot_efficient_frontier, 
    summary_table
)

from helpers.button_styles import (
    COLORS, 
    verified_button_portfolio, unverified_button_portfolio,
    verified_button_style, unverified_button_style,
    verified_toggle_button, unverified_toggle_button, 
    default_style_time_range, active_style_time_range
)

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

        #################
        ### Pop-up Toast 
        #################

        # Add at the end of layout (outside left console/right panel)
        dbc.Toast(
            id="portfolio-toast",
            header="Success",
            icon="success",
            is_open=False,
            duration=4000,
            dismissable=True,
        ),

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
                    html.Div(
                        style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'gap': '10px'  
                        },
                        
                        children=[

                            # Button for user to explore weights via MPT
                            html.Button("Explore Efficient Frontier", 
                                id="btn-efficient-frontier", 
                                style=verified_button_style,
                                disabled=False,
                                className="simple" 
                            ),

                        ]
                    ),

                            # Toggle switch for Max Sharpe / Min Variance

                    # Four toggle buttons to highlight important portfolio
                    html.Div(
                        style={
                            'width': '100%',         
                            'maxWidth': '400px',     
                            'padding': '10px',      
                            'display': 'flex',
                            'flexDirection': 'column',
                            'gap': '10px'            
                        },

                        children=[
                            html.Label("Highlight Portfolios", 
                                       style={
                                        'color': COLORS['primary'],
                                        'fontWeight': 'bold',
                                        'marginBottom': '10px',
                                        'fontSize': '1rem'
                                    }
                            ),

                            # Maximum Sharpe Ratio toggle
                            html.Div(
                                id="toggle-max-sharpe",
                                style=unverified_toggle_button,
                                
                                children=[
                                    html.Span("Maximum Sharpe:", style={
                                        'color': COLORS['text'],
                                        'fontWeight': '400',
                                        'fontSize': '0.8rem'
                                    }),
                                    daq.ToggleSwitch(
                                        id="max-sharpe-button",
                                        value=False,
                                        size=40,
                                        style={'marginLeft': 'auto'},
                                        color=COLORS['primary'],
                                        disabled=True
                                    )
                                ]
                            ),

                            # Maximum Diversification Ratio toggle
                            html.Div(
                                id="toggle-max-diversification",
                                style=unverified_toggle_button,
                                
                                children=[
                                    html.Span("Maximum Diversification:", style={
                                        'color': COLORS['text'],
                                        'fontWeight': '400',
                                        'fontSize': '0.8rem'
                                    }),
                                    daq.ToggleSwitch(
                                        id="max-diversification-button",
                                        value=False,
                                        size=40,
                                        style={'marginLeft': 'auto'},
                                        color=COLORS['primary'],
                                        disabled=True
                                    )
                                ]
                            ),

                            # Minimum Variance toggle
                            html.Div(
                                id="toggle-min-variance",
                                style=unverified_toggle_button,
                                
                                children=[
                                    html.Span("Minimum Variance:", style={
                                        'color': COLORS['text'],
                                        'fontWeight': '400',
                                        'fontSize': '0.8rem'
                                    }),
                                    daq.ToggleSwitch(
                                        id="min-variance-button",
                                        value=False,
                                        size=40,
                                        color=COLORS['primary'],
                                        disabled=True,
                                    )
                                ]
                            ),

                            # Minimum Variance toggle
                            html.Div(
                                id="toggle-equal-weights",
                                style=unverified_toggle_button,
                                
                                children=[
                                    html.Span("Equal Weights:", style={
                                        'color': COLORS['text'],
                                        'fontWeight': '400',
                                        'fontSize': '0.8rem'
                                    }),
                                    daq.ToggleSwitch(
                                        id="equal-weights-button",
                                        value=False,
                                        size=40,
                                        color=COLORS['primary'],
                                        disabled=True,
                                    )
                                ]
                            ),
                        ]
                    ),
                
                    # A stylized button for users to add stock ticker to portfolio
                    html.Button("Confirm Portfolio", 
                        id="btn-confirm-portfolio", 
                        n_clicks=0,
                        disabled=True,
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
                    html.H3("Welcome to Portfolio Builder Page!", style={'color': COLORS['primary'], 'marginBottom': '1rem'}),
                    html.Br(),
                    html.Div([
                        html.P("To find the appropriate portfolio, please follow the steps below:",
                            style={'color': COLORS['text'], 'fontSize': '1.1rem'}),
                        html.Ol([
                            html.Li("Input a budget (in $) and click 'Verify Budget'.", style={'color': COLORS['text']}),
                            html.Li("Choose a subset of tickers you wish to have in the portfolio.", style={'color': COLORS['text']}),
                            html.Li("Explore efficient frontier to find the optimum risk/return ratio.", style={'color': COLORS['text']}),
                            html.Li("Confirm the portfolio weights and proceed.", style={'color': COLORS['text']}),
                        ], style={
                            'textAlign': 'left',
                            'color': COLORS['text'],
                            'maxWidth': '600px',
                            'margin': '1rem auto'
                        })
                    ], style={'maxWidth': '700px'})
                ]
            )]
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
                        "marginBottom": "20px"
                    }
                ),
            ],
        ),
    
        # Go to portfolio simulator
        html.Div(
            style={
                'display': 'flex',
                'justifyContent': 'flex-end',
                'marginTop': '20px',
                'marginLeft': '320px',
                'maxWidth': 'calc(100% - 320px)',
                'paddingRight': '20px',
                'overflow': 'hidden',
            },
            
            children=[

                html.Div(
                    className='button-container', 
                    children=[
                        # Go to portfolio simulator page
                        dcc.Link(
                            html.Button(
                                "Go to Portfolio Simulator",
                                id="btn-portfolio-simulator",
                                n_clicks=0,
                                disabled=True,
                            ),
                            href="/pages/portfolio-simulator",
                            refresh=False  # Set to True if you want full page reload
                        )
                    ]
                )
            ]
        ),
    ]
)

# Populate dropdown with tickers
@callback(
    [
        Output("dropdown-ticker-selection", "options"),
        Output("dropdown-ticker-selection", "value")
    ],
    [
        Input("portfolio-store", "data")
    ],
    prevent_initial_call=True
)
def populate_dropdown_options(data):

    options = [{"label": entry["ticker"], "value": entry["ticker"]} for entry in data]
    default_selected = [entry["ticker"] for entry in data]
    
    return options, default_selected

# Track selected tickers
@callback(
    [
        Output("dropdown-ticker-selection", "options", allow_duplicate=True),
        Output("dropdown-ticker-selection", "value", allow_duplicate=True),
        Output("selected-ticker-card", "children", allow_duplicate=True),
        Output("selected-tickers-store", "data", allow_duplicate=True),
        Output("portfolio-builder-main-content", "children", allow_duplicate=True),
        Output("max-sharpe-button", "disabled", allow_duplicate=True),
        Output("toggle-max-sharpe", "style", allow_duplicate=True),
        Output("max-diversification-button", "disabled", allow_duplicate=True),
        Output("toggle-max-diversification", "style", allow_duplicate=True),
        Output("min-variance-button", "disabled", allow_duplicate=True),
        Output("toggle-min-variance", "style", allow_duplicate=True),
        Output("equal-weights-button", "disabled", allow_duplicate=True),
        Output("toggle-equal-weights", "style", allow_duplicate=True),
        Output("summary-table-container", "children", allow_duplicate=True),
    ],
    [
        Input({"type": "remove-ticker", "index": ALL}, "n_clicks"),
        Input("dropdown-ticker-selection", "value")
    ],
    [
        State("portfolio-store", "data"),
        State("dropdown-ticker-selection", "options"),
        State("selected-ticker-card", "children")
    ],
    prevent_initial_call=True
)
def update_ticker_selection(_, dropdown_value, portfolio_data, current_options, current_children):
    
    trigger = ctx.triggered_id

    full_tickers = [entry["ticker"] for entry in portfolio_data]

    # Reconstruct current selected list from tag buttons
    current_selected = [child["props"]["id"]["index"] for child in current_children] if current_children else full_tickers

    # 1. Remove ticker if "x" clicked
    if isinstance(trigger, dict) and trigger.get("type") == "remove-ticker":
        ticker_to_remove = trigger["index"]

        # Prevent removal if it would leave fewer than 2
        if len(current_selected) <= 2:
            return no_update, no_update, no_update

        current_selected = [t for t in current_selected if t != ticker_to_remove]
        dropdown_value = None

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

    return (
        new_options,
        dropdown_value,
        selected_tags,
        [{"value": t, "label": t} for t in current_selected],
        [html.Div()],
        True,  
        unverified_toggle_button,
        True,  
        unverified_toggle_button,
        True,  
        unverified_toggle_button,
        True,  
        unverified_toggle_button,
        [html.Div()]
)

# Plot efficient frontier and conduct optimizations
@callback(
    [
        Output("portfolio-weights-store", "data"),
        Output("efficient-frontier-clicked", "data"),
    ],
    Input("btn-efficient-frontier", "n_clicks"),
    [
        State("portfolio-store", "data"),
        State("selected-tickers-store", "data"),
    ],
    prevent_initial_call=True
)
def compute_optimization(_, cache_data, selected_tickers):
    if not selected_tickers:
        raise dash.exceptions.PreventUpdate

    filtered_data = [
        entry for entry in cache_data
        if entry["ticker"] in [item["value"] for item in selected_tickers]
    ]

    return portfolio_optimization(filtered_data), True

@callback(
    [
        Output("portfolio-builder-main-content", "children"),
        Output("max-sharpe-button", "disabled"),
        Output("toggle-max-sharpe", "style"),
        Output("max-diversification-button", "disabled"),
        Output("toggle-max-diversification", "style"),
        Output("min-variance-button", "disabled"),
        Output("toggle-min-variance", "style"),
        Output("equal-weights-button", "disabled"),
        Output("toggle-equal-weights", "style"),
    ],
    [
        Input("portfolio-weights-store", "data"),
        Input("max-sharpe-button", "value"),
        Input("max-diversification-button", "value"),
        Input("min-variance-button", "value"),
        Input("equal-weights-button", "value"),
    ],
    State("efficient-frontier-clicked", "data"),
    prevent_initial_call=True
)
def update_plot(optimization_dict, max_sharpe_on, max_diversification_on, min_variance_on, equal_weights_on, has_clicked):

    if not has_clicked:
        return (
            no_update,
            True, unverified_toggle_button,
            True, unverified_toggle_button,
            True, unverified_toggle_button,
            True, unverified_toggle_button
        )
    
    if not optimization_dict:
        raise dash.exceptions.PreventUpdate

    return (
        html.Div(
            id="portfolio-builder-main-content",
            style={
                'display': 'flex',
                'flexDirection': 'column',
                'height': '100%',
                'width': '100%',
                'overflow': 'hidden',
            },
            children=[
                plot_efficient_frontier(
                    max_sharpe_on,
                    min_variance_on,
                    max_diversification_on,
                    equal_weights_on,
                    optimization_dict,
                    COLORS
                )
            ]
        ),
        False, verified_toggle_button,
        False, verified_toggle_button,
        False, verified_toggle_button,
        False, verified_toggle_button
    )

# Summary table for chosen portfolio
@callback(
    [
        Output("summary-table-container", "children"),
        Output("btn-confirm-portfolio", "disabled"),
        Output("btn-confirm-portfolio", "style"),
        Output("btn-confirm-portfolio", "className"),
        Output("confirmed-weights-store", "data")
    ],
    [
        Input("efficient-frontier-graph", "clickData")
    ],
    [
        State("portfolio-store", "data"),
        State("budget-value", "data"),
        State("selected-tickers-store", "data")
    ],
    prevent_initial_call=True
)
def display_summary_table(clickData, cache_data, budget, selected_tickers):
    
    if not clickData:
        return no_update

    point = clickData["points"][0]
    weights = point["customdata"]
    risk = point["x"]
    ret = point["y"]

    # Format risk/return info
    header_text = f"Portfolio Details for Risk: {risk:.2%} | Return: {ret:.2%}"

    # Filter portfolio data using selected tickers
    filtered_data = [
        entry for entry in cache_data
        if entry["ticker"] in [item["value"] for item in selected_tickers]
    ]

    return (
        [
            html.H4(
                header_text,
                style={
                    "color": COLORS["primary"],
                    "textAlign": "center",
                    "marginBottom": "20px"
                }
            ),
            summary_table(filtered_data, COLORS, weights=weights, budget=budget)
        ],
        False,
        verified_button_portfolio,
        "special",
        weights
    )

# Confirm portfolio and display appropriate message
@callback(
    [
        Output("portfolio-toast", "is_open"),
        Output("portfolio-toast", "children"),
        Output("portfolio-toast", "style"),
    ],
    Input("btn-confirm-portfolio", "n_clicks"),
    prevent_initial_call=True
)
def confirm_portfolio(_):
    return (
        True,  # is_open
        "Portfolio confirmed! You're ready to simulate.",  # children/message
        {
            "position": "fixed",
            "top": "70px",
            "right": "30px",
            "zIndex": 9999,
            "backgroundColor": COLORS["card"],
            "color": COLORS["text"],
            "borderLeft": f"6px solid {COLORS['primary']}",  # success = yellow
            "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
            "padding": "12px 16px",
            "borderRadius": "6px",
            "width": "350px"
        }
    )

# Allow users to navigate to portfolio simulator page
@callback(
    [
        Output("btn-portfolio-simulator", "disabled"),
        Output("btn-portfolio-simulator", "style"),
        Output("btn-portfolio-simulator", "className"),
    ],
    Input("btn-confirm-portfolio", "n_clicks"),
    prevent_initial_call=True
)
def update_portfolio_analytics_button(n_clicks):
    
    if n_clicks:
        return False, verified_button_portfolio, "special"
    
    return True, unverified_button_portfolio, ""
