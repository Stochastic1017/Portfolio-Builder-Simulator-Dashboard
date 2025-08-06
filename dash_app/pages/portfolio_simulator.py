
import os
import re
import sys
import uuid
import dash
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc

from datetime import datetime, timedelta
from dash import (html, Input, Output, State, ALL, MATCH, callback, ctx, dcc, no_update)

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.portfolio_simulator import (
    parse_ts_map,
    portfolio_dash_range_selector,
    grid_search_arima_model, simulate_arima_paths, arima_simulation_plot,
    select_best_garch_model
)

from helpers.portfolio_exploration import (
    create_historic_plots,
    create_statistics_table,
)

from helpers.button_styles import (
    COLORS,
    verified_button_style, unverified_button_style,
    default_style_time_range, active_style_time_range,
    active_labelStyle_radioitems, active_inputStyle_radioitems, active_style_radioitems
)

dash.register_page(__name__, path="/pages/portfolio-simulator")

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

# Get next business day
def next_business_day(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    next_day = date_obj + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_day += timedelta(days=1)
    return next_day.strftime("%Y-%m-%d")  # Return as string if needed

# Get exact one year business day
def max_forecast_date(latest_date_str):
    date_obj = datetime.strptime(latest_date_str, "%Y-%m-%d")
    one_year_later = date_obj + timedelta(days=365)
    return one_year_later.strftime("%Y-%m-%d")

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
            id="portfolio-simulator-toast",
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
                    "Portfolio Simulator",
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
                id="portfolio-simulator-console",
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
                
                    # Buttons to explore portfolio weights and performance
                    html.Div(
                        style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'gap': '10px'  
                        },
                        
                        children=[

                            # Button for user to explore past performance of portfolio
                            html.Button("Evaluate Past Performance", 
                                id="btn-portfolio-performance", 
                                style=verified_button_style,
                                disabled=False,
                                className="simple" 
                            ),
                        ]
                    ),

                    # Date picker, Criterion Selector, and Ensemble Generator for prediction
                    html.Div(
                        style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'gap': '10px',
                            'width': '100%',  
                        },
                        
                        children=[

                            html.Label("Choose a future date for prediction:",
                                    style={
                                        'color': COLORS['primary'],
                                        'fontWeight': 'bold',
                                        'fontSize': '1rem'
                                    }
                            ),

                            dcc.DatePickerSingle(
                                id="date-chooser-simulation",
                                disabled=True,
                                with_portal=True,
                                display_format='MMM Do, YY',
                            ),

                            html.Br(),

                            html.Label("Number of ensembles to generate:",
                                        style={
                                            'color': COLORS['primary'],
                                            'fontWeight': 'bold',
                                            'marginBottom': '10px',
                                            'fontSize': '1rem'
                                        }
                            ),

                            # Slider to choose number of ensembles to generate
                            dcc.Slider(
                                id="num-ensemble-slider",
                                min=10,
                                max=1000,
                                value=500,
                                disabled=True,
                                tooltip={"placement": "bottom", "always_visible": False}
                            ),                       

                            html.Br(),

                            html.Label("Model selection criterion:",
                                    style={
                                        'color': COLORS['primary'],
                                        'fontWeight': 'bold',
                                        'fontSize': '1rem',
                                        'marginTop': '20px'
                                    }
                            ),

                            dcc.RadioItems(
                                id='model-selection-criterion',
                                options=[
                                    {'label': 'AIC (Akaike)', 'value': 'AIC', 'disabled': True},
                                    {'label': 'BIC (Bayesian)', 'value': 'BIC', 'disabled': True},
                                    {'label': 'Log-Likelihood', 'value': 'LogLikelihood', 'disabled': True}
                                ],
                            ),
                        ]
                    ),

                    # Buttons to generate ARIMA/GARCH predictions
                    html.Div(
                        style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'gap': '10px'  
                        },
                        
                        children=[

                            # Button for user to start monte carlo exploration
                            html.Button("Predict Using ARIMA process", 
                                id="btn-arima-performance", 
                                style=verified_button_style,
                                disabled=False,
                                className="simple" 
                            ),

                            # Button for user to start monte carlo exploration
                            html.Button("Predict Using GARCH process", 
                                id="btn-garch-performance", 
                                style=verified_button_style,
                                disabled=False,
                                className="simple" 
                            ),
                        ]
                    ),
                ]
            ),
                
            # Right content
            html.Div(
                id="portfolio-simulator-main-content",
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
                        html.H3("Welcome to Portfolio Simulator Page!", style={'color': COLORS['primary'], 'marginBottom': '1rem'}),

                        html.Br(),

                        html.Div([

                        html.P("To simulate the portfolio, please follow the steps below:", 
                               style={'color': COLORS['text'], 
                                      'fontSize': '1.1rem'}),

                        html.Ol([
                            html.Li("Input a budget (in $) and click 'Verify Budget'.", 
                                    style={'color': COLORS['text']}),
                            html.Li("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.", 
                                    style={'color': COLORS['text']}),
                            html.Li("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.", 
                                    style={'color': COLORS['text']}),
                            html.Li("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.", 
                                    style={'color': COLORS['text']}),], 
                            style={'textAlign': 'left', 
                                   'color': COLORS['text'], 
                                   'maxWidth': '600px', 
                                   'margin': 
                                   '1rem auto'}),
                            ], style={'maxWidth': '700px'})
                        ])
                    ]
                ),
            
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
        return value 

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

# Show validation symbol upon successful verification
@callback(
    [
        Output("inp-budget", "valid"),
        Output("inp-budget", "invalid"),
        Output("inp-budget", "key")
    ],
    Input("verify-budget", "data"),
    prevent_initial_call=True
)
def set_budget_validation(verify_budget):
    
    # Defensive fallback key
    dynamic_key = f"key-{uuid.uuid4()}"

    is_verified = verify_budget.get("verified", False)

    if not is_verified:
        return (False, True, dynamic_key)
    
    return (True, False, dynamic_key)

# Toggle portfolio performance and date chooser after verification of budget
@callback(
    [
        Output("btn-portfolio-performance", "disabled"),
        Output("btn-portfolio-performance", "style"),
        Output("btn-portfolio-performance", "className"),

        Output("date-chooser-simulation", "disabled"),   
        Output("date-chooser-simulation", "min_date_allowed"),        
        Output("date-chooser-simulation", "max_date_allowed"), 

        Output("num-ensemble-slider", "disabled"),

        Output("model-selection-criterion", "options"),
        Output("model-selection-criterion", "labelStyle"),
        Output("model-selection-criterion", "inputStyle"),
        Output("model-selection-criterion", "style"),

    ],
    Input("verify-budget", "data"),
    State("latest-date", "data")
)
def enable_initial_controls(verify_budget, latest_date):
    is_verified = verify_budget.get("verified", False)

    if is_verified:
        return (
            False, verified_button_style, "simple",
            False, next_business_day(latest_date), max_forecast_date(latest_date),
            False,
            [
                {'label': 'AIC (Akaike)', 'value': 'AIC', 'disabled': False},
                {'label': 'BIC (Bayesian)', 'value': 'BIC', 'disabled': False},
                {'label': 'Log-Likelihood', 'value': 'LogLikelihood', 'disabled': False}
            ],
            active_labelStyle_radioitems,
            active_inputStyle_radioitems,
            active_style_radioitems
        )
    else:
        return (
            True, unverified_button_style, "",
            True, next_business_day(latest_date), max_forecast_date(latest_date),
            True,
            [
                {'label': 'AIC (Akaike)', 'value': 'AIC', 'disabled': True},
                {'label': 'BIC (Bayesian)', 'value': 'BIC', 'disabled': True},
                {'label': 'Log-Likelihood', 'value': 'LogLikelihood', 'disabled': True}
            ],
            active_labelStyle_radioitems,
            active_inputStyle_radioitems,
            active_style_radioitems
        )

@callback(
    [
        Output("btn-arima-performance", "disabled"),
        Output("btn-arima-performance", "style"),
        Output("btn-arima-performance", "className"),

        Output("btn-garch-performance", "disabled"),
        Output("btn-garch-performance", "style"),
        Output("btn-garch-performance", "className"),
        
    ],
    Input("date-chooser-simulation", "date"),
    Input("model-selection-criterion", "value")
)
def enable_post_datepicker_buttons(selected_date, selected_criterion_information):
    if selected_date and selected_criterion_information:
        return (
            False, verified_button_style, "simple",
            False, verified_button_style, "simple",
        )
    else:
        return (
            True, unverified_button_style, "",
            True, unverified_button_style, "",
        )

@callback(
    [
        Output("portfolio-simulator-main-content", "children"),
    ],
    [
        Input("btn-portfolio-performance", "n_clicks"),
        Input("btn-arima-performance", "n_clicks"),
        Input("btn-garch-performance", "n_clicks"),
        Input("model-selection-criterion", "value")
    ],
    [
        State("verify-budget", "data"),
        State("portfolio-store", "data"),
        State("selected-tickers-store", "data"),
        State("confirmed-weights-store", "data"),
        State("budget-value", "data"),
        State("portfolio-risk-return", "data"),
        State("date-chooser-simulation", "date"),
        State("num-ensemble-slider", "value"),
    ],
    prevent_initial_call=True
)
def update_portfolio_simulator_main_plot(_, __, ___, criterion_selector,
                                         verify_budget, portfolio_store, 
                                         selected_tickers, weights, budget,
                                         risk_return, forecast_until, num_ensembles):
    button_id = ctx.triggered_id

    if not verify_budget.get("verified", False):
        raise dash.exceptions.PreventUpdate
    
    _, portfolio_value_ts = parse_ts_map(
                                selected_tickers=selected_tickers,
                                portfolio_weights=weights,
                                portfolio_store=portfolio_store,
                                budget=budget
                            )

    log_returns = np.log(portfolio_value_ts / portfolio_value_ts.shift(1)).dropna() 
    
    if button_id == "btn-portfolio-performance":
        return (
            html.Div(
                id="simulator-main-panel",
                style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'width': '100%', 'overflow': 'hidden'},
                children=[
                    portfolio_dash_range_selector(default_style=default_style_time_range),
                    dcc.Tabs(
                        id="portfolio-performance-tabs",
                        value='portfolio-tab-plot',
                        style={
                            "marginTop": "10px",
                            "backgroundColor": COLORS['card'],
                            "color": COLORS['text'],
                            "height": "42px",
                            "borderRadius": "5px",
                            "overflow": "hidden",
                        },
                        colors={
                            "border": COLORS['background'],
                            "primary": COLORS['primary'],
                            "background": COLORS['card'],
                        },
                        children=[
                            dcc.Tab(
                                label="Price & Returns Plot",
                                value='portfolio-tab-plot',
                                style={...},  # keep styles
                                selected_style={...}
                            ),
                            dcc.Tab(
                                label="Statistical Summary",
                                value='portfolio-tab-stats',
                                style={...},
                                selected_style={...}
                            ),
                        ]
                    ),
                    html.Div(id="portfolio-plot-container", style={"flex": "1", "overflow": "hidden"}),
                ]
            ),
        )

    elif button_id == "btn-arima-performance":
        best_model_info = grid_search_arima_model(log_returns, criterion=criterion_selector)
        model_result = best_model_info['result']
        best_order = best_model_info['order']
        score = best_model_info['score']

        simulations, forecast_index = simulate_arima_paths(model_result=model_result, 
                                                           last_date=portfolio_value_ts.index[-1], 
                                                           forecast_until=forecast_until, 
                                                           num_ensembles=num_ensembles, 
                                                           inferred_freq='B')

        title = f"ARIMA Simulation - Criterion: {criterion_selector}, Order: {best_order}, Score: {score:.2f}, Ensembles: {num_ensembles}"

        return (
            html.Div(
                id="simulator-main-panel",
                style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'width': '100%', 'overflow': 'hidden'},
                children=[
                    html.Div(
                        id="portfolio-plot-container",
                        style={"flex": "1", "overflow": "hidden"},
                        children=[
                            arima_simulation_plot(
                                title=title,
                                dates=portfolio_value_ts.index,
                                daily_prices=portfolio_value_ts.values,
                                log_returns=log_returns,
                                simulations=simulations,
                                forecast_index=forecast_index,
                                COLORS=COLORS
                            )
                        ],
                    )
                ]
            ),
        )

    elif button_id == "btn-garch-performance":

        best_models = select_best_garch_model(
            log_returns,
            p_range=range(1, 4),
            q_range=range(1, 4),
            criterions=['AIC', 'BIC', 'LogLikelihood'],
            dists=['normal', 't', 'skewt']
        )

        return (
            html.Div(
                "TEMP!!"
            ),
        )

    return no_update

# Highlight buttons based on time-range selected
@callback(
    [
        Output("portfolio-range-1M", "style"),
        Output("portfolio-range-3M", "style"),
        Output("portfolio-range-6M", "style"),
        Output("portfolio-range-1Y", "style"),
        Output("portfolio-range-5Y", "style"),
        Output("portfolio-range-all", "style"),
        Output("portfolio-selected-range", "data")
    ],
    [
        Input("portfolio-range-1M", "n_clicks"),
        Input("portfolio-range-3M", "n_clicks"),
        Input("portfolio-range-6M", "n_clicks"),
        Input("portfolio-range-1Y", "n_clicks"),
        Input("portfolio-range-5Y", "n_clicks"),
        Input("portfolio-range-all", "n_clicks"),
    ]
)
def update_portfolio_range_styles(*btn_clicks):
    button_ids = [
        "portfolio-range-1M", "portfolio-range-3M", "portfolio-range-6M",
        "portfolio-range-1Y", "portfolio-range-5Y", "portfolio-range-all"
    ]

    # If no button has been clicked yet, fall back to "All"
    if not any(click and click > 0 for click in btn_clicks):
        selected = "portfolio-range-all"
    else:
        selected = ctx.triggered_id or "portfolio-range-all"

    style_map = {btn_id: default_style_time_range for btn_id in button_ids}
    style_map[selected] = active_style_time_range

    return (
        style_map["portfolio-range-1M"],
        style_map["portfolio-range-3M"],
        style_map["portfolio-range-6M"],
        style_map["portfolio-range-1Y"],
        style_map["portfolio-range-5Y"],
        style_map["portfolio-range-all"],
        selected.split("-")[-1]  # "1M", "3M", ..., "all"
    )

# Update historic daily plot based on range selected
@callback(
    [
        Output("portfolio-plot-container", "children"),
    ],
    [
        Input("portfolio-performance-tabs", "value"),
        Input("portfolio-selected-range", "data")
    ],
    [
        State("budget-value", "data"),
        State("portfolio-store", "data"),
        State("selected-tickers-store", "data"),
        State("confirmed-weights-store", "data"),
        State("portfolio-risk-return", "data"),
    ],
    prevent_initial_call=True
)
def update_plot_on_range_change(active_tab, selected_range, budget, portfolio_store, selected_tickers, weights, risk_return):

    # Compute budget-adjusted time series
    _, portfolio_value_ts = parse_ts_map(
        selected_tickers=selected_tickers,
        portfolio_weights=weights,
        portfolio_store=portfolio_store,
        budget=budget
    )
    
    # Portfolio returns
    portfolio_returns = portfolio_value_ts.pct_change().dropna()

    today = portfolio_value_ts.index[-1]
    range_days = {
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 365,
        "5Y": 1825,
        "all": None
    }

    selected_range = (selected_range or "all").upper()
    if selected_range not in range_days:
        selected_range = "all"

    if range_days[selected_range] is not None:
        cutoff = today - timedelta(days=range_days[selected_range])
        portfolio_value_ts = portfolio_value_ts[portfolio_value_ts.index >= cutoff]
        portfolio_returns = portfolio_value_ts.pct_change().dropna()

    if active_tab == "portfolio-tab-plot":
        return (create_historic_plots(
            full_name=f"Portfolio (Risk: {round(risk_return['risk'], 4)*100:.2f}%, Return: {round(risk_return['return'], 4)*100:.2f}%)",
            dates=portfolio_value_ts.index,
            daily_prices=portfolio_value_ts.values,
            daily_returns=portfolio_returns,
            COLORS=COLORS),
        )
    
    elif active_tab == "portfolio-tab-stats":
        return (create_statistics_table(dates=portfolio_value_ts.index, 
            daily_prices=portfolio_value_ts.values, 
            daily_returns=portfolio_returns, 
            COLORS=COLORS),
        )
