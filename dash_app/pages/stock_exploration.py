
import os
import uuid
import sys
import dash
import numpy as np
import pandas as pd
import dash.dash_table as dt
import dash_bootstrap_components as dbc

from datetime import timedelta
from io import StringIO
from dotenv import load_dotenv
from dash import (html, dcc, Input, Output, State, callback, ctx, no_update)

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.polygon_stock_api import StockTickerInformation

from helpers.polygon_stock_historic_plots import (dash_range_selector, 
                                                  create_historic_plots, 
                                                  create_statistics_table)

from helpers.polygon_stock_metadata import (company_metadata_layout)

from helpers.polygon_stock_news import (news_article_card_layout)

from helpers.button_styles import (COLORS, 
                                   verified_button_portfolio, unverified_button_portfolio,
                                   verified_button_style, unverified_button_style, 
                                   default_style_time_range, active_style_time_range)

# Register the page
dash.register_page(__name__, path="/pages/stock-exploration")

# Load .env to fetch api key
load_dotenv()
api_key = os.getenv("POLYGON_API_KEY")

# Stock ticker validation procedure
def validate_stock_ticker(ticker, api_key):
    
    """
    Function to validate ticker input by user. 
    """
    
    if not ticker or not ticker.strip():
        return {'error': 'Ticker is empty or invalid.'}

    try:
        polygon_api = StockTickerInformation(ticker=ticker.upper().strip(), api_key=api_key)

        # Metadata verification
        metadata = polygon_api.get_metadata()
        if not metadata or 'results' not in metadata:
            return {'error': 'No metadata found for this ticker.'}
        
        company_info = metadata['results']
        if not company_info or 'name' not in company_info:
            return {'error': 'Company information is incomplete.'}

        branding = company_info.get('branding', {})
        if not branding.get('logo_url'):
            return {'error': 'Branding/logo missing.'}

        logo_url_with_key = f"{branding['logo_url']}?apiKey={api_key}"
        address = company_info.get('address', {})
        if not address:
            return {'error': 'Company address is missing.'}

        # News verification
        news = polygon_api.get_news()
        if not news or len(news)==0:
            return {'error': 'No news found for this ticker.'}

        # Historical performance verification
        historical_df = polygon_api.get_all_data()
        if historical_df is None or historical_df.empty:
            return {'error': 'No historical data available.'}

        return {
            'verified': True,
            'ticker': ticker,
            'company_info': company_info,
            'branding': branding,
            'logo_url_with_key': logo_url_with_key,
            'address': address,
            'news': news,
            'historical_json': historical_df.to_json(orient="records")
        }

    except Exception as e:
        return {'error': f'Ticker verification failed: {str(e)}'}

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
                id="stock-exploration-console",
                style={
                    'backgroundColor': COLORS['card'],
                    'borderRadius': '10px',  
                    'padding': '20px 20px 40px 20px',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'gap': '15px',
                    'height': '100%',
                    'boxSizing': 'border-box',
                    'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)', 
                    'overflow': 'hidden', 
                },

                children=[

                    # Stock ticker input + verify ticker button
                    html.Div(
                        style={
                            'display': 'flex',
                            'flexDirection': 'row',
                            'alignItems': 'center'
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
                                    'width': '200px',
                                    'padding': '10px',
                                    'backgroundColor': COLORS['background'],
                                    'border': f'1px solid {COLORS['primary']}',
                                    'borderRadius': '5px',
                                    'color': COLORS['text'],
                                    'fontSize': '1em',
                                    'marginRight': '10px'
                                }
                            ),

                            # A stylized button to verify if user input ticker is correct
                            html.Button("Verify Ticker",
                                id="btn-verify-ticker",
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
                            
                            # Button for user to check latest news on stock ticker
                            html.Button("Check Latest News", 
                                id="btn-news", 
                                disabled=True, 
                            ),                    

                            # Button for user to check historic performance of stock ticker
                            html.Button("Check Historic Performance", 
                                id="btn-performance", 
                                disabled=True,
                            ),
                        ],                    
                    ),

                    # A stylized button for users to add stock ticker to portfolio
                    html.Button("Add to Portfolio", 
                        id="btn-add", 
                        n_clicks=0,
                        disabled=True,
                    ),

                    # Scrollable portfolio table
                    html.Div(
                        style={
                            'flex': '1 1 auto',
                            'display': 'flex',
                            'minHeight': 0,
                            'marginTop': '10px',
                        },
                        
                        children=[
                            
                            dt.DataTable(
                                id='portfolio-table',
                                columns=[
                                    {'id': 'ticker', 'name': 'Ticker'},
                                    {'id': 'fullname', 'name': 'Company Name'}
                                ],
                                style_table={
                                    'overflowY': 'auto',
                                    'overflowX': 'auto',
                                    'height': '100%',
                                    'border': '1px solid #ccc',
                                    'borderRadius': '10px',
                                    'marginBottom': '20px',
                                },
                                style_cell={
                                    'padding': '10px',
                                    'textAlign': 'left',
                                    'backgroundColor': '#f9f9f9',
                                    'color': '#333',
                                    'fontFamily': 'Arial',
                                    'minWidth': '100px',
                                    'maxWidth': '250px',       # ✅ prevents extra-wide cells
                                    'whiteSpace': 'normal',    # ✅ allow wrapping if needed
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                },
                                style_header={
                                    'backgroundColor': '#0E1117',
                                    'color': 'white',
                                    'fontWeight': 'bold',
                                    'border': '1px solid #ccc'
                                },
                                editable=False,           # Disable editing
                                row_deletable=True,       # Enable row deletion
                                sort_action="none",       # Disable sorting
                                filter_action="none",     # Disable filtering
                                page_action="none",       # Disable pagination
                                style_as_list_view=True,  # Remove any default interactivity styling
                                selected_rows=[],         # Prevent row selection
                                active_cell=None,         # Prevent active cell highlighting
                                row_selectable=None,      # Disable row selection
                                data=[],
                            ),
                        ]
                    ),
                ]
            ),
                
            # Right content
            html.Div(
                id="stock-exploration-main-content",
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
                    'overflow': 'hidden'
                },
                
                children=[
                    
                    html.Div([
                        html.H3("Welcome!", style={'color': COLORS['primary'], 'marginBottom': '1rem'}),

                        html.Br(),

                        html.Div([
                            html.P("This application was built by a single individual with a passion for coding, finance, mathematics, and statistics. " \
                                   "It is a personal project and does not represent the views or interests of any company or institution.",
                                style={'color': COLORS['text'], 'fontSize': '0.95rem', 'marginBottom': '0.5rem'}),

                            html.P("The insights provided are grounded in classical statistical theory, but rest on strong assumptions that often do not reflect real-world markets. " \
                                   "Please exercise caution and do not rely solely on this tool for financial decisions.",
                                style={'color': COLORS['text'], 'fontSize': '0.95rem', 'marginBottom': '0.5rem'}),

                            html.P("All market data is sourced from the Polygon.io API. " \
                                   "The free tier limits requests to 10 API calls per minute, which allows approximately 3 tickers to be analyzed every 60 seconds.",
                                style={'color': COLORS['text'], 'fontSize': '0.95rem'}),

                        html.Br(),

                        html.P("To get started, please follow the steps below:", 
                               style={'color': COLORS['text'], 
                                      'fontSize': '1.1rem'}),

                        html.Ol([
                            html.Li("Input a stock ticker and click 'Verify Ticker'.", 
                                    style={'color': COLORS['text']}),
                            html.Li("Explore the stock's recent news and historical performance.", 
                                    style={'color': COLORS['text']}),
                            html.Li("Choose to add it to your portfolio.", 
                                    style={'color': COLORS['text']}),
                            html.Li("Enter at least two tickers in portfolio before proceeding.", 
                                    style={'color': COLORS['text']}),], 
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

        # Portfolio Builder
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
                        # Go to Portfolio Builder Page
                        dcc.Link(
                            html.Button(
                                "Go to Portfolio Builder",
                                id="btn-portfolio-builder",
                                n_clicks=0,
                            ),
                            href="/pages/portfolio-builder",
                            refresh=False  # Set to True if you want full page reload
                        )

                    ]
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

# Verify ticker API call and reset status if ticker input changes
@callback(
    Output("verify-ticker", "data"),
    [
        Input("btn-verify-ticker", "n_clicks"),
        Input("inp-ticker", "value")
    ],
    prevent_initial_call=True,
)
def handle_verify_ticker(_, ticker):
    
    trigger_id = ctx.triggered_id
    
    if trigger_id == "btn-verify-ticker":
        result = validate_stock_ticker(ticker, api_key)

        if "error" in result:
            return {"verified": False}
        
        return {"verified": True, **result}

    elif trigger_id == "inp-ticker":
        return {"verified": False}

    return no_update

# Change validation symbol upon successful verification
@callback(
    Output("inp-ticker", "valid"),
    Output("inp-ticker", "invalid"),
    Output("inp-ticker", "key"),
    Input("verify-ticker", "data"),
    prevent_initial_call=True
)
def set_ticker_validation(verify_ticker):
    is_verified = verify_ticker.get("verified", False)
    
    # Change the key so Dash forces a component refresh
    dynamic_key = f"key-{uuid.uuid4()}"
    
    return is_verified, not is_verified, dynamic_key

# Toggle styles between enabled/disabled status
@callback(
    [
        Output("btn-news", "disabled"),
        Output("btn-news", "style"),
        Output("btn-news", "className"),

        Output("btn-performance", "disabled"),
        Output("btn-performance", "style"),
        Output("btn-performance", "className"),

        Output("btn-add", "disabled"),
        Output("btn-add", "style"),
        Output("btn-add", "className"),
    ],
    Input("verify-ticker", "data")
)
def toggle_button_states(verify_status):
    is_verified = verify_status.get("verified", False)

    if is_verified:
        return (
            False, verified_button_style, "simple",
            False, verified_button_style, "simple",
            False, verified_button_portfolio, "special",
        )
    
    else:
        return (
            True, unverified_button_style, "",
            True, unverified_button_style, "",
            True, unverified_button_portfolio, "",
        )

# Upon successful verification, display metadata
@callback(
    Output("stock-exploration-main-content", "children", allow_duplicate=True),
    Input("verify-ticker", "data"),
    prevent_initial_call=True
)
def display_metadata_on_verify(data):
    if not data or not data.get("verified"):
        return dash.no_update

    company_info = data['company_info']
    branding = data['branding']
    logo_url_with_key = data['logo_url_with_key']
    address = data['address']

    return company_metadata_layout(company_info, branding, logo_url_with_key, address, COLORS)

# Update main output section depending on what is chosen by user
@callback(
    Output("stock-exploration-main-content", "children"),
    [
        Input("btn-news", "n_clicks"),
        Input("btn-performance", "n_clicks"),
        Input("btn-add", "n_clicks"),
        Input("selected-range", "data"),
    ],
    State("verify-ticker", "data"),
    prevent_initial_call=True
)
def update_main_output(_, __, ___, ____, data):
    
    # Recovering cached data from API call
    company_info = data['company_info']
    news_articles = data['news']['results']

    # Button clicked by user, one of the fllowing:
    button_id = ctx.triggered_id
    if button_id == "selected-range":
        raise dash.exceptions.PreventUpdate # Prevent caallback overlap
    
    # 1. Check Latest News
    if button_id == "btn-news":
        return  dbc.Container(
                    
                    children=[
                        
                        html.H2(f"News Feed for {company_info['name']}", 
                                className="my-4"),
                        dbc.Container(
                            
                            children=[
                            
                            html.Div(
                            style={
                                    "maxHeight": "80vh",
                                    "overflowY": "scroll",
                                    "paddingRight": "10px"
                                    },
                            children=[news_article_card_layout(article, COLORS) for article in news_articles],
                                )
                            ]
                        )
                ], fluid=True)

    # 2. Check Historic Performance
    elif button_id == "btn-performance":
        return html.Div(
                id="stock-exploration-main-content",
                style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'height': '100%',
                    'width': '100%',
                    'overflow': 'hidden',
                },
                
                children=[
                    
                    dash_range_selector(default_style=default_style_time_range),
                    dcc.Tabs(
                        id="performance-tabs",
                        value='tab-plot',
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
                                value='tab-plot',
                                style={
                                    "backgroundColor": COLORS['background'],
                                    "color": COLORS['text'],
                                    "padding": "6px 18px",
                                    "fontSize": "14px",
                                    "fontWeight": "bold",
                                    "border": "none",
                                    "borderBottom": f"2px solid transparent",
                                },
                                selected_style={
                                    "backgroundColor": COLORS['background'],
                                    "color": COLORS['primary'],
                                    "padding": "6px 18px",
                                    "fontSize": "14px",
                                    "fontWeight": "bold",
                                    "border": "none",
                                    "borderBottom": f"2px solid {COLORS['primary']}",
                                },
                            ),
                            
                            dcc.Tab(
                                label="Statistical Summary",
                                value='tab-stats',
                                style={
                                    "backgroundColor": COLORS['background'],
                                    "color": COLORS['text'],
                                    "padding": "6px 18px",
                                    "fontWeight": "bold",
                                    "fontSize": "14px",
                                    "border": "none",
                                    "borderBottom": f"2px solid transparent",
                                },
                                selected_style={
                                    "backgroundColor": COLORS['background'],
                                    "color": COLORS['primary'],
                                    "padding": "6px 18px",
                                    "fontWeight": "bold",
                                    "fontSize": "14px",
                                    "border": "none",
                                    "borderBottom": f"2px solid {COLORS['primary']}",
                                },
                            ),
                        
                        ]
                    ),

                    html.Div(id="historical-plot-container", style={"flex": "1", "overflow": "hidden"}),
                ]
            )
    
    # 3. Add to Portfolio
    elif button_id == "btn-add":
        return no_update

# Highlight buttons based on time-range selected
@callback(
    [
        Output("range-1M", "style"),
        Output("range-3M", "style"),
        Output("range-6M", "style"),
        Output("range-1Y", "style"),
        Output("range-5Y", "style"),
        Output("range-all", "style"),
        Output("selected-range", "data")
    ],
    [
        Input("range-1M", "n_clicks"),
        Input("range-3M", "n_clicks"),
        Input("range-6M", "n_clicks"),
        Input("range-1Y", "n_clicks"),
        Input("range-5Y", "n_clicks"),
        Input("range-all", "n_clicks"),
    ]
)
def update_range_styles(*btn_clicks):
    button_ids = [
        "range-1M", "range-3M", "range-6M",
        "range-1Y", "range-5Y", "range-all"
    ]

    # If no button has been clicked yet, fall back to "All"
    if not any(click and click > 0 for click in btn_clicks):
        selected = "range-all"
    else:
        selected = ctx.triggered_id or "range-all"

    style_map = {btn_id: default_style_time_range for btn_id in button_ids}
    style_map[selected] = active_style_time_range

    return (
        style_map["range-1M"],
        style_map["range-3M"],
        style_map["range-6M"],
        style_map["range-1Y"],
        style_map["range-5Y"],
        style_map["range-all"],
        selected.split("-")[1]  # "1M", "3M", ..., "all"
    )

# Update historic daily plot based on range selected
@callback(
    Output("historical-plot-container", "children"),
    Input("performance-tabs", "value"),
    Input("selected-range", "data"),
    State("verify-ticker", "data"),
    prevent_initial_call=True
)
def update_plot_on_range_change(active_tab, selected_range, data):

    historical_df = pd.read_json(StringIO(data['historical_json']), orient="records")
    historical_df['date'] = pd.to_datetime(historical_df['date'])

    today = pd.Timestamp.today()
    time_deltas = {
        "1M": timedelta(days=30),
        "3M": timedelta(days=90),
        "6M": timedelta(days=180),
        "1Y": timedelta(days=365),
        "5Y": timedelta(days=1825),
        "all": None
    }

    cutoff = time_deltas.get(selected_range.upper(), None)

    if cutoff:
        start_date = today - cutoff
        filtered_df = historical_df[historical_df['date'] >= start_date]
    
    else:
        filtered_df = historical_df

    # Extract relevant metrics (filtered)
    dates = np.asarray(filtered_df['date'])
    daily_prices = np.asarray(filtered_df['close'])
    daily_returns = np.asarray(filtered_df['close'].pct_change().dropna())

    if active_tab == "tab-plot":
        return create_historic_plots(data['company_info']['name'], dates, daily_prices, daily_returns, COLORS)
    
    elif active_tab == "tab-stats":
        return create_statistics_table(dates, daily_prices, daily_returns, COLORS)

# Upon "add to portfolio" click, append to table and cache data
@callback(
    Output("portfolio-store", "data"),
    Input("btn-add", "n_clicks"),
    State("verify-ticker", "data"),
    State("portfolio-store", "data"),
    prevent_initial_call=True
)
def add_to_portfolio(_, verify_data, portfolio_data):
    
    if not verify_data.get("verified"):
        return portfolio_data

    new_entry = {
        "ticker": verify_data["company_info"]["ticker"],
        "fullname": verify_data["company_info"]["name"],
        "sic_description": verify_data["company_info"]["sic_description"],
        "market_cap": verify_data["company_info"]["market_cap"],
        "historical_json": verify_data["historical_json"]
    }

    # Avoid duplicates
    if new_entry not in portfolio_data:
        portfolio_data.append(new_entry)

    return portfolio_data

# Update table visual
@callback(
    Output("portfolio-table", "data"),
    Input("portfolio-store", "data")
)
def update_portfolio_table(data):
    return data

# Allow users to navigate to portfolio analytics page
# Provided at least two tickers were selected
@callback(
    Output("btn-portfolio-builder", "disabled"),
    Output("btn-portfolio-builder", "style"),
    Output("btn-portfolio-builder", "className"),
    Input("portfolio-store", "data")
)
def update_portfolio_analytics_button(tickers):
    if tickers is None or len(tickers) < 2:
        return True, unverified_button_portfolio, ""
    
    return False, verified_button_portfolio, "special"
