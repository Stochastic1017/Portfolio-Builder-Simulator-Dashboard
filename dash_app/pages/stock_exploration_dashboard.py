
import os
import sys
import dash
import dash.dash_table as dt
import pandas as pd

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from dash import (html, dcc, Input, Output, State, callback, ctx, no_update)
import dash_bootstrap_components as dbc
from helpers.polygon_stock_api import StockTickerInformation
from helpers.polygon_stock_historic_plots import (empty_placeholder_figure, create_historic_plots)
from helpers.polygon_stock_metadata import (company_metadata_layout)
from helpers.polygon_stock_news import (news_article_card_layout)

# Register the page
dash.register_page(__name__, path="/pages/stock-exploration-dashboard")

# Load .env to fetch api key
load_dotenv()
api_key = os.getenv("POLYGON_API_KEY")

# Define color constants
COLORS = {
    'primary': '#FFD700',      # Golden Yellow
    'secondary': '#FFF4B8',    # Light Yellow
    'background': '#1A1A1A',   # Dark Background
    'card': '#2D2D2D',         # Card Background
    'text': '#FFFFFF'          # White Text
}

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

verified_button_style = {'padding': '10px', 
                         'backgroundColor': COLORS['text'], 
                         'border': 'none', 
                         'borderRadius': '5px', 
                         'color': COLORS['background'], 
                         'fontWeight': 'bold', 
                         'cursor': 'pointer'}

unverified_button_style = {'padding': '10px', 
                           'border': 'none', 
                           'borderRadius': '5px', 
                           'fontWeight': 'bold', 
                           'cursor': 'pointer'}

verified_button_portfolio = {'padding': '12px',
                            'backgroundColor': COLORS['primary'],
                            'border': 'none',
                            'borderRadius': '8px',
                            'color': '#000000',
                            'fontWeight': 'bold',
                            'fontSize': '1em',
                            'cursor': 'pointer',
                            'marginTop': '10px'}

unverified_button_portfolio = {'padding': '12px',
                               'border': 'none',
                               'borderRadius': '8px',
                               'fontWeight': 'bold',
                               'fontSize': '1em',
                               'cursor': 'pointer',
                               'marginTop': '10px'}

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
        ### Analytics button
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
                        debounce=True,
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

                    # A stylized button to verify if user input ticker is correct
                    html.Button("Verify Ticker",
                        id="btn-verify",
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
                            'flexGrow': 1,
                            'minHeight': 0,
                            'marginTop': '10px',
                            'marginBottom': '10px',
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
                                    'maxHeight': '300px',
                                    'border': '1px solid #ccc',
                                    'borderRadius': '10px',
                                },
                                style_cell={
                                    'padding': '10px',
                                    'textAlign': 'left',
                                    'backgroundColor': '#f9f9f9',
                                    'color': '#333',
                                    'fontFamily': 'Arial',
                                },
                                style_header={
                                    'backgroundColor': '#0E1117',
                                    'color': 'white',
                                    'fontWeight': 'bold',
                                    'border': '1px solid #ccc'
                                },
                                editable=False,           # Disable editing
                                row_deletable=True,       # Disable row deletion
                                sort_action="none",       # Disable sorting
                                filter_action="none",     # Disable filtering
                                page_action="none",       # Disable pagination
                                style_as_list_view=True,  # Remove any default interactivity styling
                                selected_rows=[],         # Prevent row selection
                                active_cell=None,         # Prevent active cell highlighting
                                row_selectable=None,      # Disable row selection
                                data=[],
                            ),

                            # caching users stocks added to portfolio
                            dcc.Store(id='portfolio-store', data=[]),
                        ]
                    ),

                    # Store verify status in cache
                    dcc.Store(id="verify-status", data={"verified": False}),

                ]
            ),
                
            # Right content
            html.Div(id="main-output-section",
                     style={
                        'backgroundColor': COLORS['background'],
                        'borderRadius': '10px',  # rounded edges
                        'height': '100%',
                        'width': '100%',
                        'boxSizing': 'border-box',
                        'position': 'relative',
                        'overflow': 'hidden' # Prevent layout overflow
                    },
                     children=[dcc.Graph(figure=empty_placeholder_figure(COLORS))]   
                ),

            ]
        ),

        # Analytics button
        html.Div(
            style={
                'display': 'flex',
                'justifyContent': 'flex-end',
                'marginTop': '20px',
                'marginLeft': '320px',
                'maxWidth': 'calc(100% - 320px)',
                'paddingRight': '20px',
                'overflowY': 'auto',
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

# Verify ticker API call and reset status if ticker changes
@callback(
    Output("verify-status", "data"),
    [
        Input("btn-verify", "n_clicks"),
        Input("inp-ticker", "value")
    ],
    prevent_initial_call=True,
)
def handle_verify_and_input(n_clicks, ticker):
    
    trigger_id = ctx.triggered_id
    
    if trigger_id == "btn-verify":
        result = validate_stock_ticker(ticker, api_key)

        if "error" in result:
            return {"verified": False}
        
        return {"verified": True, **result}

    elif trigger_id == "inp-ticker":
        return {"verified": False}

    return no_update

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
    Input("verify-status", "data")
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
    Output("main-output-section", "children", allow_duplicate=True),
    Input("verify-status", "data"),
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

@callback(
    Output("main-output-section", "children"),
    [
        Input("btn-news", "n_clicks"),
        Input("btn-performance", "n_clicks"),
        Input("btn-add", "n_clicks"),
    ],
    State("verify-status", "data"),
    prevent_initial_call=True
)
def update_main_output(verify_clicks, news_clicks, hist_clicks, data):
    
    # Recovering cached data from API call
    company_info = data['company_info']
    historical_df = pd.read_json(data['historical_json'], orient="records")

    # Recovering news articles from API call
    news_articles = data['news']['results']

    button_id = ctx.triggered_id
    if button_id == "btn-news":

        return dbc.Container([html.H2(f"News Feed for {company_info['name']}", className="my-4"),
            dbc.Container([
                html.Div(
                    children=[news_article_card_layout(article, COLORS) for article in news_articles],
                    style={
                        "maxHeight": "80vh",
                        "overflowY": "scroll",
                        "paddingRight": "10px"
                    }
                )
            ])
        ], fluid=True)

    elif button_id == "btn-performance":
        return (create_historic_plots(company_info['name'], historical_df, COLORS), ) 
    
    elif button_id == "btn-add":
        return no_update

# Upon "add to portfolio" click, append to table
@callback(
    Output("portfolio-store", "data"),
    Input("btn-add", "n_clicks"),
    State("verify-status", "data"),
    State("portfolio-store", "data"),
    prevent_initial_call=True
)
def add_to_portfolio(n_clicks, verify_data, portfolio_data):
    
    if not verify_data.get("verified"):
        return portfolio_data

    new_entry = {
        "ticker": verify_data.get("ticker", "N/A"),
        "fullname": verify_data["company_info"]["name"]
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
