
import os
import sys
import dash
import numpy as np
import dash.dash_table as dt
import plotly.graph_objects as go
from plotly.graph_objs import Figure

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from dash import (html, dcc, Input, Output, State, callback, callback_context, no_update)
from helpers.polygon_stock_api import StockTickerInformation
from helpers.polygon_stock_historic_plots import (empty_placeholder_figure, create_historic_plots)
from helpers.polygon_stock_metadata import company_metadata_layout

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
        ### Cache Memory
        #################

        dcc.Store(id='cached-ticker-data', storage_type='memory'),

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
                
            # Right content
            html.Div(
                id="main-output-section",
                style={
                    'backgroundColor': COLORS['card'],
                    'borderRadius': '10px',  # rounded edges
                    'height': '100%',
                    'width': '100%',
                    'boxSizing': 'border-box',
                    'overflow': 'hidden' # Prevent layout overflow
                },
                children=[]   
                ),

            ]
        ),

        # Analytics button
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

@callback(
    Output('cached-ticker-data', 'data'),
    [
        Input('btn-performance', 'n_clicks'),
        Input('btn-metadata', 'n_clicks'),
        Input('btn-news', 'n_clicks'),
    ],
    State('inp-ticker', 'value'),
    prevent_initial_call=True
)
def fetch_and_cache_api_data(*args):

    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    ticker = args[-1]  # last arg is the State input

    if not ticker:
        return dash.no_update

    try:        

        # Single API call one any button click and caching result
        polygon_api = StockTickerInformation(ticker=ticker, api_key=api_key)

        # Fetch and parse metadata
        metadata = polygon_api.get_metadata()
        company_info = metadata['results']
        branding = company_info.get('branding', {})
        logo_url_with_key = f"{branding['logo_url']}?apiKey={api_key}"
        address = company_info.get('address', {})

        # Fetch and parse news
        news = polygon_api.get_news()
 
        # Fetch and create historical performance figures
        historical_data = polygon_api.get_all_data()
        historical_fig = create_historic_plots(company_info['name'],
                                               historical_data)

    except Exception as e:
        return {'error': str(e)}

    return {'triggered_id': triggered_id,
            'ticker': ticker, 
            'company_info': company_info,
            'branding': branding,
            'logo_url_with_key': logo_url_with_key,
            'address': address, 
            'news': news, 
            'historical_fig': historical_fig.to_dict()}

@callback(
    Output('main-output-section', 'children'),
    Input('cached-ticker-data', 'data'),
    prevent_initial_call=True
)
def update_main_output(cached_data):
    if not cached_data or 'error' in cached_data:
        return dcc.Graph(
            id="main-output-graph",
            figure=empty_placeholder_figure(),
            config={'responsive': True},
            style={'height': '100%', 'width': '100%'}
        )

    triggered_id = cached_data.get('triggered_id')

    if triggered_id == 'btn-metadata':
        return company_metadata_layout(
            company_info=cached_data.get('company_info', {}),
            branding={'logo_url': cached_data.get('logo_url_with_key', '')},
            logo_url_with_key=cached_data.get('logo_url_with_key', ''),
            address=cached_data.get('address', {})
        )

    elif triggered_id == 'btn-news':
        return html.Div("News section layout not implemented yet.")

    elif triggered_id == 'btn-performance':
        return dcc.Graph(
            id="main-output-graph",
            figure=go.Figure(cached_data['historical_fig']),
            config={'responsive': True},
            style={'height': '100%', 'width': '100%'}
        )

    return no_update
