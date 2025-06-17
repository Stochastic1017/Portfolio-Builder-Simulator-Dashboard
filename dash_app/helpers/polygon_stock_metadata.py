
from dash import html
from polygon_stock_api import StockTickerInformation

def company_metadata_layout(company_info, branding, logo_url_with_key, address, COLORS):
    return html.Div(
        style={
            'padding': '30px',
            'height': '100%',
            'width': '100%',
            'boxSizing': 'border-box',
            'overflowY': 'auto',
            'display': 'flex',
            'flexDirection': 'column',
            'gap': '20px'
        },
        
        children=[

        # Company name and logo
        html.Div(
            style={'display': 'flex', 
                   'alignItems': 'center', 
                   'gap': '20px'},
            
            children=[

                html.Div(
                style={
                    'backgroundColor': '#ffffff',  # light background
                    'padding': '6px',
                    'borderRadius': '10px',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center'
                },

                children=[
                            html.Img(
                                src=logo_url_with_key,
                                style={
                                    'height': '60px',
                                    'objectFit': 'contain',
                                    'filter': 'none'  # prevent any inversion
                                }
                            )
                        ]
                    ) 
                    
                    if branding.get('logo_url') else None,

                    html.H2(company_info['name'], 
                                    style={'margin': 0, 
                                        'color': COLORS['primary']})
                        ]
                    ),

            # Description
            html.P(company_info.get('description', ''), 
                style={'fontSize': '1.1em'}),

            # Basic info section
            html.Div(
                style={'display': 'grid', 
                    'gridTemplateColumns': 'repeat(auto-fit, minmax(220px, 1fr))', 
                    'gap': '15px'},
                children=[
                    html.Div([html.Strong("Ticker: "), company_info['ticker']]),
                    html.Div([html.Strong("Market: "), company_info['market']]),
                    html.Div([html.Strong("Exchange: "), company_info['primary_exchange']]),
                    html.Div([html.Strong("Locale: "), company_info['locale'].upper()]),
                    html.Div([html.Strong("Type: "), company_info['type']]),
                    html.Div([html.Strong("Active: "), str(company_info['active'])]),
                    html.Div([html.Strong("Currency: "), company_info['currency_name'].upper()]),
                    html.Div([html.Strong("SIC: "), f"{company_info['sic_code']} – {company_info['sic_description']}"]),
                    html.Div([html.Strong("Market Cap: "), f"${company_info['market_cap']:,.0f}"]),
                    html.Div([html.Strong("Employees: "), f"{company_info.get('total_employees', 'N/A'):,}"]),
                    html.Div([html.Strong("List Date: "), company_info['list_date']]),
                ]
            ),

            # Address and Contact
            html.Div([
                html.H4("Contact", style={'color': COLORS['primary'], 'marginTop': '20px'}),
                html.P(company_info.get('phone_number', 'N/A')),
                html.P(f"{address.get('address1', '')}, {address.get('city', '')}, {address.get('state', '')} {address.get('postal_code', '')}"),
                html.A("Visit Website", href=company_info['homepage_url'], target="_blank", style={'color': COLORS['primary'], 'textDecoration': 'underline'})
            ])
        ]
    )
