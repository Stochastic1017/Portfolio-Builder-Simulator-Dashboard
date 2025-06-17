
import os
import sys
import dash
import pandas as pd
import dash.dash_table as dt

from io import StringIO
from dash import html, Input, Output, callback

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.markowitz_portfolio_theory import plot_efficient_frontier

dash.register_page(__name__, path="/pages/portfolio-builder")

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

        html.Div(id="portfolio-summary"),
        html.Div(id="portfolio-table-output"),
        html.Div(id="efficient-frontier")
    ]

)

@callback(
    Output("portfolio-summary", "children"),
    Output("portfolio-table-output", "children"),
    Input("portfolio-store", "data")
)
def load_portfolio(data):

    # Prepare rows for the table
    table_data = []
    for entry in data:
        hist_df = pd.read_json(StringIO(entry["historical_json"]), orient="records")
        hist_df['date'] = pd.to_datetime(hist_df['date'])

        # Get latest price from most recent record
        latest_price = hist_df['close'].iloc[-1]
        table_data.append({
            "Ticker": entry["ticker"],
            "Full Name": entry["fullname"],
            "Latest Price": latest_price,
            "SIC": entry["sic_description"],
            "Market Cap": entry["market_cap"],
            "Weight (%)": 0.0
        })

    # Column formatting and properties
    columns = [
        {"name": "Ticker", "id": "Ticker", "type": "text"},
        {"name": "Full Name", "id": "Full Name", "type": "text"},
        {"name": "SIC", "id": "SIC", "type": "text"},
        {"name": "Latest Price", "id": "Latest Price", "type": "numeric", "format": dt.FormatTemplate.money(0)},
        {"name": "Market Cap", "id": "Market Cap", "type": "numeric", "format": dt.FormatTemplate.money(0)},
        {"name": "Weight (%)", "id": "Weight (%)", "type": "numeric", "editable": True}
    ]

    # Create DataTable
    table = dt.DataTable(
        data=table_data,
        columns=columns,
        style_table={
            "maxHeight": "200px",
            "overflowY": "auto",
            "border": "1px solid #444"
        },
        style_cell={
            "padding": "10px",
            "backgroundColor": COLORS["background"],
            "color": COLORS["text"],
            "border": "1px solid #555",
            "textAlign": "left",
            "minWidth": "100px",
            "maxWidth": "200px",
            "whiteSpace": "normal"
        },
        style_header={
            "backgroundColor": COLORS["card"],
            "color": COLORS["primary"],
            "fontWeight": "bold"
        },
        row_deletable=False
    )

    summary = html.Div(
        f"{len(data)} stock(s) loaded into portfolio."
        )

    return summary, table

@callback(
    Output("efficient-frontier", "children"),
    Input("portfolio-store", "data")
)
def markowitz_portfolio_theory_plot(data):

    return plot_efficient_frontier(data, COLORS)