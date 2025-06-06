
import dash
from dash import html
import dash_bootstrap_components as dbc

from dotenv import load_dotenv
from dash_app.helpers.polygon_stock_api import StockTickerInformation
import os

# Define color constants
COLORS = {
    'primary': '#FFD700',      # Golden Yellow
    'secondary': '#FFF4B8',    # Light Yellow
    'background': '#1A1A1A',   # Dark Background
    'card': '#2D2D2D',         # Card Background
    'text': '#FFFFFF'          # White Text
}

load_dotenv()
api_key = os.getenv("POLYGON_API_KEY")
ticker = "AMZN"
STI = StockTickerInformation(ticker=ticker, api_key=api_key)
articles = STI.get_news()['results']

# Function to render each card
def render_article_card(article):
    sentiment_color = {'positive': 'success', 'neutral': 'secondary', 'negative': 'danger'}
    sentiment = article['insights'][0]['sentiment']

    return dbc.Card([
        dbc.CardHeader([
            html.Img(src=article['publisher']['favicon_url'], height="20px", style={'marginRight': '10px'}),
            html.A(article['publisher']['name'], href=article['publisher']['homepage_url'], target="_blank", className='me-2', style={'color': COLORS['primary']}),
            dbc.Badge(sentiment.capitalize(), color=sentiment_color.get(sentiment, 'secondary'), className="float-end")
        ],
        style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'borderBottom': '1px solid #444'}
        ),
        dbc.CardBody([
            html.H5(html.A(article['title'], href=article['article_url'], target="_blank"),
                    className="card-title",
                    style={'color': COLORS['primary']}
            ),
            html.Small(f"By {article['author']} | Published: {article['published_utc'][:10]}",
                       className='text-muted',
                       style={'color': COLORS['secondary']}
            ),
            html.Br(),
            html.Img(src=article['image_url'], style={'width': '100%', 'marginTop': '10px', 'marginBottom': '10px'}),
            html.P(article['description'], style={'color': COLORS['text']}),
            html.Div([
                dbc.Badge(keyword, color="warning", className="me-1 mb-1", style={'backgroundColor': COLORS['primary'], 'color': COLORS['background']})
                for keyword in article['keywords']
            ])
        ],
        style={'backgroundColor': COLORS['card'], 'color': COLORS['text']}
        )
    ], className="mb-4 shadow-sm", style={'backgroundColor': COLORS['card'], 'border': 'none'})


# Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("📊 News Feed for AAPL", className="my-4"),
    dbc.Container([
        html.Div(
            children=[render_article_card(article) for article in articles],
            style={
                "maxHeight": "80vh",
                "overflowY": "scroll",
                "paddingRight": "10px"
            }
        )
    ])
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True)
