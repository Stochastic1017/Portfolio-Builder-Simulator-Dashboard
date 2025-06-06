
from dash import html
import dash_bootstrap_components as dbc

# Function to render each card
def news_article_card_layout(article, COLORS):
    
    sentiment_color = {'positive': 'success', 
                       'neutral': 'secondary', 
                       'negative': 'danger'}
    
    sentiment = article['insights'][0]['sentiment']

    return dbc.Card(
            [
                dbc.CardHeader(
                    style={
                        'backgroundColor': COLORS['card'], 
                        'color': COLORS['text'], 
                        'borderBottom': '1px solid #444',
                    },

                    children=[
                        
                        ### Publisher Details

                        # Publisher logo image 
                        html.Img(src=article['publisher']['favicon_url'], 
                                height="20px", 
                                style={'marginRight': '10px'}),
                        
                        # Publisher name
                        html.A(article['publisher']['name'], 
                            href=article['publisher']['homepage_url'], 
                            target="_blank", className='me-2', 
                            style={'color': COLORS['primary']}),
                        
                        # Sentiment of news article
                        dbc.Badge(sentiment.capitalize(), 
                                  color=sentiment_color.get(sentiment, 'secondary'), 
                                  className="float-end")
                    
                    ],
                
                ),
                
                # Article title
                dbc.CardBody(
                    [
                        html.H5(html.A(article['title'], href=article['article_url'], target="_blank"),
                                className="card-title",
                                style={'color': COLORS['primary']}
                        ),
                    
                        html.Small(f"By {article['author']} | Published: {article['published_utc'][:10]}",
                                className='text-muted',
                                style={'color': COLORS['secondary']}
                        ),
                    
                        html.Br(),
                    
                        html.Img(src=article['image_url'], 
                                 style={'width': '100%', 
                                        'marginTop': '10px', 
                                        'marginBottom': '10px'}),
                    
                        html.P(article['description'], style={'color': COLORS['text']}),
                    
                        html.Div(
                            [
                                dbc.Badge(keyword, color="warning", className="me-1 mb-1", style={'backgroundColor': COLORS['primary'], 'color': COLORS['background']})
                                for keyword in article['keywords']
                            ]
                        )
                    ],
                style={'backgroundColor': COLORS['card'], 'color': COLORS['text']}
                )
            ], className="mb-4 shadow-sm", style={'backgroundColor': COLORS['card'], 'border': 'none'})
