
# Define color constants
COLORS = {
    'primary': '#FFD700',      # Golden Yellow
    'secondary': '#FFF4B8',    # Light Yellow
    'background': '#1A1A1A',   # Dark Background
    'card': '#2D2D2D',         # Card Background
    'text': '#FFFFFF'          # White Text
}

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

default_style_time_range = {
    'padding': '4px 12px',  # smaller pill-style padding
    'backgroundColor': COLORS['card'],
    'color': COLORS['text'],
    'border': f'1px solid {COLORS["primary"]}',
    'borderRadius': '999px',  # makes it pill-shaped
    'cursor': 'pointer',
    'fontSize': '0.85rem',
    'marginRight': '8px',  # spacing between buttons
    'display': 'inline-block',
    'transition': 'all 0.2s ease-in-out',
}

active_style_time_range = {
    **default_style_time_range,
    'backgroundColor': COLORS['primary'],
    'color': COLORS['background'],
    'fontWeight': '600',
    'boxShadow': '0 0 6px rgba(0, 0, 0, 0.15)',
    'transform': 'scale(1.05)',
}
