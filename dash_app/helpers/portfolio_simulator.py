
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy import stats
from io import StringIO
from dash import dcc, html
from arch import arch_model
from pmdarima.arima import auto_arima
from plotly.subplots import make_subplots

# Append the current directory to the system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def portfolio_dash_range_selector(default_style):
    
    return html.Div(
        id="portfolio-simulator-range-selector-container",
        children=[
            html.Div(
                style={"display": "flex", 
                       "justifyContent": "start", 
                       "flexWrap": "wrap",
                       "gap": "10px",
                       "paddingTop": "10px", 
                       "marginBottom": "10px"},

                children=[
                    html.Button("1M", id="portfolio-range-1M", n_clicks=0, style=default_style, className="simple"),
                    html.Button("3M", id="portfolio-range-3M", n_clicks=0, style=default_style, className="simple"),
                    html.Button("6M", id="portfolio-range-6M", n_clicks=0, style=default_style, className="simple"),
                    html.Button("1Y", id="portfolio-range-1Y", n_clicks=0, style=default_style, className="simple"),
                    html.Button("5Y", id="portfolio-range-5Y", n_clicks=0, style=default_style, className="simple"),
                    html.Button("All", id="portfolio-range-all", n_clicks=0, style=default_style, className="simple"),
                ],

            )
        ]
    )

def parse_ts_map(selected_tickers, portfolio_weights, portfolio_store, budget, threshold=1e-6):
    ts_map = {}
    selected_values = {t['value'] for t in selected_tickers}
    total_value = None

    for entry, weight in zip(portfolio_store, portfolio_weights):
        if weight <= threshold:
            continue
        if entry['ticker'] not in selected_values:
            continue

        df = pd.read_json(StringIO(entry["historical_json"]), orient="records")
        df['date'] = pd.to_datetime(df['date'], unit='ms' if isinstance(df['date'].iloc[0], (int, float)) else None)
        df.set_index('date', inplace=True)

        price_series = df['close'].sort_index()

        # Use budget to get actual dollar allocation and number of shares
        latest_price = price_series.iloc[-1]
        dollar_alloc = weight * budget
        shares = dollar_alloc / latest_price if latest_price > 0 else 0

        # Dollar value of the ticker over time
        dollar_ts = price_series * shares

        ts_map[entry['ticker']] = {
            "weight": weight,
            "shares": shares,
            "price": price_series,
            "value": dollar_ts,
        }

        total_value = dollar_ts if total_value is None else total_value.add(dollar_ts, fill_value=0)

    return ts_map, total_value

def forecast_arima(log_returns, forecast_until):
    
    # Fit best ARIMA model using AIC
    model = auto_arima(
        log_returns,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False,
        information_criterion='aic'
    )

    # Forecast horizon (in number of periods)
    last_date = log_returns.index[-1]
    target_date = pd.to_datetime(forecast_until)
    freq = pd.infer_freq(log_returns.index) or 'B'  # Business day fallback
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(1, unit='D'), end=target_date, freq=freq)
    n_periods = len(forecast_index)

    # Make forecast
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
    
    # Prepare forecast DataFrame
    forecast_df = pd.DataFrame({
        'Forecast': forecast,
        'Lower CI': conf_int[:, 0],
        'Upper CI': conf_int[:, 1]
    }, index=forecast_index)
    
    return forecast_df

def arima_forecast_plot(full_name, dates, daily_prices, log_returns, forecast_df, COLORS):

    ######################
    ### Compute Forecasted Prices
    ######################

    # Start from last known price
    last_price = daily_prices[-1]
    cumulative_returns = forecast_df['Forecast'].cumsum()
    forecasted_prices = last_price * np.exp(cumulative_returns)

    # Upper and lower confidence bands on price (exponential of cumulative bounds)
    cumulative_lower = forecast_df['Lower CI'].cumsum()
    cumulative_upper = forecast_df['Upper CI'].cumsum()

    lower_prices = last_price * np.exp(cumulative_lower)
    upper_prices = last_price * np.exp(cumulative_upper)

    forecast_dates = forecast_df.index
    print(forecast_df)

    ######################
    ### Setup Subplots
    ######################

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=[
            "Daily Prices with ARIMA Forecast Cone (95% CI)",
            "Daily Returns with 95% Confidence Intervals",
            "Histogram of Log Returns with 95% Confidence Intervals"
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        shared_xaxes=False,
        shared_yaxes=False,
        column_widths=[0.7, 0.3]
    )

    ######################
    ### Price + Forecast Cone
    ######################

    # Historical prices
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=daily_prices,
            mode='lines',
            name='Historical Prices',
            line=dict(color=COLORS['primary'])
        ),
        row=1, col=1
    )

    # Forecasted prices (mean)
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecasted_prices,
            mode='lines',
            name='ARIMA Forecast (Mean)',
            line=dict(color="#00C2FF", dash='dash')
        ),
        row=1, col=1
    )

    # Confidence interval cone
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
            y=np.concatenate([upper_prices, lower_prices[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 194, 255, 0.1)',  # Light cyan translucent
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Forecast CI',
            showlegend=True
        ),
        row=1, col=1
    )

    ######################
    ### Returns + CI
    ######################

    # Historical returns
    fig.add_trace(
        go.Scatter(
            x=log_returns.index,
            y=log_returns,
            mode='lines',
            name='Log Returns',
            line=dict(color=COLORS['primary'])
        ),
        row=2, col=1
    )

    # Forecasted returns
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_df['Forecast'],
            mode='lines',
            name='Forecasted Returns',
            line=dict(color="#00C2FF", dash='dash')
        ),
        row=2, col=1
    )

    # Confidence interval bounds
    mean_return = log_returns.mean()
    std_return = log_returns.std()
    lower_bound = mean_return - 1.96 * std_return
    upper_bound = mean_return + 1.96 * std_return

    ######################
    ### Histogram
    ######################

    return_range = np.linspace(log_returns.min(), log_returns.max(), 500)
    hist_counts, _ = np.histogram(log_returns, bins=50, density=True)

    # Histogram
    fig.add_trace(
        go.Histogram(
            y=log_returns,
            name="Log Returns Histogram",
            marker_color=COLORS['primary'],
            nbinsy=50,
            orientation='h',
            opacity=0.6,
            histnorm='probability density'
        ),
        row=2, col=2
    )

    # Gaussian Fit
    fig.add_trace(
        go.Scatter(
            x=stats.norm.pdf(return_range, mean_return, std_return),
            y=return_range,
            mode='lines',
            name="Gaussian Fit",
            line=dict(color="#E0E0E0", width=4)
        ),
        row=2, col=2
    )

    # Confidence bounds on histogram
    fig.add_trace(
        go.Scatter(
            x=[0, hist_counts.max()],
            y=[upper_bound] * 2,
            mode='lines',
            name='Upper 95% Bound',
            line=dict(color="rgba(255,255,255,0.4)", dash="dot"),
            showlegend=False
        ),
        row=2, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[0, hist_counts.max()],
            y=[lower_bound] * 2,
            mode='lines',
            name='Lower 95% Bound',
            line=dict(color="rgba(255,255,255,0.4)", dash="dot"),
            fill='tonexty',
            fillcolor="rgba(255,255,255,0.05)",
            showlegend=False
        ),
        row=2, col=2
    )

    ######################
    ### Layout
    ######################

    fig.update_layout(
        template="plotly_dark",
        title=f"ARIMA Forecast Performance Analysis for {full_name}",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        title_font=dict(color=COLORS['primary']),
        showlegend=False
    )

    fig.update_yaxes(tickformat=".2%", row=2, col=1)
    fig.update_yaxes(tickformat=".2%", row=2, col=2)

    return dcc.Graph(
        id="arima-performance-plot",
        figure=fig,
        config={'responsive': True},
        style={'height': '100%', 'width': '100%'}
    )

def forecast_garch(log_returns, forecast_until):

    # GARCH model on mean + volatility (AR + GARCH)
    am = arch_model(log_returns, vol='Garch', p=1, q=1, mean='AR', lags=1, rescale=True)
    res = am.fit(disp="off")

    # Define forecast range
    last_date = log_returns.index[-1]
    target_date = pd.to_datetime(forecast_until)
    freq = pd.infer_freq(log_returns.index) or 'B'
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(1, unit='D'), end=target_date, freq=freq)
    n_periods = len(forecast_index)

    # Forecast
    forecasts = res.forecast(horizon=n_periods)
    mean_forecast = forecasts.mean.values[-1]
    variance_forecast = forecasts.variance.values[-1]
    std_forecast = np.sqrt(variance_forecast)

    # Create DataFrame with forecast and 95% CI
    forecast_df = pd.DataFrame({
        'Forecast': mean_forecast,
        'Lower CI': mean_forecast - 1.96 * std_forecast,
        'Upper CI': mean_forecast + 1.96 * std_forecast
    }, index=forecast_index)

    return forecast_df
