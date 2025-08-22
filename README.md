# Statistical Market Analysis, Efficient Frontier and Forecasting

## Introduction

This open-source project is built for mathematically inclined data-driven market enthusiasts who want more than just charts, the app can help users do the following:

1. Explore any U.S. stock ticker — analyze news, past performance, and derive statistical summaries with clean visualizations (price trends, log-returns, Gaussian overlays). This utilizes the Polygon.io Stock API.

2. Engineer smarter portfolios — apply Modern Portfolio Theory to your own combination of chosen tickers, plot efficient frontiers, and instantly compare optimal strategies (max Sharpe, min risk, max diversification, equal weights).

3. Simulate ensembles to forecast prices and returns using Auto-Regressive Integrated Moving Averages (ARIMA), Generalized Autoregressive Conditional Heteroskedasticity (GARCH), Euler–Maruyama with Gradient Boosted parameters, or Long-Short Term Memory (LSTM) models, with 95% confidence bounds and performance tables for forecasts.

Whether you’re a casual investor curious about market dynamics, a data science student exploring finance, or a quantitative researcher testing portfolio strategies, this repo gives you the tools to analyze, optimize, and simulate — all in one workflow.

## Portfolio Exploration 

This page allows users to:

* Verify and explore any U.S. stock ticker via Polygon API.

* Fetch company metadata, news articles (with sentiment tags), and financial details.

* Visualize historical performance:
  * Line chart of price trends over multiple ranges (1M, 3M, 6M, 1Y, 2Y).
  * Line chart of daily log-returns overlayed with 95% confidence intervals.
  * Histogram of returns overlaid with a Gaussian distribution with estimated mean and variance.

* Generate a statistical summary including mean, variance, skewness, kurtosis, hypothesis testing, and normality checks.

The file for the dash html layout can be found [`dash_app/pages/portfolio_exploration.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/pages/portfolio_exploration.py). 

The helper functions used to call Polygon API, validate tickers, create the metadata layout, create historic plots, summary tables, and range selectors can all be found: [`dash_app/helpers/portfolio_exploration.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/helpers/portfolio_exploration.py).

The callbacks for user interactivity within the app can be found [`dash_app/callbacks/portfolio_exploration.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/tree/main/dash_app/callbacks).
