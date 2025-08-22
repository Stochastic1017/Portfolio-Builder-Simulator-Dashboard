# Statistical Market Analysis, Efficient Frontier and Forecasting

## Introduction

This open-source project is built for mathematically inclined data-driven market enthusiasts who want more than just charts, the app can help users do the following:

1. Explore any U.S. stock ticker — analyze news, past performance, and derive statistical summaries with clean visualizations (price trends, log-returns, Gaussian overlays). This utilizes the Polygon.io Stock API.

2. Engineer smarter portfolios — apply Modern Portfolio Theory to your own combination of chosen tickers, plot efficient frontiers, and instantly compare optimal strategies (max Sharpe, min risk, max diversification, equal weights).

3. Simulate ensembles to forecast prices and returns using Auto-Regressive Integrated Moving Averages (ARIMA), Generalized Autoregressive Conditional Heteroskedasticity (GARCH), Euler–Maruyama with Gradient Boosted parameters, or Long-Short Term Memory (LSTM) models, with 95% confidence bounds and performance tables for forecasts.

Whether you’re a casual investor curious about market dynamics, a data science student exploring finance, or a quantitative researcher testing portfolio strategies, this repo gives you the tools to analyze, optimize, and simulate — all in one workflow.

## Breakdown of the app

### Portfolio Exploration 

This page allows users to:

* Verify and explore any U.S. stock ticker via Polygon API, and maintain a custom wishlist/portfolio of verified tickers.

* Fetch company metadata, news articles (with sentiment tags), and financial details.

* Visualize historical performance:
  * Line chart of price trends over multiple ranges (1M, 3M, 6M, 1Y, 2Y).
  * Line chart of daily log-returns overlayed with 95% confidence intervals.
  * Histogram of returns overlaid with a Gaussian distribution with estimated mean and variance.

* Generate a statistical summary including mean, variance, skewness, kurtosis, hypothesis testing, and normality checks.

The file for the dash html layout can be found [`dash_app/pages/portfolio_exploration.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/pages/portfolio_exploration.py). 

The helper functions used to call Polygon API, validate tickers, create the metadata layout, create historic plots, summary tables, and range selectors can all be found: [`dash_app/helpers/portfolio_exploration.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/helpers/portfolio_exploration.py).

The callbacks for user interactivity within the app can be found [`dash_app/callbacks/portfolio_exploration.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/callbacks/portfolio_exploration.py).

### Portfolio Builder 

This page allows users to:

* Choose any subset of tickers from custom wishlist/portfolio of verified ticker from exploration page.

* Plot efficient frontier with various highlighted portfolios, over multiple ranges (1M, 3M, 6M, 1Y, 2Y). 

* Optimize portfolios to find (a.) Maximum Sharpe, (b.) Maximum Diversification, (c.) Minimum Risk, and (d.) Equal Weights.

* Generate allocation and weight summary, as well as implied returns for each portfolio given a budget input (in $).

The file for the dash html layout can be found [`dash_app/pages/portfolio_builder.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/pages/portfolio_builder.py). 

The helper functions used to call Polygon API, validate tickers, create the metadata layout, create historic plots, summary tables, and range selectors can all be found: [`dash_app/helpers/portfolio_builder.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/helpers/portfolio_builder.py).

The callbacks for user interactivity within the app can be found [`dash_app/callbacks/portfolio_builder.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/callbacks/portfolio_builder.py).

## Mathematics Overview of the key concepts

Consider the following definitions:

* $T$ - Chosen time-range in days (one among 1M, 3M, 6M, 1Y, and 2Y).
* $d$ - Number of subset tickers chosen.
* $A_j$ - Arbitrary ticker $j$, where $j \in \{1, \dots, d\} are all tickers in the subset.

### At a per-asset level:

For an arbitrary asset $A_j$ with prices $P_{j1}, P_{j2}, \dots, P_{jT}$, we define log-returns for the asset on day $i \in \{2,\dots,T\}$ (defined as $r_i$) as:
```math
r_{ji} = \text{log} \frac{P_{ji}}{P_{j(i-1)}} \quad \text{where} \; i \in \{2,\dots,T\}
```

Then, for the entire chosen time-range $T$, we define the log-returns vector for asset $A_j$ as: 
```math
\mathbf{r_j} = \begin{pmatrix} r_{j2} & r_{j3} & \dots & r_{jT} \end{pmatrix} \in \mathbb{R}^{T-1}
```

Then, we define expected (mean) log-returns for asset $A_j$ as:
```math
\hat{\mu_j} = \frac{1}{T-1} \sum_{i=2}^{T} r_{ji}
```

and risk (unbiased standard deviation) log-returns for asset $A_j$ as:
```math
\hat{\sigma_j} = \sqrt{\frac{1}{T-2} \sum_{i=2}^{T} \big(r_{ji} - \hat{\mu_j}\big)^2}
```

Finally, for any two assets $A_j$ and $A_k$, we define the covariance between their log-returns as:
```math
\hat{\sigma_{jk}} = \sqrt{\frac{1}{T-2} \sum_{t=2}^{T} \big(r_{jt} - \hat{\mu_j}\big) \big(r_{kt} - \hat{\mu_k}\big) }
```

### At a portfolio level:

We can consolidate all expected log-returns and risk for all assets $A_j$ where $j=1,\dots,d$ as follows:

Portfolio log-return mean vector (defined as $\mathbf{\hat\mu}$):
```math
\mathbf{\hat\mu} = \begin{pmatrix} \hat\mu_1 & \hat\mu_2 & \dots & \hat\mu_d \end{pmatrix} \in \mathbb{R}^{d}
```

Portfolio log-return volatility vector (defined as $\mathbf{\hat\sigma}$):
```math
\mathbf{\hat\sigma} = \begin{pmatrix} \hat\sigma_1 & \hat\sigma_2 & \dots & \hat\sigma_d \end{pmatrix} \in \mathbb{R}^{d}
```

Portfolio log-return covariance matrix (defined as $\mathbf{\hat\Sigma}$):
```math
\mathbf{\hat\Sigma} = \begin{pmatrix}
    \hat{\sigma_1}^2 & \hat{\sigma_{12}} & \dots & \hat{\sigma_{1T}}\\
    \hat{\sigma_{21}} & \hat{\sigma_{2}}^2 & \dots & \hat{\sigma_{2T}}\\
    \vdots & \vdots & \ddots & \vdots\\
    \hat{\sigma_{d1}} & \hat{\sigma_{d2}} & \dots & \hat{\sigma_{d}^2}
\end{pmatrix} \in \mathbb{R}^{d \times d}
```

### Using weights:

Let $\mathbf{w}$ be normalized weights that represents the percentage of budget to be invested in $d$ tickers.
```math
\mathbf{w} = \begin{pmatrix} w_1 & w_2 & \dots & w_d \end{pmatrix} \in \mathbb{R}^{d} \quad \text{where}\;\sum_{i=1}^{d} w_i = 1
```

Then, the expected (weighted mean) of log-returns for the portfolio is:
```math
\hat{\mu}_P = \mathbf{w}^T \mathbf{\hat\mu} = \sum_{i=1}^{d} w_i \hat\mu_i
```

and weighted average of asset volatilities (without covariance) to quantify diversification among assets:
```math
\hat{D}_P = \mathbf{w}^T \hat\sigma = \sum_{i=1}^{d} w_i \hat\sigma_i
```

Likewise, the risk (weighted standard deviation) including covariance of log-returns for the portfolio is:
```math
\hat{\sigma}_P = \sqrt{\mathbf{w}^T \mathbf{\hat\Sigma} \mathbf{w}} = \sqrt{\sum_{j=1}^{d} \sum_{k=1}^{d} w_j w_k \hat\sigma_{jk}}
```

### Efficient Frontier and Markowitz Portfolio Theory

All optimizations referred in this section were solved using Sequential Least Squares Programming Algorithm (SLSQP).

**For efficient frontier:**

The efficient frontier plot consists of all points whose weights $\mathbf{w}$ and desired expected return $R^*$ satisfy the following optimization problem:
```math
\min_{\mathbf{w}} \; \underbrace{\sqrt{\mathbf{w}^T \mathbf{\hat\Sigma} \mathbf{w}}}_{\hat\sigma_P} \quad \text{s.t.} \quad \underbrace{\mathbf{w}^T \mathbf{\hat\mu}}_{\hat\mu_P} = R^*, \; \| \mathbf{w} \|_1 = 1, \; \mathbf{w} > \mathbf{0}
```

**For Equal Weights:**

This is a trivial case, and we just plot the point where all tickers have the same weight, i.e.,
```math
\forall i \in \{1,\dots,d\}, \quad w_i = \frac{1}{d}
```

**For Minimum Variance:**

To minimize risk (or variance), we plot the point that satisfies the following optimization problem:
```math
\min_{\mathbf{w}} \; \underbrace{ \mathbf{w}^T \hat\Sigma \mathbf{w} }_{\sigma_P^2} \quad \text{s.t.} \| \mathbf{w} \|_1 = 1, \; \mathbf{w} > \mathbf{0}
```

**For Sharpe Ratio:**

To maximize Sharpe ratio, we plot the point that satisfies the following optimization problem:
```math
\max_{\mathbf{w}} \; \underbrace{ \bigg(\frac{\mathbf{w}^T \mathbf{\hat\mu} }{ \sqrt{\mathbf{w}^T \mathbf{\hat\Sigma} \mathbf{w}}}\bigg) }_{\hat\mu_P / \hat\sigma_P} \quad \text{s.t.} \| \mathbf{w} \|_1 = 1, \; \mathbf{w} > \mathbf{0}
```

**For Diversification Ratio:**

To maximize diversification ratio, we plot the point that satisfies the following optimization problem:
```math
\max_{\mathbf{w}} \; \underbrace{ \bigg(\frac{ \mathbf{w}^T \sigma }{ \sqrt{\mathbf{w}^T \mathbf{\hat\Sigma} \mathbf{w}}}\bigg) }_{\hat{D}_P / \hat\sigma_P} \quad \text{s.t.} \| \mathbf{w} \|_1 = 1, \; \mathbf{w} > \mathbf{0}
```

 
