# Statistical Market Analysis, Efficient Frontier and Forecasting

## Introduction

This open-source project is built for mathematically inclined data-driven market enthusiasts who want more than just charts, the app can help users do the following:

1. Explore any U.S. stock ticker — analyze news, past performance, and derive statistical summaries with clean visualizations (price trends, log-returns, Gaussian overlays). This utilizes the Polygon.io Stock API.

2. Engineer smarter portfolios — apply Modern Portfolio Theory to your own combination of chosen tickers, plot efficient frontiers, and instantly compare optimal strategies (max Sharpe, min risk, max diversification, equal weights).

3. Simulate ensembles to forecast prices and returns using Auto-Regressive Integrated Moving Averages (ARIMA), Generalized Autoregressive Conditional Heteroskedasticity (GARCH) with 95% confidence quantile bounds and performance tables for forecasts.

Whether you’re a casual investor curious about market dynamics, a data science student exploring finance, or a quantitative researcher testing portfolio strategies, this repo gives you the tools to analyze, optimize, and simulate — all in one workflow.

## Breakdown of the app

### Portfolio Exploration 

This page allows users to:

* Verify and explore any U.S. stock ticker via Polygon API, and maintain a custom wishlist/portfolio of verified tickers.

* Fetch company metadata, news articles (with sentiment tags), and financial details.

* Visualize historical performance of chosen asset:
  * Line chart of price trends over multiple ranges (1M, 3M, 6M, 1Y, 2Y).
  * Line chart of daily log-returns overlaid with 95% confidence intervals.
  * Histogram of returns overlaid with a Gaussian distribution with estimated mean and variance.

* Generate a statistical summary including mean, variance, skewness, kurtosis, hypothesis testing, and normality checks.

The file for the dash html layout can be found [`dash_app/pages/portfolio_exploration.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/pages/portfolio_exploration.py). 

The helper functions used to call Polygon API, validate tickers, create the metadata layout, create historic plots, summary tables, and range selectors can all be found: [`dash_app/helpers/portfolio_exploration.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/helpers/portfolio_exploration.py).

The callbacks for user interactivity within the app can be found [`dash_app/callbacks/portfolio_exploration.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/callbacks/portfolio_exploration.py).

Here is the demo of the page:

https://github.com/user-attachments/assets/b26fecee-3741-4ebd-9b23-0ec93b8ac40d

### Portfolio Builder 

This page allows users to:

* Choose any subset of tickers from custom wishlist/portfolio of verified ticker from exploration page.

* Plot efficient frontier with various highlighted portfolios, over multiple ranges (1M, 3M, 6M, 1Y, 2Y). 

* Optimize portfolios to find (a.) Maximum Sharpe, (b.) Maximum Diversification, (c.) Minimum Risk, and (d.) Equal Weights.

* Generate allocation and weight summary, as well as implied returns for each portfolio given a budget input (in $).

The file for the dash html layout can be found [`dash_app/pages/portfolio_builder.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/pages/portfolio_builder.py). 

The helper functions used to call Polygon API, validate tickers, create the metadata layout, create historic plots, summary tables, and range selectors can all be found: [`dash_app/helpers/portfolio_builder.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/helpers/portfolio_builder.py).

The callbacks for user interactivity within the app can be found [`dash_app/callbacks/portfolio_builder.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/callbacks/portfolio_builder.py).

### Portfolio Simulator 

This page allows users to:

* Visualize historical performance of the confirmed portfolio:
  * Line chart of price trends over multiple ranges (1M, 3M, 6M, 1Y, 2Y).
  * Line chart of daily log-returns overlaid with 95% confidence intervals.
  * Histogram of returns overlaid with a Gaussian distribution with estimated mean and variance.

* Choose a future date (upto 1 year from the latest date) for forecasting. Longer time implies more uncertainty and more compute.

* Choose number of ensembles to generate for forecasting (upto 100 ensembles). Larger ensemble implies more compute, but better estimates.

* Choose information criterion upon which to grid-search and find appropriate ARIMA/GARCH models.

* Choose confidence interval for quantiles which best represent variability (non-parametric) of log-returns and prices.

* Choose forecasting model between ARIMA or GARCH.

* Upon successful forecasting, a summary table is generated showing forecasts and confidence intervals for portfolio, as well as each ticker.

The file for the dash html layout can be found [`dash_app/pages/portfolio_simulator.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/pages/portfolio_simulator.py). 

The helper functions used to call Polygon API, validate tickers, create the metadata layout, create historic plots, summary tables, and range selectors can all be found: [`dash_app/helpers/portfolio_simulator.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/helpers/portfolio_simulator.py).

The callbacks for user interactivity within the app can be found [`dash_app/callbacks/portfolio_simulator.py`](https://github.com/Stochastic1017/Portfolio-Builder-Simulator-Dashboard/blob/main/dash_app/callbacks/portfolio_simulator.py).


## Mathematics Overview of the key concepts

Consider the following definitions:

* $T$ - Chosen time-range in days (one among 1M, 3M, 6M, 1Y, and 2Y).
* $d$ - Number of subset tickers chosen.
* $A_j$ - Arbitrary ticker $j$, where $j \in \{1, \dots, d\}$ are all tickers in the subset.

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
\hat{\sigma_{jk}} = \frac{1}{T-2} \sum_{t=2}^{T} \big(r_{jt} - \hat{\mu_j}\big) \big(r_{kt} - \hat{\mu_k}\big)
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

### Forecasting Models

In general, for all four forecasting procedures, we generate $N$ ensembles up-to a future time-step $h$ and compute the mean and confidence interval at level q of portfolio log-returns (and subsequently, the portfolio price). Using the weights, we can backtrack compute mean and confidence interval at level q in log-returns and prices for each of the $d$ tickers.

**For Auto-Regressive Integrated Moving Averages (ARIMA):**

For the forecasting procedures, users can choose one of three information criterions upon which to optimize model parameters.
Let $L$ be the maximized likelihood of the fitted model, i.e., the joint probability of observing the data under a given model, with parameters chosen to maximize that probability.
Let $p,q$ be AR and MA orders respectively, and $d$ be the order of differencing needed to make data stationary. Lastly, let $q$ be the chosen confidence interval. 

Given below are the Information Criterions available to the user:
1. Akaike Information Criterion (`AIC`) : $-2 \cdot \text{ln}(L) + 2 \cdot (p+q-1)$
2. Bayesian Information Criterion (`BIC`) : $-2 \cdot \text{ln}(L) + \text{ln}(T-1) \cdot (p+q-1)$
3. `LogLikelihood` : $\text{ln}(L)$

Given chosen criterion, a grid-search is performed over $p \in \{0,1,2,3\}$, $d \in \{0,1,2\}$, $q \in \{0,1,2,3\}$, selects the model with the best score under the chosen criterion (AIC/BIC minimized, LL maximized) for portfolio log-returns.

The `ARIMA(p, q, d)` model is as follows, suppose $r_t$ is log-returns for the portfolio:
```math
r_t = \bigg(r_0 + \sum_{i=1}^{p} \phi_i \; r_{t-i}\bigg) + \bigg(\epsilon_t + \sum_{j=1}^{q} \theta_j \; \epsilon_{t-j}\bigg)
```

where
* $r_0$ : Constant (Drift Term)
* $\phi_i$ : Auto-regressive coefficients
* $\theta_j$ : Moving-Averages Coefficients
* $\epsilon_t \sim N(0, \sigma^2)$ : White-noise shocks

For each ensemble path $n \in \{1,\dots,N\}$ and forecast time-step $h$, we simulate log-returns as follows:
```math
\hat{r}^{(n)}_{t+h} = \bigg( r_0 + \sum_{i=1}^{p} \phi_i \; \hat{r^{(n)}_{t+h-i}} \bigg) + \bigg( \epsilon^{(n)}_{t+h} + \sum_{j=1}^{q} \theta_j \; \epsilon^{(n)}_{t+h-j} \bigg)
```

**For Generalized Autoregressive Conditional Heteroskedasticity (GARCH):** 

Just like ARIMA, users can choose one of three information criterions (AIC, BIC, LogLikelihood) upon which to optimize model parameters. Let $p,q$ be the number of lags for conditional variance and squared residuals respectively, and $d$ be the order of differencing needed to make data stationary. 

Given chosen criterion, a grid-search is performed over the models \{GARCH, EGARCH, GJR-GARCH\}, distributions \{Gaussian, Student-t, Skew-t, GED\}, and $p,q \in \{1,2,3\}$.

For each ensemble path $n \in \{1,\dots,N\}$ and forecast time-step $h$, we simulate log-returns as follows:

* For `Standard GARCH(p, q, d)`:
```math
\hat{\sigma}^{2, (n)}_{t+h} = \bigg( \alpha_0 + \sum_{i=1}^{q} \alpha_i \; \epsilon_{t+h-i}^{2, (n)} + \sum_{j=1}^{p} \beta_j \; \hat{\sigma}_{t+h-j}^{2,(n)} \bigg)
```

* For `Exponential GARCH(p, q, d)`:
```math
\text{ln} \hat{\sigma}^{2,(n)}_{t+h} = \omega + \sum_{i=1}^{q} \big( \alpha_i |z^{(n)}_{t+h-i}| + \gamma_i z^{(n)}_{t+h-i} \big) + \sum_{j=1}^{p} \beta_j \ln \hat{\sigma}^{2,(n)}_{t+h-j}
```

* For `GJR-GARCH(p, q, d)`:
```math
\hat{\sigma}_{t+h}^{2,(n)} = \alpha_0 + \sum_{i=1}^{q} \left(\alpha_i \, (\epsilon_{t+h-i}^{(n)})^2 + \gamma_i (\epsilon_{t+h-i}^{(n)})^2 \, \mathbf{1}_{\{\epsilon_{t+h-i}^{(n)}<0\}} \right) + \sum_{j=1}^{p} \beta_j \, \hat{\sigma}_{t+h-j}^{2,(n)}
```

For all of the above, we can compute $N$ ensembles of log-returns as follows:
```math
\hat{r}^{(n)}_{t+h} = \bar{r}_{t+h-1} + \sigma^{(n)}_{t+h} \; z^{(n)}_{t+h} \quad \text{where} \; z^{(n)}_{t+h} \sim F
```
where, F is one of Gaussian, Student-t, Skew-t, GED distributions.

### Simulation Procedure:

Given all required inputs from the user, the portfolio log-returns and prices are forecasted in two key steps.

**Per-asset Ensemble Forecasting:** 

For each asset ticker $A_j$ in the selected portfolio, where $j \in \{1,\dots,d\}$, we do the following:
* The chosen model with inputs are fitted on historical log-returns of asset.
* We generate $N$ ensemble paths of future log-returns, for each $n \in \{1,\dots,N\}$, we have $\hat{r}^{(n)}_{t+h}$.
* From the $N$ ensemble log-return paths, we compute the following statistics. let $q$ be the chosen confidence interval:
```math
\text{median}(r_{A_j, t+h}) = Q_{0.5} \bigg\{\hat{r}^{(n)}_{A_j, t+h}\bigg\}_{n=1}^{N}, \quad CI_{q}(r_{A_j, t+h}) = \bigg[Q_{(1-q)/2}\bigg\{\hat{r}^{(n)}_{A_j, t+h} \bigg\}_{n=1}^{N}, \;\; Q_{(1+q)/2}\bigg\{\hat{r}^{(n)}_{A_j, t+h} \bigg\}_{n=1}^{N} \bigg]
```
*  Asset prices are then forecasted using the ensembles by taking exponentiating cumulatively summing log-return forecasts:
```math
\hat{P}^{(n)}_{A_j, t+h} = P_t \cdot \text{exp}\bigg(\sum_{i=1}^{h} \hat{r}^{(n)}_{A_j, t+i}\bigg)
```
* From the $N$ ensemble price paths, we compute the same statistics as before:
```math
\text{median}(P_{A_j, t+h}) = Q_{0.5} \bigg\{\hat{P}^{(n)}_{A_j, t+h}\bigg\}_{n=1}^{N}, \quad CI_{q}(P_{A_j, t+h}) = \bigg[Q_{(1-q)/2}\bigg\{\hat{P}^{(n)}_{A_j, t+h} \bigg\}_{n=1}^{N}, \;\; Q_{(1+q)/2}\bigg\{\hat{P}^{(n)}_{A_j, t+h} \bigg\}_{n=1}^{N} \bigg]
```

**Portfolio Aggregation:**

* Each asset $A_j$ price forecast is then weighted by portfolio weights $w_j$ and summed together to get the portfolio price, i.e., for price $\hat{P}^{(n)}_{A_j, t+h}$, we have:
```math
\hat{P}^{(n)}_{\text{port}, t+h} = \sum_{j=1}^{d} w_j \cdot P^{(n)}_{A_j, t+h}
```
* Log-Returns are then calculated as before:
```math
 \hat{r}^{(n)}_{\text{port}, t+h} = \text{log} \frac{\hat{P}^{(n)}_{\text{port}, t+h}}{\hat{P}^{(n)}_{\text{port}, t+h-1}}
```
* Same statistics as before are computed:
```math
\text{median}(P_{\text{port}, t+h}) = Q_{0.5} \bigg\{\hat{P}^{(n)}_{\text{port}, t+h}\bigg\}_{n=1}^{N}, \quad CI_{q}(P_{\text{port}, t+h}) = \bigg[Q_{(1-q)/2}\bigg\{\hat{P}^{(n)}_{\text{port}, t+h} \bigg\}_{n=1}^{N}, \;\; Q_{(1+q)/2}\bigg\{\hat{P}^{(n)}_{\text{port}, t+h} \bigg\}_{n=1}^{N} \bigg]
```

### Issues and Assumptions

A key assumption made by the above procedure, that often does not hold true in real-world circumstances, is the idea that ticker movements are independent. This assumption simplifies portfolio aggregation and allows us to weight and sum each individual ticker forecast into one portfolio forecast. Assets are often correlated with each other, thus the independence assumption understates portfolio risk and confidence intervals are too-narrow as diversification is overly optimistic.

There are some ways to fix the independence assumptions, these are not implemented in the application:
1. Correlated $\epsilon$ draws: Estimate the historical correlation matrix of ticker log-returns. During ensemble simulation, draw **correlated** normal innovations (via Cholesky decomposition) instead of independent ones, and feed them into each asset’s ARIMA/GARCH simulator.
2. Multivariate Approaches: Models like ARIMAX, DCC-GARCH, RNN, LSTM take into account conditional correlations and models cross-asset correlations as well. These require more compute but are theoretically sound.
3. Copula approaches: Joint dependencies among assets can be modelled using Copulas model to sample correlated shocks.
