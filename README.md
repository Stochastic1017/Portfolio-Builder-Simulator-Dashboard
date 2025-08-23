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

### Forecast and Simulation

In general, for all four forecasting procedures, we generate $N$ ensembles up-to a future time-step $h$ and compute the mean and 95% confidence of portfolio log-returns (and subsequntly, the portfolio price). Using the weights, we can backtrack compute mean and 95% confidence in log-returns and prices for each of the $d$ tickers.

**For Auto-Regressive Integrated Moving Averages (ARIMA):**

For the forecasting procedures, users can choose one of three information criterions upon which to optimize model parameters.
Let $L$ be the maximized likelihood of the fitted model, i.e., the joint probability of observing the data under a given model, with parameters chosen to maximize that probability.
Let $p,q$ be AR and MA orders respectively. Lastly, let $d$ be the order of differencing needed to make data stationary.

1. Akaike Information Criterion (AIC) : $-2 \cdot \text{ln}(L) + 2 \cdot (p+q-1)$
2. Bayesiam Information Criterion (BIC) : $-2 \cdot \text{ln}(L) + \text{ln}(T-1) \cdot (p+q-1)$
3. LogLikelihood : $\text{ln}(L)$

Given chosen criterion, a grid-search is performed over $p \in \{0,1,2,3\}$, $d \in \{0,1,2\}$, $q \in \{0,1,2,3\}$, selects the model with the best score under the chosen criterion (AIC/BIC minimized, LL maximized) for portfolio log-returns.

The ARIMA(p, q) model is as follows, suppose $r_t$ is log-returns for the portfolio:
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

Just like ARIMA, users can choose one of three information criterions (AIC, BIC, LogLikelihood) upon which to optimize model parameters. Let $p,q$ be the number of lags for conditional variance and squared residuals respectively.

Given chosen criterion, a grid-search is performed over the models \{GARCH, EGARCH, GJR-GARCH\}, distributions \{Gaussian, Student-t, Skew-t, GED\}, and $p,q \in \{1,2,3\}$.

For each ensemble path $n \in \{1,\dots,N\}$ and forecast time-step $h$, we simulate log-returns as follows:

* For Standard GARCH(p,q):
```math
\hat{\sigma}^{2, (n)}_{t+h} = \bigg( \alpha_0 + \sum_{i=1}^{q} \alpha_i \; \epsilon_{t+h-i}^{2, (n)} + \sum_{j=1}^{p} \beta_j \; \hat{\sigma}_{t+h-j}^{2,(n)} \bigg)
```

* For Exponential GARCH(p,q):
```math
\text{ln} \hat{\sigma}^{2,(n)}_{t+h} = \omega + \sum_{i=1}^{q} \big( \alpha_i |z^{(n)}_{t+h-i}| + \gamma_i z^{(n)}_{t+h-i} \big) + \sum_{j=1}^{p} \beta_j \ln \hat{\sigma}^{2,(n)}_{t+h-j}
```

* For GJR-GARCH(p, q):
```math
\hat{\sigma}_{t+h}^{2,(n)} = \alpha_0 + \sum_{i=1}^{q} \left(\alpha_i \, (\epsilon_{t+h-i}^{(n)})^2 + \gamma_i (\epsilon_{t+h-i}^{(n)})^2 \, \mathbf{1}_{\{\epsilon_{t+h-i}^{(n)}<0\}} \right) + \sum_{j=1}^{p} \beta_j \, \hat{\sigma}_{t+h-j}^{2,(n)}
```

For all of the above, we can compute $N$ ensembles of log-returns as follows:
```math
\hat{r}^{(n)}_{t+h} = \bar{r}_{t+h-1} + \sigma^{(n)}_{t+h} \; z^{(n)}_{t+h} \quad \text{where} \; z^{(n)}_{t+h} \sim F
```
where, F is one of Gaussian, Student-t, Skew-t, GED distributions.

**For Euler–Maruyama (Gradient Boosted - Estimated Parameters):**

Unlike ARIMA and GARCH, we do not require information criterion information for this forecasting precedure. Instead, we use gradient boosting (with train/test/validation split) to estimate the parameters.

Assume portfolio log-returns follows a stochastic differential equation:
```math
dr_t = \underbrace{\mu(r_t, t)}_{\text{Drift Term}} \; dt + \underbrace{\sigma(r_t, t)}_{\text{Volatility Term}} \; dW_t
```

Gradient Boosted Regression is used to estimate drift and volatility terms as follows:
```math
\hat\mu(r_t, t) \approx \mathbb{E}[r_{t+1} | r_t, t], \quad 
\hat\sigma(r_t, t) \approx \mathbb{V}[r_{t+1} | r_t, t]
```

In practice:
- A gradient boosting regressor is trained on input features (past returns, lagged variables, time index) to predict the next return $r_{t+1}$.
- The predicted mean becomes $\hat{\mu}(r_t, t)$.
- The residual variance (squared error between predictions and observed returns) is used as an estimate of $\hat{\sigma}^2(r_t, t)$.

Then, by Euler-Maruyama we can generate $N$ ensembles at forecast step $t+h$ as:
```math
\hat{r}^{(n)}_{t+h} = \hat{r^{(n)}_{t+h-1}} + \hat\mu(\hat{r^{(n)}_{t+h-1}}, t+h-1) \cdot \Delta t + \hat\sigma(\hat{r^{(n)}_{t+h-1}}, t+h-1) \cdot \sqrt{\Delta t} \cdot z^{(n)}_{t+h}
```
where $z^{(n)}_{t+h} \sim N(0,1)$ and $\Delta t = 1$ day.

**For Long-Short Term Memory Models (LSTM):**

Let $L$ be the short-window over which we update parameters for the neural network model. Then, we initialize input sequence with last observed window as follows:
```math
(r_t, r_{t-1}, r_{t-2}, \dots, r_{t-L+1})
```

We model our log-returns as a non-linear sequencing program as follows:
```math
\hat{r_{t+1}} = f_{\theta}(r_t, r_{t-1}, r_{t-2}, \dots, r_{t-L+1})
```

The mapping can be learned using a 2-plus-2 layer LSTM neural network with the following architecture:
- cell state $c_t$ (long-term memory)
- hidden state $h_t$ (short-term memory / output)

The update equations are:
```math
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, r_t] + b_f) && \text{(forget gate)} \\
i_t &= \sigma(W_i [h_{t-1}, r_t] + b_i), \quad \tilde{c}_t = \tanh(W_c [h_{t-1}, r_t] + b_c) && \text{(input gate)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t && \text{(cell update)} \\
o_t &= \sigma(W_o [h_{t-1}, r_t] + b_o), \quad h_t = o_t \odot \tanh(c_t) && \text{(output gate)}
\end{aligned}
```

Finally, the one-step forecast is obtained from the hidden state added with random shock $\epsilon^{(n)}_{t+1}$:
```math
\hat{r}_{t+1} = W_y h_t + b_y + \epsilon^{(n)}_{t+1} =  f_{\theta}(r_t, r_{t-1}, r_{t-2}, \dots, r_{t-L+1}) + \epsilon^{(n)}_{t+1}
```

The above is repeated $N$ times to generate the required ensembles.

**Using ensembles to backcalculate portfolio pricing and per-ticker pricing:**

After all ensembles are generated (using any of the four aforementioned procedures), we aggregate the forecasted portfolio log-returns as follows:
```math
\bar{r}_{t+h} = \frac{1}{N} \sum_{n=1}^{N} \hat{r}^{(n)}_{t+h}, \quad CI_{95\%}(r_{t+h}) = \bigg[Q_{2.5\%}\bigg\{\hat{r}^{(n)}_{t+h}\bigg\}_{n=1}^{N}, Q_{97.5\%}\bigg\{\hat{r}^{(n)}_{t+h}\bigg\}_{n=1}^{N}\bigg]
```
where $Q_{p}$ is the p-th quantile from ensemble distribution.

Then, we re-construct all $N$ ensemble portfolio prices by taking exponential cumulative sums of portfolio log-returns:
```math
\hat{P}^{(n)}_{t+h} = P_t \cdot \text{exp}\bigg( \sum_{l=1}^{h} \hat{r}^{(n)}_{t+l} \bigg)
```

The ensemble prices then give us the mean estimate and 95% confidence as follows:
```math
\bar{P}_{t+h} = \frac{1}{N} \sum_{n=1}^{N} \hat{P}^{(n)}_{t+h}, \quad CI_{95\%}(P_{t+h}) = \bigg[Q_{2.5\%}\bigg\{\hat{P}^{(n)}_{t+h}\bigg\}_{n=1}^{N}, Q_{97.5\%}\bigg\{\hat{P}^{(n)}_{t+h}\bigg\}_{n=1}^{N}\bigg]
```

Thus, for each ticker corresponding to asset $A_j$ with weight $w_j$, we can compute mean estimate of asset prices and asset log-returns as:
```math
\bar{P}^{(A_j)}_{t+h} = \bar{P}_{t+h} * w_j, \quad \bar{r}^{(A_j)}_{t+h} = \bar{r}_{t+h} * w_j
```

Finally, the 95% confidence of asset prices and asset log-returns is:
```math
\begin{align}
CI_{95\%}(P^{(A_j)}_{t+h}) &= \bigg[Q_{2.5\%}\bigg\{\hat{P}^{(n)}_{t+h} \bigg\}_{n=1}^{N} \cdot w_j, \;\; Q_{97.5\%}\bigg\{\hat{P}^{(n)}_{t+h} \bigg\}_{n=1}^{N} \cdot w_j \bigg]\\
CI_{95\%}(r^{(A_j)}_{t+h}) &= \bigg[Q_{2.5\%}\bigg\{\hat{r}^{(n)}_{t+h} \bigg\}_{n=1}^{N} \cdot w_j, \;\; Q_{97.5\%}\bigg\{\hat{r}^{(n)}_{t+h} \bigg\}_{n=1}^{N} \cdot w_j \bigg]
\end{align}
```
