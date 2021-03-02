import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
import pypfopt as ppo

from progress.bar import ShadyBar

import slurp
import util
from util.numbers import F

ANNUAL_TRADING_DAYS = F(365*5/7 -6 -3*5/7)

def risk_free_rate_of_return():
    '''
    TODO: Make this dynamic
    '''
    return F(0.02)


def treynor(ticker, beta):
    '''
    Treynor Ratio for a single stock
    '''
    risk_free_ror = risk_free_rate_of_return()

    adj_close_price = slurp.stock_historic_prices(ticker)['Adj Close']
    daily_returns = adj_close_price.pct_change()
    returns_mean = F(np.mean(daily_returns))
    variance = daily_returns.cov(daily_returns)   # AKA ~np.var(daily_returns)
    volatility = beta                             # AKA correlation to S&P500/Other Index, Risk

    sharpe = util.numbers.NaN
    if volatility:
        sharpe = (ANNUAL_TRADING_DAYS * returns_mean - risk_free_ror) / volatility

    return sharpe


def sharpe(ticker=None, holdings=None):
    if (ticker, holdings).count(None) != 1:
        raise RuntimeError("Supply either `ticker' or `holdings'")

    if ticker is not None:
        return _sharpe_on_ticker(ticker)

    if holdings is not None:
        return _sharpe_on_holdings(holdings)


def _sharpe_on_ticker(ticker):
    '''
    Sharpe Ratio for a single stock
    '''
    risk_free_ror = risk_free_rate_of_return()

    adj_close_price = slurp.stock_historic_prices(ticker)['Adj Close']
    daily_returns = adj_close_price.pct_change()
    returns_mean = F(np.mean(daily_returns))
    variance = daily_returns.cov(daily_returns)   # AKA ~np.var(daily_returns)
    volatility = F(np.sqrt(variance))             # AKA Standard Deviation, AKA Risk

    sharpe = util.numbers.NaN
    if volatility:
        sharpe = (ANNUAL_TRADING_DAYS * returns_mean - risk_free_ror) / volatility

    return sharpe


def _sharpe_on_holdings(holdings):
    '''
    holdings: api.rh.build_holdings()
    '''
    df = pd.DataFrame(index=None)
    equities = {}
    with ShadyBar('%32s' % 'Pulling Historic Quote Prices', max=len(holdings)) as bar:
        for ticker, datum in holdings.items():
            df[ticker] = slurp.stock_historic_prices(ticker)['Adj Close']
            equities[ticker] = F(datum['equity'])
            bar.next()

    # Calculate the current weight of each stock in the portfolio
    equity = sum(equities.values())
    weights = np.array([equities[ticker]/equity  for ticker in equities.keys()], dtype=np.float)

    # Calculate historic returns (percentage change close-to-close)
    returns = df.pct_change()

    # Number of Trading Days per Year (roughly)
    annual_trading_days = int(365*5/7 -6 -3*5/7)

    # Calculate the Annual Covariance
    # - The diagonal of this matrix is the variance (sqrt of the variance is volatility)
    # - Every other cell is the covariance (between the two non-identical tickes)
    annual_covariance = returns.cov() * annual_trading_days

    # Calculate the Portfolio Variance
    annual_variance = np.dot(weights.T, np.dot(annual_covariance, weights))

    # Calculate the Portfolio Volatility
    annual_volatility = np.sqrt(annual_variance)

    # Calculate the Simple Annual Return
    simple_annual_return = np.sum(returns.mean() * weights * annual_trading_days)

    # Measure the Portfolio
    response = dict(
        status_quo=dict(
            expected_annual_return=simple_annual_return,
            annual_volatility=annual_volatility,
            annual_variance=annual_variance,
        ),
        efficient_frontier=dict()
    )


    # Improve the Portfolio
    # Calculate expected returns and annualized sample covariance matrix of asset returns
    mu = ppo.expected_returns.mean_historical_return(df)
    S = ppo.risk_models.sample_cov(df)

    # Optimize for max sharpe ratio (k) - measures the performance of the porfolio compared
    # to risk-free investments like bonds or treasuries
    ef = EfficientFrontier(mu, S)
    reweighted = ef.max_sharpe()
    cleanweighted = ef.clean_weights()

    response['efficient_frontier']['portfolio'] = {
        ticker: pct for ticker, pct in cleanweighted.items() if pct > 0
    }

    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
    risk_free_ror=expected_annual_return-sharpe_ratio*annual_volatility
    response['efficient_frontier']['expectation'] = dict(
        expected_annual_return=expected_annual_return,
        annual_volatility=annual_volatility,
        sharpe_ratio=sharpe_ratio,
        risk_free_ror=risk_free_ror,
    )

    response['status_quo']['risk_free_rate'] = risk_free_ror
    response['status_quo']['sharpe_ratio'] = (
        simple_annual_return-risk_free_ror
    )/annual_volatility

    return response
