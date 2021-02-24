import api
import stocks
import slurp
import models

import pandas as pd
import scipy.stats

import util
from util.numbers import D


class Account:
    def __init__(self, tickers=()):
        self._portfolio = {}

        self._transactions = slurp.transactions()

        portfolio_is_complete = False
        if bool(len(tickers) == 0):
            portfolio_is_complete = True
            tickers = tuple(api.symbols())

        self._portfolio.update({
            ticker: stocks.Stock(self, ticker)
            for ticker in filter(api.is_white, tickers)
        })

        self._stocks = slurp.stocks(
            self._transactions,
            self._portfolio,
            portfolio_is_complete=portfolio_is_complete,
        )

        profile = api.rh.build_user_profile()
        self._equity = D(profile['equity'])
        self._extended_hours_equity = D(profile['extended_hours_equity'])
        self._cash = D(profile['cash'])
        self._dividends_total = D(profile['dividend_total'])

    def get_stock(self, ticker):
        return self._portfolio[ticker]

    def sharpe(self):
        data = models.sharpe(holdings=api.holdings())
        util.debug.ddump(data, force=True)
        return data

    def momentum(self, ticker):
        '''
        Comparitive Momentum between this ticker, and all other tickers in portfolio
        '''
        ticker = ticker.upper()
        S = self._stocks
        return {
            change: scipy.stats.percentileofscore(
                S[change].to_numpy(),
                S.loc[S['ticker']==ticker, change].to_numpy(),
            ) for change in (
                'y5cp', 'y2cp', 'y1cp', 'm6cp', 'm3cp', 'm1cp', 'd30cp', 'd5cp'
            )
        }

    @property
    def transactions(self):
        return self._transactions

    @property
    def stocks(self):
        return self._stocks

    @property
    def portfolio(self):
        return self._portfolio.items()


def __getattr__(ticker: str) -> stocks.Stock:
    _ticker = ticker.upper()
    if _ticker not in api.symbols():
        raise NameError("name `%s' is not defined, or a valid ticker symbol" % ticker)

    return self._portfolio[ticker.upper()]
