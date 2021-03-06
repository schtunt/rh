import api
import stocks
import slurp
import models

import pandas as pd
import scipy.stats

import util
from util.numbers import F


class Account:
    def __init__(self, tickers=()):
        self._portfolio = {}

        self._transactions = slurp.transactions()

        portfolio_is_complete = False
        if bool(len(tickers) == 0):
            portfolio_is_complete = True
            tickers = tuple(api.symbols(remove='expired'))

        self._portfolio.update({
            ticker: stocks.Stock(self, ticker)
            for ticker in filter(api.whitelisted, tickers)
        })

        self._stocks = slurp.stocks(
            self._transactions,
            self._portfolio,
            portfolio_is_complete=portfolio_is_complete,
        )

        profile = api.rh.build_user_profile()
        self._equity = F(profile['equity'])
        self._extended_hours_equity = F(profile['extended_hours_equity'])
        self._cash = F(profile['cash'])
        self._dividends_total = F(profile['dividend_total'])

    def underlying(self, ticker):
        return self._portfolio[ticker]

    def __getitem__(self, ticker):
        return self.underlying(ticker)

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
                'y5c%', 'y2c%', 'y1c%', 'm6c%', 'm3c%', 'm1c%', 'd30c%', 'd5c%'
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


#def __getattr__(ticker: str) -> stocks.Stock:
#    _ticker = ticker.upper()
#    if _ticker not in api.symbols():
#        raise NameError("name `%s' is not defined, or a valid ticker symbol" % ticker)
#
#    return self._portfolio[ticker.upper()]
