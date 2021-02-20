import api
import stocks
import slurp

import pandas as pd
import scipy.stats

import util
from util.numbers import D


class Account:
    PORTFOLIO = {}

    def __init__(self, tickers=()):
        self._transactions = slurp.transactions()

        portfolio_is_complete = bool(len(tickers) == 0)

        Account.PORTFOLIO.update({
            ticker: stocks.Stock(self, ticker)
            for ticker in (
                api.symbols() if portfolio_is_complete else tickers
            ) if not api.is_black(ticker)
        })

        self._stocks = slurp.stocks(
            self._transactions,
            Account.PORTFOLIO,
            portfolio_is_complete=portfolio_is_complete,
        )
        #self._stocks.reset_index(inplace=True)
        #self._stocks.set_index('ticker', inplace=True)

        T = self._transactions
        S = self._stocks

        # Embelishments -={
        slurp.embelish(
            obj=S,
            attributes=('d200ma', 'd50ma', 'price',),
            column='ma',
            chain=(util.numbers.growth_score, D),
        )

        slurp.embelish(
            obj=S,
            attributes=('ticker',),
            column='trd0',
            chain=(
                lambda attrs: T[T['symbol'] == attrs[0]].date,
                lambda dates: min(dates) if len(dates) else pd.NaT,
            )
        )

        slurp.embelish(
            obj=S,
            attributes=('price', 'pcp'),
            column='change',
            chain=(
                lambda attrs: 100 * (attrs[0] / attrs[1] - 1),
            )
        )
        # }=-

    def get_stock(self, ticker):
        return Account.PORTFOLIO[ticker]

    def momentum(self, ticker):
        S = self._stocks
        return {
            change: scipy.stats.percentileofscore(
                pd.to_numeric(S[change]),
                pd.to_numeric(list(S.loc[S['ticker']==ticker, change])),
            ) for change in (
                'y5cp', 'y2cp', 'y1cp', 'm6cp', 'm3cp', 'm1cp', 'd30cp', 'd5cp'
            )
        }

    @property
    def transactions(self):
        return self._transactions.iterrows()

    @property
    def stocks(self):
        return self._stocks.iterrows()

    @property
    def portfolio(self):
        return Account.PORTFOLIO.items()


def __getattr__(ticker: str) -> stocks.Stock:
    return Account.PORTFOLIO[ticker.upper()]
