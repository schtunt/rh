import api
import stocks
import slurp

import pandas as pd

import util
from util.numbers import D


PORTFOLIO = {}


def __getattr__(ticker: str) -> stocks.Stock:
    return PORTFOLIO[ticker.upper()]


#@util.singleton
class Account:
    def __init__(self, tickers=()):
        self._transactions = slurp.transactions()

        portfolio_is_complete = bool(len(tickers) == 0)

        PORTFOLIO.update({
            ticker: stocks.Stock(self, ticker)
            for ticker in (
                api.symbols() if portfolio_is_complete else tickers
            )
        })

        self._stocks = slurp.stocks(
            self._transactions,
            PORTFOLIO,
            portfolio_is_complete=portfolio_is_complete,
        )

        S = self._stocks
        T = self._transactions

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
        return PORTFOLIO[ticker]

    @property
    def transactions(self):
        return self._transactions.iterrows()

    @property
    def stocks(self):
        return self._stocks.iterrows()

    @property
    def portfolio(self):
        return PORTFOLIO.ITEMS()
