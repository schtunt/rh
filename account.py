import api
import stocks
import slurp

import pandas as pd

import util
from util.numbers import D

class Account:
    def __init__(self, tickers=()):
        self._transactions = slurp.transactions()

        portfolio_is_complete = bool(len(tickers) == 0)

        self._portfolio = {
            ticker: stocks.Stock(self, ticker)
            for ticker in (
                api.symbols() if portfolio_is_complete else tickers
            )
        }

        self._stocks = slurp.stocks(
            self._transactions,
            self._portfolio,
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
            attributes=('ticker',),
            column='trd0',
            chain=(
                lambda attrs: T[T['symbol'] == attrs[0]].date,
                lambda dates: min(dates) if len(dates) else pd.NaT,
            )
        )
        # }=-

    def get_stock(self, ticker):
        return self._portfolio[ticker]

    @property
    def transactions(self):
        return self._transactions.iterrows()

    @property
    def stocks(self):
        return self._stocks.iterrows()

    @property
    def portfolio(self):
        return self._portfolio.items()
