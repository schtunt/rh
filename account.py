import api
import stocks
import slurp

import pandas as pd
import scipy.stats
import statistics

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
            chain=(
                lambda data: data.values(),
                lambda data: util.numbers.growth_score(data),
            ),
        )

        slurp.embelish(
            obj=S,
            attributes=('ticker',),
            column='trd0',
            chain=(
                lambda data: T[T['symbol'] == data['ticker']].date,
                lambda dates: min(dates) if len(dates) else pd.NaT,
            )
        )

        slurp.embelish(
            obj=S,
            attributes=('price', 'pcp'),
            column='change',
            chain=(
                lambda data: 100 * (data['price'] / data['pcp'] - 1),
            )
        )

        slurp.embelish(
            obj=S,
            attributes=('ticker', 'y5cp', 'y2cp', 'y1cp', 'm6cp', 'm3cp', 'm1cp', 'd30cp', 'd5cp'),
            column='momentum',
            chain=(
                lambda data: { data.pop('ticker'): data },
                lambda data: [
                    scipy.stats.percentileofscore(
                        pd.to_numeric(S[period]),
                        pd.to_numeric(percentile),
                    ) for ticker, percentiles in data.items()
                    for period, percentile in percentiles.items()
                ],
                statistics.mean,
            )
        )

        # }=-

    def get_stock(self, ticker):
        return Account.PORTFOLIO[ticker]

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
        return Account.PORTFOLIO.items()


def __getattr__(ticker: str) -> stocks.Stock:
    _ticker = ticker.upper()
    if _ticker not in api.symbols():
        raise NameError("name `%s' is not defined, or a valid ticker symbol" % ticker)

    return Account.PORTFOLIO[ticker.upper()]
