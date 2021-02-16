import api
import stocks
import slurp

import util
from util.numbers import dec as D

class Account:
    def __init__(self):
        print("A. Create Stock objects for all tickers in Robinhood account")
        self._portfolio = {
            ticker: stocks.Stock(self, ticker)
            for ticker in api.symbols()
        }

        print("B. Slurp CSV and generate Transactions DataFrame")
        self._transactions = slurp.transactions()

        print("C. Embelish I: Slurp the APIs and embelish the TransactionEvents")
        slurp.embelish_transactions(self._transactions, self._portfolio)

        print("D. Embelish II: Slurp the APIs and embelish the Stocks DataFrame")
        self._stocks = slurp.stocks(self._transactions, self._portfolio)

        print("E. Embelish III: Additional ad-hoc embelishments, if any")
        slurp.embelish(
            obj=self._stocks,
            attributes=('d200ma', 'd50ma', 'price'),
            column='ma',
            chain=(util.numbers.growth_score, D)
        )

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
