import api
import stocks
import slurp

class Account:
    def __init__(self):
        print("A. Create Stock objects for all tickers in Robinhood account")
        self._portfolio = {
            ticker: stocks.Stock(self, ticker)
            for ticker in api.symbols()
        }

        print("B. Slurp CSV and generate Transactions DataFrame")
        self._transactions = slurp.transactions()

        print("C. Embelish DataFrame with TransactionEvent ledger columns")
        slurp.embelish_transactions(self._transactions, self._portfolio)

        print("D. Slurp the CSV")
        self._stocks = slurp.stocks(self._transactions, self._portfolio)

    def get_stock(self, ticker):
        return self._portfolio[ticker]

    @property
    def transactions(self):
        return self._transactions.iterrows()

    @property
    def stocks(self):
        return self._stocks.iterrows()
