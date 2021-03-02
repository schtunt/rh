import util
from util.numbers import F

import api
import events


class Stock:
    def __init__(self, account, ticker):
        self.account = account
        self.ticker = ticker
        self.tockers = api.tockers4ticker(ticker)

        self._quantity = 0
        self._ledger = []

        self.lots = []

        self.pointer = 0
        self.events = []

        self.splits = []
        for ticker in self.tockers:
            self.splits.extend(
                events.StockSplitEvent(
                    stock=self,
                    date=ss['exDate'],
                    divisor=ss['fromFactor'],
                    multiplier=ss['toFactor'],
                ) for ss in api.splits(ticker)
            )
        self.splits.sort(key=lambda ss: ss.date)

        # StockSplits - These splits take effect on queries anytime the date of the question
        # and the date of the buy date lie on different sides of a split date.  That means, that
        # the buy event needs to know "who's asking", or more specifically, "when's asking?", in
        # order to answer meaningfully.
        self._pool = self.splits[:]

    def __getitem__(self, i):
        return self.events[i]

    def __repr__(self):
        return '<Stock %s - %s x %s>' % (
            self.ticker,
            util.color.qty(self._quantity),
            util.color.mulla(self.price),
        )

    @property
    def earnings(self):
        return api.earnings(self.ticker)

    @property
    def marketcap(self):
        return F(api.marketcap(self.ticker))

    @property
    def sector(self):
        return api.sector(self.ticker)

    @property
    def shoutstanding(self):
        return api.shares_outstanding(self.ticker)

    @property
    def intrades(self):
        return {
            t['transactionDate']: F(t['transactionShares'])
            for t in api.insider_transactions(self.ticker)
        }

    @property
    def stats(self):
        return api.stats(self.ticker)

    @property
    def beta(self):
        return F(api.beta(self.ticker))

    @property
    def news(self):
        return [
                   '. '.join((blb['preview_text'], blb['title'])) for blb in api.news(self.ticker, 'rh')
               ] + [
                   '. '.join([blb['headline'], blb['summary']]) for blb in api.news(self.ticker, 'iex')
               ]

    @property
    def ttm(self):
        return dict(
            eps=self.stats['ttmEPS'],
            div=self.stats['ttmDividendRate'],
        )

    @property
    def fundamentals(self):
        return self.account.fundamentals[self.ticker]

    @property
    def price(self):
        return F(api.price(self.ticker))

    @property
    def sells(self):
        return filter(lambda e: e.side == 'sell', self.transactions)

    @property
    def buys(self):
        return filter(lambda e: e.side == 'buy', self.transactions)

    @property
    def equity(self):
        return self._quantity * self.price

    @property
    def transactions(self):
        return filter(lambda e: type(e) is events.TransactionEvent, self.events)

    @property
    def recommendations(self):
        return api.recommendations(self.ticker)

    @property
    def target(self):
        return api.target(self.ticker)

    @property
    def quote(self):
        return api.quote(self.ticker)

    def quantity(self, when=None):
        raise NotImplementedError

    def traded(self, when=None):
        return sum(map(lambda buy: buy.quantity(when=when), self.buys))
