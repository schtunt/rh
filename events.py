import util
from util.numbers import D

from constants import ZERO as Z

class Event:
    ident = 0

    def __init__(self, stock):
        self.ident = Event.ident
        Event.ident += 1

        self.stock = stock

        self.connections = []

    @property
    def ticker(self):
        return self.stock.ticker

    @property
    def unsettled(self):
        return 0

    def date(self):
        return util.datetime.short(self.timestamp)

    def connect(self, lc):
        self.connections.append(lc)


class TransactionEvent(Event):
    def __init__(self, stock, dfrow, fulfillment_buy=None):
        super().__init__(stock)

        if fulfillment_buy is None:
            self.side = dfrow.side
            self._quantity = dfrow.quantity
            self._price = dfrow.average_price
            self.timestamp = dfrow.date
            self.otype = dfrow.order_type
        else:
            self.side = 'buy'
            self._quantity = fulfillment_buy
            self._price = Z
            self.timestamp = util.datetime.datetime(
                2020, 1, 1, 0, 0, tzinfo=util.datetime.timezone.utc
            )
            self.otype = 'free'


    def __repr__(self):
        price = self.price()
        qty = self.quantity()

        q = util.color.qty(qty)
        if len(self.connections) > 1:
            q = '%s (%s)' % (
                q, ', '.join([util.color.qty(lc.quantity()) for lc in self.connections])
            )

        rstr = '%s#%05d %-32s -- %s %s %s x %s = %s -> %sx' % (
            self.ticker,
            self.ident,
            self.timestamp,
            self.otype,
            self.side,
            q,
            util.color.mulla(price),
            util.color.mulla(price * qty),
            util.color.qty(self.stock.quantity(when=self.timestamp)),
        )

        return '<TransactionEvent %s>' % rstr

    def price(self, when=None):
        price = self._price
        if when is None: return price

        for splitter in self.stock.splits:
            if self.timestamp <= splitter.timestamp <= when:
                price = splitter.reverse(price)

        return price

    def quantity(self, when=None):
        quantity = self._quantity
        if when is None: return quantity

        q0 = quantity
        for splitter in self.stock.splits:
            if self.timestamp <= splitter.timestamp <= when:
                quantity = splitter.forward(quantity)

        return quantity

    def available(self, when=None):
        '''
        of the total number of shares associated with this transaction, how many have not yet
        been accounted for via a sell.  furthermore, the time of the question itself needs to be
        specified; for every stock split event that lies between this buy and the request date,
        the stock split has to be invoked.
        '''

        assert self.side == 'buy'

        # Get availability at time of purchase (when=None)
        available = self.quantity(when=when)

        for lc in self.connections:
            available -= lc.quantity(when=when)

        return available

    def settle(self, stock):
        signum = lambda side: {'buy': +1, 'sell':-1}[side]
        stock._quantity += signum(self.side) * self._quantity


class StockSplitEvent(Event):
    def __init__(self, stock, date, multiplier, divisor):
        super().__init__(stock)

        self.timestamp = util.datetime.parse(date)
        self.multiplier = D(multiplier)
        self.divisor = D(divisor)

    def __repr__(self):
        rstr = '%s#%05d %-32s -- stock-split %s:%s' % (
            self.ticker,
            self.ident,
            self.timestamp,
            util.color.qty0(self.multiplier),
            util.color.qty0(self.divisor),
        )

        return '<StockSplitEvent  %s>' % rstr

    def forward(self, qty):
        qty *= self.multiplier
        qty /= self.divisor
        return qty

    def reverse(self, qty):
        qty /= self.multiplier
        qty *= self.divisor
        return qty
