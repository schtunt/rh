import constants
from constants import ZERO as Z

import util
from util import dec as D


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

    def connect(self, lc):
        self.connections.append(lc)


class TransactionEvent(Event):
    def __init__(self, stock, qty, price, side, timestamp, otype):
        super().__init__(stock)

        self.side = side
        self._quantity = D(qty)
        self._price = D(price)
        self.timestamp = util.datetime.dtp.parse(timestamp)
        self.otype = otype

    def price(self, when=None):
        price = self._price
        if not when: return price

        for splitter in self.stock.splitters:
            if self.timestamp <= splitter.timestamp <= when:
                price = splitter.reverse(price)

        return price

    def quantity(self, when=None):
        quantity = self._quantity
        if not when: return quantity

        for splitter in self.stock.splitters:
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


    #def __repr__(self):
        #return str((self.ticker, self.side, self.available(), self.quantity(), self.price(), self.otype, len(self.connections)))
        #q = util.color.qty(self.qty)
        #if len(self.ties) > 1:
        #    q = '%s (%s)' % (q, ', '.join([tie.portion(self) for tie in self.ties]))

        #rstr = '#%05d %s %s %s %s (cg=%s): %s x %s = %s @ %s' % (
        #    self.ident,
        #    self.ticker,
        #    util.color.qty(self.running),
        #    self.otype,
        #    self.side,
        #    self.term,
        #    q,
        #    util.color.mulla(self.price),
        #    util.color.mulla(self.price * self.qty),
        #    self.timestamp,
        #)

        #return '<TransactionEvent %s>' % rstr


class StockSplitEvent(Event):
    def __init__(self, stock, date, multiplier, divisor):
        super().__init__(stock)

        self.timestamp = util.datetime.parse(date)
        self.multiplier = D(multiplier)
        self.divisor = D(divisor)

    def __repr__(self):
        rstr = '#%05d %s %s stock-split %s-for-%s @ %s' % (
            self.ident,
            self.ticker,
            -1,
            util.color.qty(self.multiplier),
            util.color.qty(self.divisor),
            self.timestamp,
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
