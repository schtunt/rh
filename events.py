import util
from util.numbers import F


class Event:
    ident = 0

    def __init__(self, stock):
        self.ident = Event.ident
        Event.ident += 1

        self.stock = stock

        self.connections = []
        self.timestamp = util.datetime.NaT

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


class StockSplitEvent(Event):
    def __init__(self, stock, date, multiplier, divisor):
        super().__init__(stock)

        self.timestamp = util.datetime.parse(date)
        self.multiplier = F(multiplier)
        self.divisor = F(divisor)

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
