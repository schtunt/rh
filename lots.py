import events
import util


class Lot:
    def __init__(self, stock, sell):
        """
        Every sell (transaction event) will have a corresponding realized Lot,
        and there's one additional active (unrealized) Lot per ticker, if any
        non-zero number of shares are still held for that stock beyond the last
        sale or first investment in the stock.
        """
        self.type = 'realized'
        self.sell, self.qty = sell, sell.quantity()
        self.buys = []

        required = self.qty
        while required > 0:
            # Bad data from robinhood; workaround
            if stock.pointer == len(stock.events):
                stock.events.append(
                    events.TransactionEvent(
                        stock,
                        required,
                        0.00,
                        'buy',
                        str(util.datetime.datetime(2020, 1, 1, 0, 0, tzinfo=util.datetime.timezone.utc)),
                        'FREE',
                    )
                )

            event = stock.events[stock.pointer]
            if type(event) is events.TransactionEvent:
                if event.side == 'buy':
                    available = event.available()
                    if available <= 0:
                        stock.pointer += 1
                        continue

                    lc = LotConnector(stock=stock, sell=self.sell, buy=event, requesting=required)
                    required -= lc.quantity()
                    self.buys.append(lc)
                else:
                    stock.pointer += 1
            elif type(event) is events.StockSplitEvent:
                stock.pointer += 1
            else:
                raise

    def bought(self):
        return {
            'qty': sum([e.quantity() for e in self.events if type(e) is events.TransactionEvent]),
            'value': sum([e.quantity() * e.price() for e in self.events if type(e) is events.TransactionEvent]),
        }

    def sold(self):
        return {
            'qty': self.sell.quantity(),
            'value': self.sell.quantity() * self.sell.price(),
        }

    @property
    def washsale_exempt(self):
        return True or False

    def costbasis(self, when=None):
        costbasis = {
            'short': {'qty': 0, 'value': 0},
            'long': {'qty': 0, 'value': 0},
        }

        for lc in self.buys:
            costbasis[lc.term]['qty'] += lc.quantity(when=when)
            costbasis[lc.term]['value'] += lc.costbasis(when=when)

        # util.output.ddump(costbasis, force=True)

        return costbasis


class LotConnector:
    def __init__(self, stock, sell, buy, requesting):
        self.stock = stock
        self.sell = sell
        self.buy = buy

        self._quantity = min(requesting, buy.available())

        # short-term or long-term sale (was the buy held over a year)
        self.term = util.finance.conterm(self.buy.timestamp, self.sell.timestamp)

        self.buy.connect(self)
        self.sell.connect(self)

    def quantity(self, when=None):
        quantity = self._quantity
        if when is None: return quantity

        for split in self.stock.splits:
            if self.timestamp <= split.timestamp <= when:
                quantity = split.forward(quantity)

        return quantity

    def costbasis(self, when=None):
        return (self.sell.price(when=when) - self.buy.price(when=when)) * self.quantity(when=when)

    @property
    def timestamp(self):
        return self.sell.timestamp
