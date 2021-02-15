from collections import defaultdict

import api
import constants
import util
import events
import stocks
import readers

from util.numbers import dec as D
from constants import ZERO as Z


class Account:
    def __init__(self):
        self.data = None
        self._portfolio = {}

        self.iex_api_key = None
        self.finnhub_api_key = None
        self.polygon_api_key = None
        self.monkeylearn_api = None
        self.alpha_vantage_api = None

        self.stockReader = readers.StockReader(api.download('stocks'))

    @property
    def tickers(self):
        return api.tickers()

    def get_stock(self, ticker):
        ticker = ticker.upper()
        if ticker not in self._portfolio:
            self._portfolio[ticker] = stocks.Stock(self, ticker)

        return self._portfolio[ticker]

    @property
    def stocks(self):
        return self._portfolio

    @property
    def holdings(self):
        return defaultdict(dict, api.holdings())

    def slurp(self):
        transactions = ((stock.ticker, stock.parameters) for stock in self.stockReader)
        for ticker, parameters in transactions:
            stock = self.get_stock(ticker)
            transaction = events.TransactionEvent(stock, *parameters)
            stock.push(transaction)

        self.data = self.holdings

        now = util.datetime.now()
        for ticker, stock in self.stocks.items():
            self.data[ticker].update({
                'realized': stock.costbasis(realized=True, when=now),
                'unrealized': stock.costbasis(realized=False, when=now),
            })

        positions = self.positions
        dividends = api.dividends()
        prices = api.prices()
        fundamentals = api.fundamentals()

        for ticker, datum in self.data.items():
            fifo = self.get_stock(ticker)

            datum['ticker'] = ticker
            datum['premium_collected'] = positions['premiums'][ticker]
            datum['dividends_collected'] = sum(D(div['amount']) for div in dividends[ticker])

            if ticker not in fundamentals:
                fundamentals[ticker] = api.fundamental(ticker)
            datum['pe_ratio'] = D(fundamentals[ticker]['pe_ratio'])
            datum['pb_ratio'] = D(fundamentals[ticker]['pb_ratio'])

            datum['collateral'] = positions['collateral'][ticker]

            opened = '\n'.join(positions['opened'].get(ticker, []))
            closed = '\n'.join(positions['closed'].get(ticker, []))
            datum['expiry'] = positions['next_expiry'].get(ticker, None)

            datum['ttl'] = D(
                999 if datum['expiry'] is None else (
                        datum['expiry'] - util.datetime.now()
                ).days
            )

            if opened and closed:
                datum['activities'] = ('\n%s\n' % ('─' * 32)).join((opened, closed))
            elif opened:
                datum['activities'] = opened
            elif closed:
                datum['activities'] = closed
            else:
                collateral = D(datum['collateral']['call'])
                optionable = D(datum.get('quantity', 0)) - collateral

                datum['activities'] = (
                    'N/A' if optionable < 100 else '%sx100 available' % (
                        util.color.qty0(optionable // 100)
                    )
                )

            datum['marketcap'] = fifo.marketcap

            if ticker not in prices:
                prices[ticker] = api.price(ticker)
            datum['price'] = D(prices[ticker])

            datum['50dma'] = D(fifo.stats['day50MovingAvg'])
            datum['200dma'] = D(fifo.stats['day200MovingAvg'])


    @property
    def positions(self):
        next_expiry = {}
        data = api.positions('options', 'all')
        collateral = defaultdict(lambda: {'put': Z, 'call': Z})
        opened = defaultdict(list)
        for option in [o for o in data if D(o['quantity']) != Z]:
            ticker = option['chain_symbol']

            uri = option['option']
            instrument = api.instrument(uri)

            premium = Z
            if instrument['state'] != 'queued':
                premium -= D(option['quantity']) * D(option['average_price'])

            itype = instrument['type']
            opened[ticker].append("%s %s %s x%s P=%s K=%s X=%s" % (
                instrument['state'],
                option['type'],
                itype,
                util.color.qty(option['quantity'], Z),
                util.color.mulla(premium),
                util.color.mulla(instrument['strike_price']),
                instrument['expiration_date'],
            ))
            expiry = util.datetime.parse(instrument['expiration_date'])
            if next_expiry.get(ticker) is None:
                next_expiry[ticker] = expiry
            else:
                next_expiry[ticker] = min(next_expiry[ticker], expiry)

            collateral[ticker][itype] += 100 * D(dict(
                put=instrument['strike_price'],
                call=option['quantity']
            )[itype])

        closed = defaultdict(list)
        premiums = defaultdict(lambda: Z)
        data = api.orders('options', 'all')
        for option in [o for o in data if o['state'] not in ('cancelled')]:
            ticker = option['chain_symbol']

            if ticker in constants.DEBUG:
                util.output.ddump(option, title='orders.options:all')

            strategies = []
            o, c = option['opening_strategy'], option['closing_strategy']
            if o:
                tokens = o.split('_')
                strategies.append('o[%s]' % ' '.join(tokens))
            if c:
                tokens = c.split('_')
                strategies.append('c[%s]' % ' '.join(tokens))

            legs = []
            premium = Z
            for leg in option['legs']:
                uri = leg['option']
                instrument = api.instrument(uri)

                if ticker in constants.DEBUG:
                    util.output.ddump(instrument, title=f'stocks:instrument({uri})')

                legs.append('%s to %s K=%s X=%s' % (
                    leg['side'],
                    leg['position_effect'],
                    util.color.mulla(instrument['strike_price']),
                    instrument['expiration_date'],
                ))

                premium += sum([
                    100 * D(x['price']) * D(x['quantity']) for x in leg['executions']
                ])

            premium *= -1 if option['direction'] == 'debit' else +1
            premiums[ticker] += premium

            closed[ticker].append("%s %s %s x%s P=%s K=%s" % (
                option['state'],
                option['type'],
                '/'.join(strategies),
                util.color.qty(option['quantity'], Z),
                util.color.mulla(premium),
                util.color.mulla(instrument['strike_price']),
            ))

            for l in legs:
                closed[ticker].append(' + l:%s' % l)

        stocks = defaultdict(list)
        data = api.orders('stocks', 'open')
        for order in data:
            uri = order['instrument']
            ticker = api.ticker(uri)
            stocks[ticker].append("%s %s x%s @%s" % (
                order['type'],
                order['side'],
                util.color.qty(order['quantity'], Z),
                util.color.mulla(order['price']),
            ))

        return dict(
            stocks=stocks,
            opened=opened,
            closed=closed,
            premiums=premiums,
            collateral=collateral,
            next_expiry=next_expiry,
        )
