#!/usr/bin/env python3

import os, sys, locale
import csv, requests, tempfile
import click, pickle, json
import math, hashlib, random
from collections import defaultdict, namedtuple
from functools import reduce

from datetime import datetime, timedelta, timezone

import pathlib
from pathlib import posixpath

from collections import defaultdict

import robin_stocks as rh
import yahoo_earnings_calendar as yec
import iexfinance.stocks as iex
import polygon

from pprint import pformat

from beautifultable import BeautifulTable

import __main__

import constants
from constants import ZERO as Z

import util
from util import dec as D

def fmt(f, d):
    def fn(k):
        try:
            return f.get(k, str)(d.get(k, 'N/A'))
        except:
            raise RuntimeError(f'Invalid parameters for fmt; d={d.get(k, "N/A")}, k={k})')
    return fn

def conterm(fr, to):
    delta = to - fr
    return 'short' if delta <= timedelta(days=365, seconds=0) else 'long'

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
        available = self.quantity(when=None)

        for lc in self.connections:
            available -= lc.qty

        if not when: return available

        for splitter in self.stock.splitters:
            if self.timestamp <= splitter.timestamp <= when:
                available = splitter.forward(available)

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

class Lot:
    def __init__(self, stock, sell):
        '''
        Every sell (transaction event) will have a corresponding realized Lot,
        and there's one additional active (unrealized) Lot per ticker, if any
        non-zero number of shares are still held for that stock beyond the last
        sale or first investment in the stock.
        '''
        self.type = 'realized'
        self.sell, self.qty = sell, sell.quantity()
        self.buys = []

        required = self.qty
        while required > 0:
            # Bad data from robinhood; workaround
            if stock.pointer == len(stock.events):
                stock.events.append(
                    TransactionEvent(
                        stock,
                        required,
                        0.00,
                        'buy',
                        str(datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)),
                        'FREE',
                    )
                )

            event = stock.events[stock.pointer]
            if type(event) is TransactionEvent:
                if event.side == 'buy':
                    available = event.available()
                    if available <= 0:
                        stock.pointer += 1
                        continue

                    lc = LotConnector(sell=self.sell, buy=event, requesting=required)
                    required -= lc.qty
                    self.buys.append(lc)
                else:
                    stock.pointer += 1
            elif type(event) is StockSplitEvent:
                stock.pointer += 1
            else:
                raise

    def bought(self):
        return {
            'qty': sum([e.qty for e in self.events if type(e) is TransactionEvent]),
            'value': sum([e.qty * e.price for e in self.events if type(e) is TransactionEvent]),
        }

    def sold(self):
        return {
            'qty': self.sell.qty,
            'value': self.sell.qty * self.sell.price,
        }

    @property
    def washsale_exempt(self):
        return True or False

    @property
    def costbasis(self):
        costbasis = {
            'short': { 'qty': 0, 'value': 0 },
            'long':  { 'qty': 0, 'value': 0 },
        }

        for lc in self.buys:
            costbasis[lc.term]['qty'] += lc.qty
            costbasis[lc.term]['value'] += lc.costbasis

        return costbasis

class LotConnector:
    def __init__(self, sell, buy, requesting):
        self.sell  = sell
        self.buy = buy

        self.qty = min(requesting, buy.available())

        # short-term or long-term sale (was the buy held over a year)
        self.term = conterm(self.buy.timestamp, self.sell.timestamp)

        self.buy.connect(self)

    @property
    def costbasis(self):
        return (self.sell.price() - self.buy.price()) * self.qty


class StockFIFO:
    mappings = {
        'FCAU': 'STLA'
    }

    def __init__(self, account, ticker):
        self.account = account
        self.ticker = StockFIFO.mappings.get(ticker, ticker)

        self.quantity = 0

        self.pointer = 0
        self.events = []

        self._ledger = []

        self.lots = []

        # StockSplits - These splitters will act as identity functions, but take effect whenever
        # the date of query and the date of a buy lie on different sides of the split date.  That
        # means, that the buy event needs to know "who's asking", or more specifically,
        # "when's asking?".
        self.splitters = [
            StockSplitEvent(
                stock=self,
                date=ss['exDate'],
                divisor=ss['fromFactor'],
                multiplier=ss['toFactor'],
            ) for ss in self.account.cached('stocks', 'splits', self.ticker)
        ]

        self._event_pool = self.splitters + self._events()

        self._iex = iex.Stock(self.ticker)


    def __getitem__(self, i):
        return self.events[i]

    def __repr__(self):
        return '<StockFIFO %s x %s @ mean unit cost of %s and current equity of %s>' % (
            self.ticker,
            util.color.qty(self.quantity),
            util.color.mulla(self.pps),
            util.color.mulla(self.price),
        )

    @property
    def earnings(self):
        return self.account.cached('events', 'earnings', self.ticker)

    @property
    def yesterday(self):
        return self.account.cached('stocks', 'yesterday', self.ticker)

    @property
    def marketcap(self):
        return self.account.cached('stocks', 'marketcap', self.ticker)

    @property
    def stats(self):
        return self.account.cached('stocks', 'stats', self.ticker)

    @property
    def beta(self):
        return self._iex.get_beta()

    @property
    def ttm_eps(self):
        return self.stats['ttmEPS']

    @property
    def fundamentals(self):
        return self.account.cached('stocks', 'fundamentals', self.tickers)[0]

    @property
    def price(self):
        return D(self.account.cached('stocks', 'prices', self.ticker)[0])

    @property
    def robinhood_id(self):
        '''Map ticker to robinhood id'''
        return self.account.cached('stocks', 'instruments', self.ticker, info='id')[0]

    @property
    def timestamp(self):
        return self.events[-1].timestamp

    @property
    def sells(self):
        return filter(lambda e: e.side == 'sell', self.transactions)

    @property
    def buys(self):
        return filter(lambda e: e.side == 'buy', self.transactions)

    @property
    def equity(self):
        return self.quantity * self.price

    @property
    def gain(self):
        return self.equity - self.cost

    @property
    def transactions(self, reverse=False):
        return filter(lambda e: type(e) is TransactionEvent, self.events)

    @property
    def subject2washsale(self):
        '''<now> <-- >30d? --> <current> <-- >30d? --> <last>'''

        thirty = timedelta(days=30, seconds=0)
        transactions = self.transactions
        current, last = None, None
        try:
            while True: last, current = next(transactions), next(transactions)
        except StopIteration:
            pass

        assert current is not None

        if (util.datetime.now() - current.timestamp) <= thirty: return True
        if last and (current.timestamp - last.timestamp) > thirty: return False
        return True

    def _events(self):
         return [
            TransactionEvent(
                self,
                ec['quantity'],
                ec['price'],
                ec['side'],
                se['updated_at'],
                se['type'],
            ) for se in self.account.cached('events', 'activities', self.ticker)
                for ec in se['equity_components']
        ]

    def costbasis(self, realized=True):
        '''TODO
        st cb qty -> ST Shs; realized [Andre Notes] ok – In my last email, I was expecting this
        to be the unrealized amount, so that’s why my comment “I would expect that the difference
        between the “equity” field and the total tax cost basis field(s) would equal the total of
        the “st cb capgain” and the “lt cb capgain”.
        '''
        costbasis = {
            'short': { 'qty': Z, 'value': Z },
            'long':  { 'qty': Z, 'value': Z },
        }

        if realized:
            for lot in self.lots:
                for term in costbasis.keys():
                    costbasis[term]['qty'] += lot.costbasis[term]['qty']
                    costbasis[term]['value'] += lot.costbasis[term]['value']
        else:
            for buy in self.buys:
                now = util.datetime.now()
                term = conterm(buy.timestamp, now)
                available = buy.available(when=now)
                price = buy.price(when=now)
                costbasis[term]['qty'] += available
                costbasis[term]['value'] += available * (self.price - price)

        return costbasis

    def push(self, transaction):
        self._event_pool.append(transaction)
        self._event_pool.sort(key=lambda e: e.timestamp)
        event = None
        while event is not transaction:
            event = self._event_pool.pop(0)

            if type(event) is TransactionEvent:
                if event.side == 'sell':
                    lot = Lot(self, event)
                    self.lots.append(lot)
                    self.quantity -= event.quantity()
                else:
                    self.quantity += event.quantity()
            elif type(event) is StockSplitEvent:
                self.quantity = event.forward(self.quantity)

            self.events.append(event)

            # Ledger for unit-testing
            cbr = self.costbasis(realized=True)
            cbu = self.costbasis(realized=False)
            self._ledger.append({
                'cnt':  len(self.events),
                'dts':  transaction.timestamp,
                'qty':  self.quantity,
                'ptr':  self.pointer,
                'pps':  self.pps,
                'crsq': cbr['short']['qty'],
                'crsv': cbr['short']['value'],
                'crlq': cbr['long']['qty'],
                'crlv': cbr['long']['value'],
                'cusq': cbu['short']['qty'],
                'cusv': cbu['short']['value'],
                'culq': cbu['long']['qty'],
                'culv': cbu['long']['value'],
            })

    @property
    def pps(self):
        '''
        average cost per share based on entire history for this stock
        '''
        realized = self.costbasis(realized=True)
        unrealized = self.costbasis(realized=False)
        aggregate = { 'value': Z, 'qty': Z }
        for costbasis in realized, unrealized:
            for term in 'short', 'long':
                for key in 'qty', 'value':
                    aggregate[key] += unrealized[term][key]

        return 0 if (
            aggregate['qty'] == Z
        ) else aggregate['value'] / aggregate['qty']


    def summarize(self):
        print('#' * 80)

        for event in self.events:
            if type(event) is StockSplitEvent: continue
            elif event.side == 'buy': continue
            elif event.side == 'sell':
                # The buys
                for tie in event.ties:
                    print(tie.bought)

                # The sell
                print('(%s)' % event)
                print()

        # Remaining buys (without a corresponding sell; current equity)
        for event in self.events[self.pointer:]:
            print(event)

        print("Cost : %s x %s = %s" % (util.color.qty(self.quantity), util.color.mulla(self.pps), util.color.mulla(self.cost)))
        print("Value: %s x %s = %s" % (util.color.qty(self.quantity), util.color.mulla(self.price), util.color.mulla(self.equity)))
        print("Capital Gains: %s" % (util.color.mulla(self.gain)))
        print()


class CSVReader:
    def __init__(self, account, context):
        '''
        CSV file is expected to be in reverse time sort order (that's just
        what the robinhood API happens to return.  No further sorting is done
        to account for the case where the CSV is not sorted in exactly this
        manner!
        '''
        self.active = None

        account.cached(
            'export', context,
            constants.CACHE_DIR, '%s/%s' % (constants.CACHE_DIR, context),
        )
        self.cachefile = os.path.join(constants.CACHE_DIR, '%s.csv' % context)
        with open(self.cachefile, newline='') as fh:
            reader = csv.reader(fh, delimiter=',', quotechar='"')
            self.header = next(reader)
            self.reader = reversed([line for line in reader if not line[0].startswith('#')])

    def __iter__(self):
        return self

    def __next__(self):
        self.active = next(self.reader, None)
        if self.active: return self
        raise StopIteration

    def get(self, field):
        return self.active[self.header.index(field)]

    @property
    def timestamp(self):
        raise NotImplementedError

    @property
    def ticker(self):
        return self.get(self._ticker_field)

    @property
    def side(self):
        return self.get('side')

    @property
    def otype(self):
        raise NotImplementedError

    @property
    def parameters(self):
        raise NotImplementedError


class StockReader(CSVReader):
    def __init__(self, account):
        super().__init__(account, 'stocks')
        self._ticker_field = 'symbol'

    @property
    def timestamp(self):
        return self.get('date')

    @property
    def otype(self):
        return self.get('order_type')

    @property
    def parameters(self):
        return (
            D(self.get('quantity')),
            D(self.get('average_price')),
            self.side,
            self.timestamp,
            self.otype,
        )


#class OptionReader(CSVReader):
#    def __init__(self, account, importer='options'):
#        super().__init__(account, importer)
#        self._ticker_field = 'chain_symbol'
#
#    @property
#    def timestamp(self):
#        return util.datetime.dtp.parse('%sT15:00:00.000000Z' % self.get('expiration_date'))
#
#    @property
#    def side(self):
#        side, otype = self.get('side'), self.get('option_type')
#
#        if (side, otype) == ('sell', 'call'):
#            side = 'sell'
#        elif (side, otype) == ('sell', 'put'):
#            side = 'buy'
#        elif (side, otype) == ('buy', 'call'):
#            side = 'buy'
#        elif (side, otype) == ('buy', 'put'):
#            side = 'sell'
#        else:
#            raise NotImplementedError
#
#        return side
#
#    @property
#    def otype(self):
#        return self.get('option_type')
#
#    @property
#    def parameters(self):
#        return (
#            100 * D(self.get('processed_quantity')),
#            D(self.get('strike_price')),
#            self.side,
#            self.timestamp,
#            '%s %s' % (self.side, self.otype,)
#        )


class Account:
    def __init__(self):
        self.robinhood = None
        self.yec = None

        self.polygon_api_key = None
        self.iex_api_key = None
        self.connect()

        self._portfolio = {}
        self.tickers = None
        self.stockReader = StockReader(self)
        #self.optionReader = OptionReader(self)

        self.data = None

    def get_stock(self, ticker):
        ticker = ticker.upper()
        if ticker not in self._portfolio:
            self._portfolio[ticker] = StockFIFO(self, ticker)

        return self._portfolio[ticker]

    @property
    def stocks(self):
        return self._portfolio

    def connected(fn):
        def connected(self, *args, **kwargs):
            self.connect()
            return fn(self, *args, **kwargs)
        return connected

    def connect(self):
        self.yec = yec.YahooEarningsCalendar()

        with open(os.path.join(pathlib.Path.home(), ".rhrc")) as fh:
            username, password, self.polygon_api_key, self.iex_api_key = [
                token.strip() for token in fh.readline().split(',')
            ]

            if self.robinhood is None:
                self.robinhood = rh.login(username, password)

            os.environ['IEX_TOKEN'] = self.iex_api_key
            os.environ['IEX_OUTPUT_FORMAT'] = 'json'

    def slurp(self):
        for ticker, parameters in self.transactions():
            stock = self.get_stock(ticker)
            transaction = TransactionEvent(stock, *parameters)
            stock.push(transaction)

        self.data = self.cached('account', 'holdings')
        self.tickers = sorted(self.data.keys())

        for ticker in self.tickers:
            stock = self.get_stock(ticker)
            self.data[ticker].update({
                'realized': stock.costbasis(realized=True),
                'unrealized': stock.costbasis(realized=False),
            })

        positions = self._get_positions()
        dividends = self._get_dividends()
        prices = self._get_prices()
        fundamentals = self._get_fundamentals()

        for ticker, datum in self.data.items():
            fifo = self.get_stock(ticker)

            datum['ticker'] = ticker
            datum['premium_collected'] = positions['premiums'][ticker]
            datum['dividends_collected'] = sum([
                D(div['amount']) for div in dividends[ticker]
            ])
            datum['activities'] = '\n'.join(positions['activities'].get(ticker, []))
            datum['pe_ratio'] = D(fundamentals[ticker]['pe_ratio'])
            datum['pb_ratio'] = D(fundamentals[ticker]['pb_ratio'])

            datum['marketcap'] = fifo.marketcap

            datum['price'] = D(prices[ticker])
            datum['50dma'] = D(fifo.stats['day50MovingAvg'])
            datum['200dma'] = D(fifo.stats['day200MovingAvg'])

    def _get_dividends(self):
        dividends = defaultdict(list)
        for datum in self.cached('account', 'dividends'):
            uri = datum['instrument']
            ticker = self.cached('stocks', 'instrument', uri, 'symbol')
            instrument = self.cached('stocks', 'instrument', uri)
            dividends[ticker].append(datum)
        return dividends

    def _get_fundamentals(self):
        return dict(zip(self.tickers, self.cached('stocks', 'fundamentals', self.tickers)))

    def _get_prices(self):
        return dict(zip(self.tickers, self.cached('stocks', 'prices', self.tickers)))

    def _get_positions(self):
        data = self.cached('options', 'positions:all')

        activities = defaultdict(list)
        for option in [o for o in data if D(o['quantity']) != Z]:
            ticker = option['chain_symbol']

            uri = option['option']
            instrument = self.cached('stocks', 'instrument', uri)
            if instrument['state'] == 'expired':
                pass
                #raise
            elif instrument['tradability'] == 'untradable':
                raise

            premium = Z
            if instrument['state'] != 'queued':
                #signum = -1 if option['type'] == 'short' else +1
                premium -= D(option['quantity']) * D(option['average_price'])

            activities[ticker].append("%s %s %s x%s K=%s X=%s P=%s" % (
                instrument['state'],
                option['type'],
                instrument['type'],
                util.color.qty(option['quantity'], Z),
                util.color.mulla(instrument['strike_price']),
                instrument['expiration_date'],
                util.color.mulla(premium),
            ))

        premiums = defaultdict(lambda: Z)
        data = self.cached('orders', 'options:all')
        for option in [o for o in data if o['state'] not in ('cancelled', 'expired')]:
            ticker = option['chain_symbol']

            if ticker in DEBUG:
                print('#' * 80)
                dprint(option, title='orders.options:all')

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
                instrument = self.cached('stocks', 'instrument', uri)

                if ticker in DEBUG:
                    dprint(instrument, title=f'stocks:instrument({uri})')

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

            activities[ticker].append("%s %s %s x%s P=%s K=%s" % (
                option['state'],
                option['type'],
                '/'.join(strategies),
                util.color.qty(option['quantity'], Z),
                util.color.mulla(premium),
                util.color.mulla(instrument['strike_price']),
            ))

            for l in legs:
                activities[ticker].append(' + l:%s' % l)

        data = self.cached('orders', 'stocks:open')
        for order in data:
            uri = order['instrument']
            ticker = self.cached('stocks', 'instrument', uri, 'symbol')
            activities[ticker].append("%s %s x%s @%s" % (
                order['type'],
                order['side'],
                util.color.qty(order['quantity'], Z),
                util.color.mulla(order['price']),
            ))

        return dict(premiums=premiums, activities=activities)

    @connected
    def tickers(self):
        return (stock.ticker for stock in self.stockReader)

    @connected
    def transactions(self):
        return ((stock.ticker, stock.parameters) for stock in self.stockReader)

    @connected
    def _machine(self, area, subarea, *args):
        return ROBIN_STOCKS_API[area][subarea](*args)

    def human(self, area, subarea, *args):
        return util.dump(f'{area}:{subarea}', self._machine(area, subarea, *args))

    @connected
    def _pickled(self, cachefile, area, subarea, *args, **kwargs):
        if cachefile.exists():
            data = pickle.load(open(cachefile, 'rb'))
        else:
            endpoint = ROBIN_STOCKS_API[area][subarea]

            if type(endpoint) is PolygonEndpoint:
                with polygon.RESTClient(self.polygon_api_key) as pg:
                    fn = getattr(pg, endpoint.function)
                    response = fn(*args, **kwargs)
                    if response.status == 'OK':
                        data = response.results
                    else:
                        raise
            elif type(endpoint) is IEXFinanceEndpoint:
                if area == 'stocks':
                    fn = getattr(iex.Stock(args[0]), endpoint.function)
                    data = fn(*args[1:], **kwargs)
                else:
                    raise
            elif type(endpoint) is YahooEarningsCalendarEndpoint:
                fn = getattr(self.yec, endpoint.function)
                data = datetime.fromtimestamp(fn(args[0]))
            else:
                data = endpoint.function(*args, **kwargs)

            arguments = [
                ','.join(map(str, args)),
                ','.join(['%s=%s' % (k, v) for k, v in kwargs]),
            ]
            dprint("Cache fault on %s:%s(%s)" % (
                area, subarea, ','.join(arguments),
            ))

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                print(tmp.name)
                pickle.dump(data, tmp)
                tmp.flush()
                os.rename(tmp.name, cachefile)

        return data

    def cached(self, area, subarea, *args, **kwargs):
        endpoint = ROBIN_STOCKS_API[area][subarea]
        uniqname = '-'.join([
            area,
            subarea,
            hashlib.sha1(','.join(map(str, args)).encode()).hexdigest(),
        ])
        cachefile = pathlib.Path(f'{constants.CACHE_DIR}/{uniqname}.pkl')

        data, age, fresh, this_second = {}, -1, False, util.datetime.now()
        while not fresh:
            data = self._pickled(cachefile, area, subarea, *args, **kwargs)

            then = datetime.fromtimestamp(cachefile.lstat().st_mtime, tz=timezone.utc)
            age = this_second - then

            jitter = 1 + random.random()
            fresh = (endpoint.ttl == -1) or age.total_seconds() < endpoint.ttl * jitter
            if not fresh:
                cachefile.unlink()

        return data

RobinStocksEndpoint = namedtuple('RobinStocksEndpoint', ['ttl', 'function'])
PolygonEndpoint = namedtuple('PolygonEndpoint', ['ttl', 'function'])
IEXFinanceEndpoint = namedtuple('PolygonEndpoint', ['ttl', 'function'])
YahooEarningsCalendarEndpoint = namedtuple('YahooEarningsCalendarEndpoint', ['ttl', 'function'])

ROBIN_STOCKS_API = {
    'profiles': {
        'account'     : RobinStocksEndpoint(7200, rh.profiles.load_account_profile),
        'investment'  : RobinStocksEndpoint(7200, rh.profiles.load_investment_profile),
        'portfolio'   : RobinStocksEndpoint(7200, rh.profiles.load_portfolio_profile),
        'security'    : RobinStocksEndpoint(7200, rh.profiles.load_security_profile),
        'user'        : RobinStocksEndpoint(7200, rh.profiles.load_user_profile),
    },
    'stocks': {
        'fundamentals': RobinStocksEndpoint(7200,  rh.stocks.get_fundamentals),
        'instruments' : RobinStocksEndpoint(7200,  rh.stocks.get_instruments_by_symbols),
        'instrument'  : RobinStocksEndpoint(7200,  rh.stocks.get_instrument_by_url),
        'prices'      : RobinStocksEndpoint(7200,  rh.stocks.get_latest_price),
        'news'        : RobinStocksEndpoint(7200,  rh.stocks.get_news),
        'quote'       : IEXFinanceEndpoint(7200,  'get_quote'),
        'quotes'      : RobinStocksEndpoint(7200,  rh.stocks.get_quotes),
        'ratings'     : RobinStocksEndpoint(7200,  rh.stocks.get_ratings),
        'splits'      : IEXFinanceEndpoint(86400,  'get_splits'),
        'yesterday'   : IEXFinanceEndpoint(21600,  'get_previous_day_prices'),
        'marketcap'   : IEXFinanceEndpoint(86400,  'get_market_cap'),
        'losers'      : IEXFinanceEndpoint(300,    'get_market_losers'),
        'gainers'     : IEXFinanceEndpoint(300,    'get_market_gainers'),
        'stats'       : IEXFinanceEndpoint(21600,  'get_key_stats'),
        'historicals' : RobinStocksEndpoint(7200,  rh.stocks.get_stock_historicals),
    },
    'options': {
        'positions:all'  : RobinStocksEndpoint(300, rh.options.get_all_option_positions),
        'positions:open' : RobinStocksEndpoint(300, rh.options.get_open_option_positions),
        'positions:agg'  : RobinStocksEndpoint(300, rh.options.get_aggregate_positions),
        'chains'         : RobinStocksEndpoint(300, rh.options.get_chains),
    },
    'orders': {
        'options:open'   : RobinStocksEndpoint(300, rh.orders.get_all_open_option_orders),
        'options:all'    : RobinStocksEndpoint(300, rh.orders.get_all_option_orders),
        'stocks:open'    : RobinStocksEndpoint(300, rh.orders.get_all_open_stock_orders),
        'stocks:all'     : RobinStocksEndpoint(300, rh.orders.get_all_stock_orders),
    },
    'markets': {
        'movers'         : RobinStocksEndpoint(3600, rh.markets.get_top_movers),
    },
    'events': {
        'earnings:next'  : YahooEarningsCalendarEndpoint(21600, 'get_next_earnings_date'),
        'earnings:all'   : YahooEarningsCalendarEndpoint(86400, 'get_earnings_of'),
        'earnings'       : RobinStocksEndpoint(7200,  rh.stocks.get_earnings),
        'activities'     : RobinStocksEndpoint(7200,  rh.stocks.get_events),
    },
    'account': {
        'notifications'  : RobinStocksEndpoint(3600, rh.account.get_latest_notification),
        'positions:all'  : RobinStocksEndpoint(3600, rh.account.get_all_positions),
        'positions:open' : RobinStocksEndpoint(3600, rh.account.get_open_stock_positions),
        'margin'         : RobinStocksEndpoint(3600, rh.account.get_margin_interest),
        'dividends:total': RobinStocksEndpoint(3600, rh.account.get_total_dividends),
        'dividends'      : RobinStocksEndpoint(3600, rh.account.get_dividends),
        'fees'           : RobinStocksEndpoint(3600, rh.account.get_subscription_fees),
        'phoenix'        : RobinStocksEndpoint(3600, rh.account.load_phoenix_account),
        'holdings'       : RobinStocksEndpoint(3600, rh.account.build_holdings),
    },
    'export': {
        'stocks'         : RobinStocksEndpoint(3*3600, rh.export_completed_stock_orders),
        'options'        : RobinStocksEndpoint(3*3600, rh.export_completed_option_orders),
    },
}

DEBUG = []
def dprint(data, title=None):
    if len(DEBUG) == 0: return

    title = '%s(%d entries)' % (f'{title} ' if title else '', len(data))
    print(
        util.dump(
            f'[DEBUG:%s]-=[ %s ]=-' % (
                ','.join(DEBUG),
                title
            ),
            data
        )
    )

@click.group()
@click.option('-D', '--debug-tickers',multiple=True,  default=None)
@click.pass_context
def cli(ctx, debug_tickers):
    global DEBUG
    DEBUG=debug_tickers
    ctx.ensure_object(dict)

CONTEXT_SETTINGS = dict(token_normalize_func=lambda x: x.lower())

@cli.command(help='Stock Overview (raw API dumps)', context_settings=CONTEXT_SETTINGS)
@click.option('-s', '--subarea', required=True, type=click.Choice(ROBIN_STOCKS_API['stocks']))
@click.option('-t', '--ticker', required=True)
@click.pass_context
def stocks(ctx, subarea, ticker):
    account = ctx.obj['account']
    print(account.human('stocks', subarea, ticker))

@cli.command(help='Options Overviews (raw API dumps)')
@click.option('-s', '--subarea', required=True, type=click.Choice(ROBIN_STOCKS_API['options']))
@click.option('-t', '--ticker', required=False)
@click.pass_context
def options(ctx, subarea, ticker):
    account = ctx.obj['account']
    print(account.human('options', subarea, ticker))

@cli.command(help='Overviews (raw API dumps)')
@click.option('-s', '--subarea', required=True, type=click.Choice(ROBIN_STOCKS_API['profiles']))
@click.pass_context
def profiles(ctx, subarea):
    account = ctx.obj['account']
    print(account.human('profiles', subarea))

@cli.command(help='Account Overviews (raw API dumps)')
@click.option('-s', '--subarea', required=True, type=click.Choice(ROBIN_STOCKS_API['account']))
@click.pass_context
def account(ctx, subarea):
    account = ctx.obj['account']
    print(account.human('account', subarea))

@cli.command(help='Market Overviews (raw API dumps)')
@click.option('-s', '--subarea', required=True, type=click.Choice(ROBIN_STOCKS_API['markets']))
@click.pass_context
def markets(ctx, subarea):
    account = ctx.obj['account']
    print(account.human('markets', subarea))

VIEWS = {
    'pie': {
        'sort_by': 'rank',
        'columns': [
            'ticker', 'percentage',
            'price', 'quantity',
            'equity', #cost
            'equity_change', 'percent_change',
            #'cost_basis',
            'growth',
            'premium_collected', 'dividends_collected',
            'short', 'bucket', 'rank', 'ma', 'since_close', 'since_open', 'alerts',
            'pe_ratio', 'pb_ratio', 'beta',
            'activities',
        ],
    },
    'active': {
        'sort_by': 'ticker',
        'filter_by': lambda d: len(d['activities']),
        'columns': [
            'ticker', 'percentage',
            'price', 'quantity',
            'equity', #cost
            'equity_change', 'percent_change',
            #'cost_basis',
            'growth',
            'premium_collected', 'dividends_collected',
            'activities',
        ],
    },
    'tax': {
        'sort_by': 'ticker',
        'columns': [
            'ticker',
            'price', 'quantity',
            'equity', #cost
            #'cost_basis',
            'st_cb_qty', 'st_cb_capgain',
            'lt_cb_qty', 'lt_cb_capgain',
            'premium_collected', 'dividends_collected',
        ],
    },
}

@cli.command(help='Views')
@click.option('-v', '--view', default='pie', type=click.Choice(VIEWS.keys()))
@click.option('-r', '--reverse', default=False, is_flag=True, type=bool)
@click.option('-l', '--limit', default=-1, type=int)
@click.pass_context
def tabulize(ctx, view, reverse, limit):
    account = ctx.obj['account']
    account.slurp()

    buckets = [ 0, 3, 5, 8, 13, 21, 34 ]
    pie = { k: [] for k in buckets }

    fundamentals = account._get_fundamentals()
    for ticker, datum in account.data.items():
        fifo = account.get_stock(ticker)
        index = fifo.pointer
        buy = fifo[index]
        assert buy.side == 'buy'

        alerts = []

        marketcap = datum['marketcap']
        if marketcap is not None:
            marketcap /= 1000000000
            if marketcap > 10: sizestr = util.color.colored('L', 'green')
            elif marketcap > 2:
                if 4 < marketcap < 5:
                    sizestr = util.color.colored('SWEET!', 'magenta')
                else:
                    sizestr = util.color.colored('M', 'blue')
            else: sizestr = util.color.colored('S', 'yellow')
        else:
            marketcap, sizestr = Z, util.color.colored('?', 'red')
        alerts.append('%s/%sB' % (sizestr, util.color.mulla(marketcap)))

        #if buy.term == 'st': alerts.append(util.color.colored('ST!', 'yellow'))
        if fifo.subject2washsale: alerts.append(util.color.colored('WS!', 'yellow'))
        if datum['pe_ratio'].is_nan() or datum['pe_ratio'] < 10: alerts.append(util.color.colored('PE!', 'red'))

        datum['alerts'] = ' '.join(alerts)

        prices = (datum['200dma'], datum['50dma'], datum['price'])
        score = reduce(lambda a,b:a*b, map((lambda n: n[0]/n[1]), map(sorted, zip(prices, sorted(prices)))))

        p200, p50, p = prices
        c = 'yellow'
        if p > p50 > p200: c = 'green'
        elif p < p50 < p200: c = 'red'
        datum['ma'] = util.color.colored('%0.3f'%score, c)

        yesterday = fifo.yesterday

        c = D(yesterday['close'])
        datum['since_close'] = (p - c) / c if c > Z else Z

        o = D(fundamentals[ticker]['open'])
        datum['since_open'] = (p - o) / o

        datum['growth'] = Z
        #(
        #    sum((
        #        D(datum['equity']),
        #        D(datum['dividends_collected']),
        #        D(datum['premium_collected']),
        #    )) / D(datum['cost']) - 1
        #)

        #l = D(fundamentals[ticker]['low_52_weeks'])
        #h = D(fundamentals[ticker]['high_52_weeks'])
        #datum['year'] = 100 * (p - l) / h

        eq = D(datum['equity'])
        datum['bucket'] = 0
        datum['short'] = 0
        distance = D('0.1')
        for i, b in enumerate([s * 1000 for s in buckets[1:]]):
            l, h = eq*(1-distance), eq*(1+distance)
            if l < b < h:
                if p < 25: round_to = D('1')
                elif p < 100: round_to = D('0.5')
                elif p < 1000: round_to = D('0.2')
                else: round_to = D('.1')
                datum['bucket'] = b//1000
                datum['short'] = util.rnd((eq - b) / datum['price'], round_to)
                break

        # Maps [-100/x .. +100/x] to [0 .. 100]
        normalize = lambda p, x: (p*x+1)/2

        yesterchange = D(yesterday['marketChangeOverTime'])
        totalchange = D(datum['percent_change'])/100
        eq = D(datum['equity'])
        scores = {
            'yesterchange':     (25, normalize(yesterchange, D('4'))),
            'since_close':      (25, normalize(datum['since_close'], D('4'))),
            'since_open':       (15, normalize(datum['since_open'], D('4'))),
            'bucket':           (15, normalize((eq/1000 - datum['bucket'])/13, 1) if datum['bucket'] > 0 else 1),
            'totalchange':      (10, normalize(-totalchange, D('0.25'))),
            'growth':           (10, normalize(datum['growth'], D('1'))),
        }

        if ticker in DEBUG:
            print(util.dump('scores', scores))

        datum['rank'] = sum([pct * min(100, score) for (pct, score) in scores.values()])

    #if DEBUG:
    #    for o in [o for o in account.cached('account', 'positions:all')]:
    #        dprint(o, title='account:positions:all')
    #    for o in [o for o in account.cached('options', 'positions:agg')]:
    #        dprint(o, title='options.positions:agg')

    formats = {
        'bucket': lambda b: util.color.colored('$%dk' % b, 'blue'),
        'since_open': util.color.mpct,
        'since_close': util.color.mpct,
        'price': util.color.mulla,
        'quantity': util.color.qty0,
        'average_buy_price': util.color.mulla,
        'equity': util.color.mulla,
        'percent_change': util.color.pct,
        'equity_change': util.color.mulla,
        'pe_ratio': util.color.qty,
        'pb_ratio': util.color.qty,
        'percentage': util.color.pct,
        'rank': int,
        'delta': util.color.qty,
        'short': util.color.qty1,
        'premium_collected': util.color.mulla,
        'dividends_collected': util.color.mulla,
        'growth': util.color.mpct,
        'st_cb_qty': util.color.qty,
        'st_cb_capgain': util.color.mulla,
        'lt_cb_qty': util.color.qty,
        'lt_cb_capgain': util.color.mulla,
        'cost': util.color.mulla,
        'cost_basis': util.color.mulla,
    }

    table = BeautifulTable(maxwidth=300)
    table.set_style(BeautifulTable.STYLE_GRID)
    table.columns.header = [h.replace('_', '\n') for h in VIEWS[view]['columns']]

    if 'activate' in VIEWS[view]['columns']:
        table.columns.alignment['activities'] = BeautifulTable.ALIGN_LEFT

    for datum in account.data.values():
        f = fmt(formats, datum)
        table.rows.append([f(h) for h in VIEWS[view]['columns']])

    sort_by, filter_by = (
        VIEWS[view].get('sort_by', 'ticker'),
        VIEWS[view].get('filter_by', None)
    )

    if filter_by is not None:
        table = table.rows.filter(filter_by)

    table.rows.sort(sort_by, reverse)

    if DEBUG:
        print(table.rows.filter(lambda row: row['ticker'] in DEBUG))
    else:
        print(table.rows[:limit] if limit > -1 else table)


@cli.command(help='Account History')
@click.pass_context
def history(ctx):
    account = ctx.obj['account']
    account.slurp()
    for ticker, stock in account.stocks.items():
        if len(DEBUG) > 0 and ticker not in DEBUG: continue
        stock.summarize()

def preinitialize(repl=False):
    locale.setlocale(locale.LC_ALL, '')
    rh.helper.set_output(open(os.devnull,"w"))

    if not posixpath.exists(constants.CACHE_DIR):
        os.mkdir(constants.CACHE_DIR)

acc = None
if __name__ == '__main__':
    preinitialize()
    cli(obj={'account': Account()})
elif not hasattr(__main__, '__file__'):
    print("Interactive Execution Mode Detected")

    print("Pre-Initializing REPL Environment...")
    preinitialize(repl=True)

    print("Initializing REPL Environment...")
    module = sys.modules[__name__]
    acc = Account()
    acc.slurp()

    print("Injecting Stock Objects for all known Tickers from Robinhood...")
    for ticker in acc.tickers:
        key = ticker.lower()
        setattr(module, key, acc.get_stock(ticker))
        setattr(module, '_%s' % key, iex.Stock(ticker))

    print("Done! Available ticker objects:")
    print(" + rh.acc           (Local Robinhood Account object)")
    print(" + rh.acc.rh        (RobinStocksEndpoint API)")
    print(" + rh.acc.iex       (IEXFinanceEndpoint API)")
    print(" + rh.acc.yec       (YahooEarningsCalendarEndpoint API)")
    print(" + rh._<ticker>     (IEXFinanceEndpoint Stock object API)")
    print(" + rh.<ticker>      (Local Stock object API multiplexor")
