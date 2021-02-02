#!/usr/bin/env python3

import os, sys, locale
import csv, requests
import click, pickle, json
import math, hashlib, random
from collections import defaultdict, namedtuple

import pytz
from datetime import datetime, timedelta, timezone
import datetime as dt

import pathlib
from pathlib import posixpath

import numpy as np
import dateutil.parser as dtp
from collections import defaultdict, Counter

import robin_stocks as rh
import iexfinance as iex
import iexfinance.stocks as iex_stocks
import polygon
pg = None

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter

from pprint import pformat

from beautifultable import BeautifulTable
from colorama import init, Back, Fore
from termcolor import colored


def ts_to_datetime(ts) -> str:
    return dt.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M')


def dump(heading, obj):
    return '%s\n%s' % (
        heading,
        highlight((json.dumps(obj, indent=4)), PythonLexer(), Terminal256Formatter()),
    )

def c(f, n, d, v):
    n = flt(n)
    c = d[0]
    for i, v in enumerate(v):
        if v < n:
            c = d[i+1]

    return colored(f % n, c)

def mulla(m):
    c = 'red' if flt(m) < 0 else 'green'
    return colored(locale.currency(flt(m), grouping=True), c)

ZERO = 1e-5

flt = np.single
rnd = lambda a, r: round(a / r) * r
sgn = lambda n: -1 if n <= 0 else 1

pct = lambda p: c('%0.2f%%', p, ['red', 'green', 'magenta'], [0, 70])
qty = lambda q, dp=2: c(f'%0.{dp}f', q, ['yellow', 'cyan'], [0])

fmt = lambda f, d: lambda k: f.get(k, str)(d.get(k, 'N/A'))

now = lambda: pytz.UTC.localize(datetime.now())


CACHE_DIR='/tmp'
if len(sys.argv) > 0:
    CACHE_DIR=posixpath.join(
        posixpath.dirname(sys.argv[0]),
        '.cached',
    )
    if not posixpath.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)


class Event:
    ident = 0
    tally = defaultdict(int)

    def __init__(self):
        self.running = 0

class StockSplitEvent(Event):
    def __init__(self, ticker, date, multiplier, divisor):
        super().__init__()

        self.ident = Event.ident
        Event.ident += 1

        self.ticker = ticker
        self.timestamp = pytz.UTC.localize(dtp.parse(date))
        self.multiplier = flt(multiplier)
        self.divisor = flt(divisor)

    def __repr__(self):
        rstr = '#%05d %s %s stock-split %s-for-%s @ %s' % (
            self.ident,
            self.ticker,
            qty(self.running),
            qty(self.multiplier),
            qty(self.divisor),
            self.timestamp,
        )

        return '<StockSplitEvent  %s>' % rstr


    def settle(self, stock):
        Event.tally[self.ticker] *= self.multiplier
        Event.tally[self.ticker] /= self.divisor
        self.running = Event.tally[self.ticker]

    @property
    def unsettled(self):
        return 0


class TransactionEvent(Event):

    @property
    def signum(self):
        return { 'buy': +1, 'sell': -1 }[self.side]

    def settle(self, stock):
        Event.tally[self.ticker] += self.signum * self.qty
        self.running = Event.tally[self.ticker]
        LotConnector.settle(stock)

    def tie(self, tie):
        self.ties.append(tie)

    def __init__(self, ticker, side, qty, price, timestamp, otype):
        super().__init__()

        self.ident = Event.ident
        Event.ident += 1

        self.side = side
        self.ticker = ticker
        self.qty = flt(qty)
        self.price = flt(price)
        self.timestamp = dtp.parse(timestamp)
        self.ties = []
        self.otype = otype

    def __repr__(self):
        q = qty(self.qty)
        if len(self.ties) > 0:
            q = '%s (%s)' % (
                q,
                '+'.join([qty(tie.qty) for tie in self.ties])
            )

        rstr = '#%05d %s %s %s %s (cg=%s): %s x %s = %s @ %s' % (
            self.ident,
            self.ticker,
            qty(self.running),
            self.otype,
            self.side,
            self.term,
            q,
            mulla(self.price),
            mulla(self.price * self.qty),
            self.timestamp,
        )

        if self.side == 'sell':
            cb = self.costbasis
            stg = cb['st']['gain']
            stq = cb['st']['qty']
            ltg = cb['lt']['gain']
            ltq = cb['lt']['qty']
            rstr += ' | cb=[st=%sx%s=%s, lt=%sx%s=%s]' % (
                mulla(stg), qty(stq), mulla(stg * stq),
                mulla(ltg), qty(ltq), mulla(ltg * ltq),
            )

        return '<TransactionEvent %s>' % rstr

    @property
    def term(self):
        term = '--'
        if self.side == 'buy':
            if len(self.ties) == 1:
                term = self.ties[0].term
            else:
                term = 'st' if (
                    (now() - self.timestamp) <= timedelta(days=365, seconds=0)
                ) else 'lt'

        return term

    @property
    def costbasis(self):
        assert self.side == 'sell'

        # Does DIV go in here?
        cb = {
            'st': Counter({ 'qty': 0, 'gain': 0 }),
            'lt': Counter({ 'qty': 0, 'gain': 0 }),
        }

        for tie in self.ties:
            cb[tie.term]['qty'] += tie.qty
            cb[tie.term]['gain'] -= tie.bought.price * tie.qty

        for term in [t for t in ('st', 'lt') if cb[t]['qty'] > 0]:
            cb[term]['gain'] += self.price * cb[term]['qty']
            cb[term]['gain'] /= cb[term]['qty']

        return cb

    @property
    def unsettled(self):
        '''
        For a buy lot, this is how many remaining units are left in this lot.
        For a sell lot, this is how many outstanding units are unaccounted for.

        Different things, but the same underlying calculation.
        '''
        return self.qty - sum([tie.qty for tie in self.ties])

class LotConnector:
    @classmethod
    def settle(self, stock):
        event = stock.events[-1]
        assert type(event) is TransactionEvent

        if event.side == 'buy':
            pass
            #stock.qty += event.qty FIXME?
        elif event.side == 'sell':
            qty = event.qty
            while qty > ZERO:
                a = stock.pointer < len(stock.events)
                if a:
                    b = type(stock.events[stock.pointer]) is StockSplitEvent
                    if b:
                        stock.pointer += 1
                    else:
                        c = stock.events[stock.pointer].side == 'sell'
                        d = stock.events[stock.pointer].unsettled <= ZERO
                        if c or d:
                            stock.pointer += 1
                        else:
                            l = LotConnector(event, stock)
                            qty -= l.qty
                else:
                    l = LotConnector(event, stock)
                    qty -= l.qty

    def __init__(self, sold, stock):
        events = stock.events

        self.sold, self.bought = sold, None
        while self.bought is None or self.bought.side != 'buy' or self.bought.unsettled <= ZERO:
            if self.bought is not None:
                stock.pointer += 1
            if stock.pointer == len(events):
                print('FIXME', sold)
                events.append(
                    TransactionEvent(
                        stock.ticker,
                        'buy',
                        sold.unsettled,
                        0.00,
                        str(datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)),
                        'FREE',
                    )
                )
            self.bought = events[stock.pointer]

        self.qty = min((self.sold.unsettled, self.bought.unsettled))
        self.term = 'st' if (
            (self.sold.timestamp - self.bought.timestamp) <= timedelta(days=365, seconds=0)
        ) else 'lt'
        self.sold.tie(self)
        self.bought.tie(self)

    def __repr__(self):
        return '%s <--(x%0.5f:%s)--> %s' % (
            self.sold,
            self.qty,
            self.term,
            self.bought,
        )

class StockFILO:
    @property
    def timestamp(self):
        return self.events[-1].timestamp

    @property
    def transactions(self):
        return [e for e in self.events if type(e) is TransactionEvent]

    def __init__(self, account, ticker):
        self.account = account
        self.ticker = ticker
        self.pointer = 0
        self.events = []
        self.event_pool = sorted(
            [
                StockSplitEvent(
                    ticker=ticker,
                    #date=ss['exDate'],
                    #multiplier=ss.get('forfactor', 1),
                    #divisor=ss.get('tofactor', ss['ratio']),
                    date=ss['exDate'],
                    divisor=ss['fromFactor'],
                    multiplier=ss['toFactor'],
                ) for ss in account.cached('stocks', 'splits', ticker)
            ] + [
                TransactionEvent(
                    self.ticker,
                    ec['side'],
                    ec['quantity'],
                    ec['price'],
                    se['updated_at'],
                    se['type'],
                ) for se in account.cached('stocks', 'events', ticker)
                    for ec in se['equity_components']
            ], key=lambda e: e.timestamp
        )

    def __getitem__(self, i):
        return self.events[i]

    def __repr__(self):
        return '<StockFILO:%-5s x%8.2f @ mean:%s>' % (
            self.ticker,
            self.quantity,
            mulla(self.average),
        )

    def push(self, qty, price, side, timestamp, otype):
        transaction = TransactionEvent(
            self.ticker,
            side,
            qty,
            price,
            timestamp,
            otype,
        )
        self.event_pool.append(transaction)
        self.event_pool.sort(key=lambda e: e.timestamp)
        e = None
        while e is not transaction:
            e = self.event_pool.pop(0)
            self.events.append(e)
            e.settle(self)

    @property
    def price(self):
        return self.account.get_price(self.ticker)

    @property
    def average(self):
        return self.cost / self.quantity if self.quantity > ZERO else 0

    @property
    def quantity(self):
        return self.events[-1].running

    @property
    def cost(self):
        return sum([
            e.price * e.unsettled
            for e in self.transactions
            if e.side == 'buy'
        ])

    @property
    def equity(self):
        return self.quantity * self.price

    @property
    def gain(self):
        return self.equity - self.cost

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
                print(event)
                print()

        # Remaining buys (without a corresponding sell; current equity)
        for event in self.events[self.pointer:]:
            print(event)

        print("Cost : %s x %s = %s" % (qty(self.quantity), mulla(self.average), mulla(self.cost)))
        print("Value: %s x %s = %s" % (qty(self.quantity), mulla(self.price), mulla(self.equity)))
        print("Capital Gains: %s" % (mulla(self.gain)))
        print()


class CSVReader:
    def __init__(self, account, importer):
        self.active = None

        account.cached(
            'export', importer,
            CACHE_DIR, '%s/%s' % (CACHE_DIR, importer),
        )
        self.cachefile = os.path.join(CACHE_DIR, '%s.csv' % importer)
        with open(self.cachefile, newline='') as fh:
            reader = csv.reader(fh, delimiter=',', quotechar='"')
            self.header = next(reader)
            self.reader = reversed(list(reader))

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
    def __init__(self, account, importer='stock'):
        super().__init__(account, importer)
        self._ticker_field = 'symbol'

    @property
    def timestamp(self):
        return self.get('date')
        return dtp.parse(self.get('date'))

    @property
    def otype(self):
        return self.get('order_type')

    @property
    def parameters(self):
        return (
            flt(self.get('quantity')),
            flt(self.get('average_price')),
            self.side,
            self.timestamp,
            self.otype,
        )


class OptionReader(CSVReader):
    def __init__(self, account, importer='option'):
        super().__init__(account, importer)
        self._ticker_field = 'chain_symbol'

    @property
    def timestamp(self):
        return dtp.parse('%sT15:00:00.000000Z' % self.get('expiration_date'))

    @property
    def side(self):
        side, otype = self.get('side'), self.get('option_type')

        if (side, otype) == ('sell', 'call'):
            side = 'sell'
        elif (side, otype) == ('sell', 'put'):
            side = 'buy'
        elif (side, otype) == ('buy', 'call'):
            side = 'buy'
        elif (side, otype) == ('buy', 'put'):
            side = 'sell'
        else:
            raise NotImplementedError

        return side

    @property
    def otype(self):
        return self.get('option_type')

    @property
    def parameters(self):
        return (
            100 * flt(self.get('processed_quantity')),
            flt(self.get('strike_price')),
            self.side,
            self.timestamp,
            '%s %s' % (self.side, self.otype,)
        )


class Account:
    def __init__(self):
        self.robinhood = None
        self.polygon_api_key = None
        self.iex_api_key = None
        self.connect()

        self.portfolio = {}
        self.tickers = None
        self.stockReader = StockReader(self)
        self.optionReader = OptionReader(self)

        self.data = None

    def __getitem__(self, ticker):
        if ticker not in self.portfolio:
            self.portfolio[ticker] = StockFILO(self, ticker)

        return self.portfolio[ticker]

    def connected(fn):
        def connected(self, *args, **kwargs):
            self.connect()
            return fn(self, *args, **kwargs)
        return connected

    def connect(self):
        with open(os.path.join(pathlib.Path.home(), ".rhrc")) as fh:
            username, password, self.polygon_api_key, self.iex_api_key = [
                token.strip() for token in fh.readline().split(',')
            ]

            if self.robinhood is None:
                self.robinhood = rh.login(username, password)

            os.environ['IEX_TOKEN'] = self.iex_api_key
            os.environ['IEX_OUTPUT_FORMAT'] = 'json'

    def ticker2id(self, ticker):
        '''Map ticker to robinhood id'''
        return self.cached('stocks', 'instruments', ticker, info='id')[0]

    def slurp(self):
        for ticker, parameters in self.transactions():
            self[ticker].push(*parameters)

        self.data = self.cached('account', 'holdings')
        self.tickers = sorted(self.data.keys())

        costbasis = self._get_costbasis()
        for ticker in self.tickers:
            self.data[ticker].update(costbasis[ticker])

        positions = self._get_positions()
        dividends = self._get_dividends()
        prices = self._get_prices()
        fundamentals = self._get_fundamentals()

        for ticker, datum in self.data.items():
            datum['ticker'] = ticker
            datum['premium_collected'] = positions['premiums'][ticker]
            datum['dividends_collected'] = sum([
                float(div['amount']) for div in dividends[ticker]
            ])
            datum['activities'] = '\n'.join(positions['activities'].get(ticker, []))
            datum['pe_ratio'] = flt(fundamentals[ticker]['pe_ratio'])
            datum['pb_ratio'] = flt(fundamentals[ticker]['pb_ratio'])
            datum['price'] = flt(prices[ticker])

    def _get_costbasis(self):
        #TODO: Add realized as well as unrealized for this data
        costbasis = defaultdict(Counter)
        for ticker, stock in self.portfolio.items():
            costbasis[ticker].update({
                'cost':           stock.cost,
                'cost_basis':     stock.gain,
                'st_cb_qty':      0,
                'st_cb_capgain':  0,
                'lt_cb_qty':      0,
                'lt_cb_capgain':  0,
            })
            for trans in (t for t in stock.transactions if t.side == 'sell'):
                costbasis[ticker].update({
                    'st_cb_qty':     trans.costbasis['st']['qty'],
                    'st_cb_capgain': trans.costbasis['st']['gain'],
                    'lt_cb_qty':     trans.costbasis['lt']['qty'],
                    'lt_cb_capgain': trans.costbasis['lt']['gain'],
                })
        return costbasis

    def _get_dividends(self):
        dividends = defaultdict(list)
        for datum in self.cached('account', 'dividends'):
            uri = datum['instrument']
            ticker = self.cached('stocks', 'instrument', uri, 'symbol')
            instrument = self.cached('stocks', 'instrument', uri)
            dividends[ticker].append(datum)
        return dividends

    def _get_fundamentals(self):
        return dict(
            zip(
                self.tickers,
                self.cached('stocks', 'fundamentals', self.tickers)
            )
        )

    def get_price(self, ticker):
        return self._get_prices().get(
            ticker,
            flt(self.cached('stocks', 'prices', ticker)[0])
        )

    def _get_prices(self):
        return dict(
            zip(
                self.tickers,
                map(flt, self.cached('stocks', 'prices', self.tickers))
            )
        )

    def _get_positions(self):
        data = self.cached('options', 'positions:all')

        activities = defaultdict(list)
        for option in [o for o in data if flt(o['quantity']) != 0]:
            ticker = option['chain_symbol']

            uri = option['option']
            instrument = self.cached('stocks', 'instrument', uri)
            if instrument['state'] == 'expired':
                raise
            elif instrument['tradability'] == 'untradable':
                raise

            premium = 0
            if instrument['state'] != 'queued':
                #signum = -1 if option['type'] == 'short' else +1
                premium -= flt(option['quantity']) * flt(option['average_price'])

            activities[ticker].append("%s %s %s x%s K=%s X=%s P=%s" % (
                instrument['state'],
                option['type'],
                instrument['type'],
                qty(option['quantity'], 0),
                mulla(instrument['strike_price']),
                instrument['expiration_date'],
                mulla(premium),
            ))

        premiums = defaultdict(lambda: 0)
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
            premium = 0
            for leg in option['legs']:
                uri = leg['option']
                instrument = self.cached('stocks', 'instrument', uri)

                if ticker in DEBUG:
                    dprint(instrument, title=f'stocks:instrument({uri})')

                legs.append('%s to %s K=%s X=%s' % (
                    leg['side'],
                    leg['position_effect'],
                    mulla(instrument['strike_price']),
                    instrument['expiration_date'],
                ))

                premium += sum([
                    100 * flt(x['price']) * flt(x['quantity']) for x in leg['executions']
                ])

            premium *= -1 if option['direction'] == 'debit' else +1
            premiums[ticker] += premium

            activities[ticker].append("%s %s %s x%s P=%s K=%s" % (
                option['state'],
                option['type'],
                '/'.join(strategies),
                qty(option['quantity'], 0),
                mulla(premium),
                mulla(instrument['strike_price']),
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
                qty(order['quantity'], 0),
                mulla(order['price']),
            ))

        return dict(premiums=premiums, activities=activities)

    @connected
    def transactions(self):
        for stock in self.stockReader:
            yield stock.ticker, stock.parameters

    def xxxxxxx(self):
        stock, option = next(self.stockReader), next(self.optionReader)

        ticker, parameters = None, None
        while True:
            if stock and (not option or stock.timestamp < option.timestamp):
                parameters = stock.parameters
                ticker = stock.ticker
                stock = next(self.stockReader)
            elif option and (not stock or stock.timestamp >= option.timestamp):
                parameters = option.parameters
                ticker = option.ticker
                option = next(self.optionReader)
            else:
                break

            yield ticker, parameters

    @connected
    def _machine(self, area, subarea, *args):
        return ROBIN_STOCKS_API[area][subarea](*args)

    def human(self, area, subarea, *args):
        return dump(f'{area}:{subarea}', self._machine(area, subarea, *args))

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
                    fn = getattr(iex_stocks.Stock(args[0]), endpoint.function)
                    data = fn(*args[1:], **kwargs)
                else:
                    raise
            else:
                data = endpoint.function(*args, **kwargs)

            arguments = [
                ','.join(map(str, args)),
                ','.join(['%s=%s' % (k, v) for k, v in kwargs]),
            ]
            dprint("Cache fault on %s:%s(%s)" % (
                area, subarea, ','.join(arguments),
            ))
            pickle.dump(data, open(cachefile, 'wb'))

        return data

    def cached(self, area, subarea, *args, **kwargs):
        endpoint = ROBIN_STOCKS_API[area][subarea]
        uniqname = '-'.join([
            area,
            subarea,
            hashlib.sha1(','.join(map(str, args)).encode()).hexdigest(),
        ])
        cachefile = pathlib.Path(f'{CACHE_DIR}/{uniqname}.pkl')

        data, age, fresh, this_second = {}, -1, False, datetime.now()
        while not fresh:
            data = self._pickled(cachefile, area, subarea, *args, **kwargs)

            then = datetime.fromtimestamp(cachefile.lstat().st_mtime)
            age = this_second - then

            jitter = 1 + random.random()
            fresh = (endpoint.ttl == -1) or age.total_seconds() < endpoint.ttl * jitter
            if not fresh:
                cachefile.unlink()

        return data

RobinStocksEndpoint = namedtuple('RobinStocksEndpoint', ['ttl', 'function'])
PolygonEndpoint = namedtuple('PolygonEndpoint', ['ttl', 'function'])
IEXFinanceEndpoint = namedtuple('PolygonEndpoint', ['ttl', 'function'])
ROBIN_STOCKS_API = {
    'profiles': {
        'account'     : RobinStocksEndpoint(7200, rh.profiles.load_account_profile),
        'investment'  : RobinStocksEndpoint(7200, rh.profiles.load_investment_profile),
        'portfolio'   : RobinStocksEndpoint(7200, rh.profiles.load_portfolio_profile),
        'security'    : RobinStocksEndpoint(7200, rh.profiles.load_security_profile),
        'user'        : RobinStocksEndpoint(7200, rh.profiles.load_user_profile),
    },
    'stocks': {
        'earning'     : RobinStocksEndpoint(7200, rh.stocks.get_earnings),
        'events'      : RobinStocksEndpoint(7200, rh.stocks.get_events),
        'fundamentals': RobinStocksEndpoint(7200, rh.stocks.get_fundamentals),
        'instruments' : RobinStocksEndpoint(7200, rh.stocks.get_instruments_by_symbols),
        'instrument'  : RobinStocksEndpoint(7200, rh.stocks.get_instrument_by_url),
        'prices'      : RobinStocksEndpoint(7200, rh.stocks.get_latest_price),
        'news'        : RobinStocksEndpoint(7200, rh.stocks.get_news),
        'quotes'      : RobinStocksEndpoint(7200, rh.stocks.get_quotes),
        'ratings'     : RobinStocksEndpoint(7200, rh.stocks.get_ratings),
       #'splits'      : PolygonEndpoint(-1, 'reference_stock_splits'),
        'splits'      : IEXFinanceEndpoint(86400, 'get_splits'),
        'historicals' : RobinStocksEndpoint(7200, rh.stocks.get_stock_historicals),
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
        'stock'          : RobinStocksEndpoint(3*3600, rh.export_completed_stock_orders),
        'option'         : RobinStocksEndpoint(3*3600, rh.export_completed_option_orders),
    },
}

DEBUG = []
def dprint(data, title=None):
    if len(DEBUG) == 0: return

    title = '%s(%d entries)' % (f'{title} ' if title else '', len(data))
    print(
        dump(
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
            'ticker',
            'price', 'quantity', 'equity', 'percentage',
            'cost', 'cost_basis',
            'short', 'bucket', 'rank',
            'equity_change', 'percent_change',
            'today', 'day', 'year',
            'premium_collected', 'dividends_collected',
            'activities',
        ],
    },
    'tax': {
        'sort_by': 'ticker',
        'columns': [
            'ticker',
            'price',
            'quantity',
            'equity',
            'cost', 'cost_basis',
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
        p = datum['price']
        l = flt(fundamentals[ticker]['low'])
        h = flt(fundamentals[ticker]['high'])
        datum['day'] = 100 * (p - l) / h

        o = flt(fundamentals[ticker]['open'])
        datum['today'] = 100 * (p - o) / o

        l = flt(fundamentals[ticker]['low_52_weeks'])
        h = flt(fundamentals[ticker]['high_52_weeks'])
        datum['year'] = 100 * (p - l) / h

        eq = flt(datum['equity'])
        datum['bucket'] = 0
        datum['short'] = 0
        for i, b in enumerate([s * 1000 for s in buckets[1:]]):
            l, h = eq * 0.75, eq * 1.25
            if l < b < h:
                datum['bucket'] = b//1000
                datum['short'] = rnd((eq - b) / datum['price'], 0.25)
                break

        # Rank = sum(
        #     x for price down since open?
        #     x for the horta; how short are you from reaching assigned bucket size
        #     x for price as percentage from today's low to high
        #     x for inverse of price as percentage from year's low to high
        #     x for all-time percentage change (positive cost basis means more risk tolerance)
        magnify = lambda p, x: x*(p+(100/x))/200
        pctch = flt(datum['percent_change'])
        eq = flt(datum['equity'])
        datum['rank'] = sum([
            50 * (eq / (datum['bucket'] * 1000) if datum['bucket'] > 0 else 0),
            30 * magnify(datum['today'], 4),  # is stock down since open?
            15 * magnify(datum['day'], 4),    # is stock down compared to low/high points, today?
             5 * (pctch / 100)
        ])

    #if DEBUG:
    #    for o in [o for o in account.cached('account', 'positions:all')]:
    #        dprint(o, title='account:positions:all')
    #    for o in [o for o in account.cached('options', 'positions:agg')]:
    #        dprint(o, title='options.positions:agg')

    formats = {
        'bucket': lambda b: colored('$%dk' % b, 'blue'),
        'today': pct,
        'day': pct,
        'year': pct,
        'price': mulla,
        'quantity': qty,
        'average_buy_price': mulla,
        'equity': mulla,
        'percent_change': pct,
        'equity_change': mulla,
        'pe_ratio': qty,
        'percentage': pct,
        'rank': int,
        'delta': qty,
        'premium_collected': mulla,
        'dividends_collected': mulla,
        'st_cb_qty': qty,
        'st_cb_capgain': mulla,
        'lt_cb_qty': qty,
        'lt_cb_capgain': mulla,
        'cost': mulla,
        'cost_basis': mulla,
    }

    table = BeautifulTable(maxwidth=300)
    table.set_style(BeautifulTable.STYLE_GRID)
    table.columns.header = [h.replace('_', '\n') for h in VIEWS[view]['columns']]

    if 'activate' in VIEWS[view]['columns']:
        table.columns.alignment['activities'] = BeautifulTable.ALIGN_LEFT

    for datum in account.data.values():
        f = fmt(formats, datum)
        table.rows.append([f(h) for h in VIEWS[view]['columns']])

    table.rows.sort(VIEWS[view]['sort_by'], reverse)

    if DEBUG:
        print(table.rows.filter(lambda row: row['ticker'] in DEBUG))
    else:
        print(table.rows[:limit] if limit > -1 else table)


@cli.command(help='Account History')
@click.pass_context
def history(ctx):
    account = ctx.obj['account']
    account.slurp()
    for ticker, stock in account.portfolio.items():
        if len(DEBUG) > 0 and ticker not in DEBUG: continue
        stock.summarize()

if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    rh.helper.set_output(open(os.devnull,"w"))
    cli(obj={'account': Account()})
