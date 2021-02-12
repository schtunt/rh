#!/usr/bin/env python3

import os, sys, locale
import csv, requests, tempfile
import click, pickle, json
from collections import defaultdict, namedtuple

import math, hashlib, random
from monkeylearn import MonkeyLearn
from functools import reduce

from datetime import datetime, timedelta, timezone

import pathlib
from pathlib import posixpath

import robin_stocks as rh
import yahoo_earnings_calendar as yec
import iexfinance.stocks as iex
import polygon
import finnhub

from beautifultable import BeautifulTable

import __main__

import constants
from constants import ZERO as Z

import events

import util
from util.numbers import dec as D
from util.output import ansistrip as S
DS = lambda s: D(S(s))

def conterm(fr, to=None):
    to = to if to is not None else util.datetime.now()
    delta = to - fr
    return 'short' if delta <= timedelta(days=365, seconds=0) else 'long'

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
                    events.TransactionEvent(
                        stock,
                        required,
                        0.00,
                        'buy',
                        str(datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)),
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
            'short': { 'qty': 0, 'value': 0 },
            'long':  { 'qty': 0, 'value': 0 },
        }

        for lc in self.buys:
            costbasis[lc.term]['qty'] += lc.quantity(when=when)
            costbasis[lc.term]['value'] += lc.costbasis(when=when)

        #util.output.ddump(costbasis, force=True)

        return costbasis

class LotConnector:
    def __init__(self, stock, sell, buy, requesting):
        self.stock = stock
        self.sell  = sell
        self.buy = buy

        self._quantity = min(requesting, buy.available())

        # short-term or long-term sale (was the buy held over a year)
        self.term = conterm(self.buy.timestamp, self.sell.timestamp)

        self.buy.connect(self)
        self.sell.connect(self)

    def quantity(self, when=None):
        quantity = self._quantity
        if when is None: return quantity

        for splitter in self.stock.splitters:
            if self.timestamp <= splitter.timestamp <= when:
                quantity = splitter.forward(quantity)

        return quantity

    def costbasis(self, when=None):
        return (self.sell.price(when=when) - self.buy.price(when=when)) * self.quantity(when=when)

    @property
    def timestamp(self):
        return self.sell.timestamp


class StockFIFO:
    mappings = {
        'FCAU': 'STLA'
    }

    def __init__(self, account, ticker):
        self.account = account
        self.ticker = StockFIFO.mappings.get(ticker, ticker)

        self._quantity = 0

        self.pointer = 0
        self.events = []

        self._ledger = []

        self.lots = []

        # StockSplits - These splitters will act as identity functions, but take effect whenever
        # the date of query and the date of a buy lie on different sides of the split date.  That
        # means, that the buy event needs to know "who's asking", or more specifically,
        # "when's asking?".
        self.splitters = [
            events.StockSplitEvent(
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
            util.color.qty(self._quantity),
            util.color.mulla(self.epst()),
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

    def _sentiment_series(self):
        '''
        Looking at historic analyst sentiments, and exagerating the strong opinions, creates
        a time series, to show if those sentiments have been growing, or shrinking with time.
        '''
        series = dict(buy=[], hold=[], sell=[])
        for r in reversed(self.account.cached('stocks', 'recommendations', self.ticker)):
            b = r['buy'] + pow(r['strongBuy'], 2)
            s = r['sell'] + pow(r['strongSell'], 2)
            h = r['hold']
            t = sum((b, h, s))
            series['buy'].append(b/t)
            series['hold'].append(s/t)
            series['sell'].append(h/t)

        return series


    def _analyst_sentiments(self):
        '''
        Convert the analyst sentiment series on the stock to a score, using extremely
        questionable mathematics.
        '''

        data = {
            'buy':  { 'risk': Z, 'mean': Z, 'std': Z },
            'sell': { 'risk': Z, 'mean': Z, 'std': Z },
            'hold': { 'risk': Z, 'mean': Z, 'std': Z },
        }

        ss = util.numbers.growth_score
        series = self._sentiment_series()
        for key, datum in series.items():
            numerator, denominator = ss(datum), ss(sorted(datum, reverse=True))
            if denominator > 0:
                data[key] = {
                    'risk': numerator / denominator,
                    'mean': util.numbers.mean(datum),
                    'std': util.numbers.std(datum),
                }

        # if the analysts rating increases in time perfectly, then the risk score will be zero,
        # and so the score will simply be the mean of their votes over time.  In the worst case,
        # however, the score will be the mean minus one standard deviation.
        mkscore = lambda key: data[key]['mean'] - data[key]['std'] * data[key]['risk']
        sentiments = dict(buy=mkscore('buy'), sell=mkscore('sell'), hold=mkscore('hold'))

        return sentiments

    def _news_sentiments(self):
        news1 = self.account.cached('stocks', 'news1', self.ticker)
        news = ['. '.join((blb['preview_text'], blb['title'])) for blb in news1]

        news2 = self.account.cached('stocks', 'news2', self.ticker)
        news += ['. '.join([blb['headline'], blb['summary']]) for blb in news2]

        result = self.account.cached('classifiers', 'sentiment', news=news)

        # filter out unnecessary data
        data = [e for ee in map(lambda e: e['classifications'], result.body) for e in ee]

        def l(e, ee):
             e[ee['tag_name']] += D(ee['confidence'])
             return e

        # aggregate the remaining serntiment data
        aggregate = reduce(l, data, defaultdict(lambda: Z))
        total = sum(aggregate.values())

        sentiments = {
            'Positive': Z,
            'Negative': Z,
            'Neutral':  Z,
        }

        sentiments.update({
            sentiment: confidence / total for sentiment, confidence in aggregate.items()
        })

        return sentiments

    def sentiments(self):
        return {
            'news': self._news_sentiments(),
            'analysts': self._analyst_sentiments(),
        }

    @property
    def ttm(self):
        return dict(
            eps=self.stats['ttmEPS'],
            div=self.stats['ttmDividendRate'],
        )

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
        return self._quantity * self.price

    @property
    def gain(self):
        return self.equity - self.cost

    @property
    def transactions(self, reverse=False):
        return filter(lambda e: type(e) is events.TransactionEvent, self.events)

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
            events.TransactionEvent(
                self,
                ec['quantity'],
                ec['price'],
                ec['side'],
                se['updated_at'],
                se['type'],
            ) for se in self.account.cached('events', 'activities', self.ticker)
                for ec in se['equity_components']
        ]

    @property
    def recommendatios(self):
        return self.account.cached('stocks', 'recommendation_trends', self.ticker)

    @property
    def target(self):
        return self.account.cached('stocks', 'price_target', self.ticker)

    @property
    def quote(self):
        return self.account.cached('stocks', 'quote', self.ticker)

    def quantity(self, when=None):
        cb = self.costbasis(realized=False, when=when)
        return cb['short']['qty'] + cb['long']['qty']

    def costbasis(self, realized=True, when=None):
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
                lcb = lot.costbasis(when=when)
                for term in costbasis.keys():
                    costbasis[term]['qty'] += lcb[term]['qty']
                    costbasis[term]['value'] += lcb[term]['value']
        else:
            for buy in self.buys:
                if when is not None and when < buy.timestamp: break

                term = conterm(buy.timestamp, when)
                available = buy.available(when=when)
                price = buy.price(when=when)
                costbasis[term]['qty'] += available
                costbasis[term]['value'] += available * (self.price - price)

                #util.output.ddump([when, buy.timestamp, available, self.price, -price], title="stock.costbasis")

        return costbasis

    def push(self, transaction):
        self._event_pool.append(transaction)
        self._event_pool.sort(key=lambda e: e.timestamp)
        event = None
        while event is not transaction:
            event = self._event_pool.pop(0)

            if type(event) is events.TransactionEvent:
                if event.side == 'sell':
                    lot = Lot(self, event)
                    self.lots.append(lot)
                    self._quantity -= event.quantity()
                else:
                    self._quantity += event.quantity()
            elif type(event) is events.StockSplitEvent:
                self._quantity = event.forward(self._quantity)

            self.events.append(event)

            now = util.datetime.now()

            # Ledger for unit-testing
            cbr = self.costbasis(realized=True, when=now)
            cbu = self.costbasis(realized=False, when=now)
            self._ledger.append({
                'cnt':  len(self.events),
                'dts':  transaction.timestamp,
                'trd':  self.traded(when=now),
                'qty':  self._quantity,
                'ptr':  self.pointer,
                'epst': self.epst(when=now),
                'crsq': cbr['short']['qty'],
                'crsv': cbr['short']['value'],
                'crlq': cbr['long']['qty'],
                'crlv': cbr['long']['value'],
                'cusq': cbu['short']['qty'],
                'cusv': cbu['short']['value'],
                'culq': cbu['long']['qty'],
                'culv': cbu['long']['value'],
            })

    def epst(self, when=None, dividends=False, premiums=False):
        '''
        On average, how much has each stock earned *you* the investor, per share ever traded,
        after taking into account everything that has happened to stocks held by you - capital
        gains, capital losses, dividends, and premiums collected and forfeited on options.
        '''
        realized = self.costbasis(realized=True, when=when)
        unrealized = self.costbasis(realized=False, when=when)

        #util.output.ddump({
        #    't': when,
        #    'r': realized,
        #    'u': unrealized,
        #    'cb': [
        #        costbasis[term]['value']
        #        for costbasis in (realized, unrealized)
        #        for term in ('short', 'long')
        #    ],
        #}, force=True)

        value, qty = Z, Z
        for costbasis in (realized, unrealized):
            for term in ('short', 'long'):
                value += costbasis[term]['value']
                qty += costbasis[term]['qty']

        if dividends:
            divs = self.account._get_dividends()[self.ticker]
            value += sum(D(div['amount']) for div in divs)

        if premiums:
            value += self.account._get_positions()['premiums'][self.ticker]

        return value / qty if qty > 0 else 0

    def traded(self, when=None):
        traded = Z
        for buy in self.buys:
            traded += buy.quantity(when=when)
        return traded

    def summarize(self):
        print('#' * 80)

        for lot in self.lots:
            for lc in lot.buys:
                print(lc.buy)
            print(lc.sell, "<future-event>")
            print()

        # Remaining buys (without a corresponding sell; current equity)
        for event in self.events[self.pointer:]:
            if type(event) is events.TransactionEvent:
                print(event)
            else:
                print()
                print(event)
                print()

        #print("Cost : %s x %s = %s" % (util.color.qty(self._quantity), util.color.mulla(self.epst()), util.color.mulla(self.cost)))
        #print("Value: %s x %s = %s" % (util.color.qty(self._quantity), util.color.mulla(self.price), util.color.mulla(self.equity)))
        #print("Capital Gains: %s" % (util.color.mulla(self.gain)))


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
        self.finnhub_api_key = None
        self.finnhub = None
        self.monkeylearn_api = None
        self.ml = None

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
            (
                username, password,
                self.polygon_api_key,
                self.iex_api_key,
                self.finnhub_api_key,
                self.monkeylearn_api,
            ) = [ token.strip() for token in fh.readline().split(',') ]

            if self.robinhood is None:
                self.robinhood = rh.login(username, password)

            os.environ['IEX_TOKEN'] = self.iex_api_key
            os.environ['IEX_OUTPUT_FORMAT'] = 'json'

            self.finnhub = finnhub.Client(api_key=self.finnhub_api_key)
            self.ml = MonkeyLearn(self.monkeylearn_api)

    def slurp(self):
        for ticker, parameters in self.transactions():
            stock = self.get_stock(ticker)
            transaction = events.TransactionEvent(stock, *parameters)
            stock.push(transaction)

        self.data = self.cached('account', 'holdings')
        self.tickers = sorted(self.data.keys())

        now = util.datetime.now()
        for ticker in self.tickers:
            stock = self.get_stock(ticker)
            self.data[ticker].update({
                'realized': stock.costbasis(realized=True, when=now),
                'unrealized': stock.costbasis(realized=False, when=now),
            })

        positions = self._get_positions()
        dividends = self._get_dividends()
        prices = self._get_prices()
        fundamentals = self._get_fundamentals()

        for ticker, datum in self.data.items():
            fifo = self.get_stock(ticker)

            datum['ticker'] = ticker
            datum['premium_collected'] = positions['premiums'][ticker]
            datum['dividends_collected'] = sum(D(div['amount']) for div in dividends[ticker])
            datum['activities'] = '\n'.join(positions['activities'].get(ticker, []))
            datum['pe_ratio'] = D(fundamentals[ticker]['pe_ratio'])
            datum['pb_ratio'] = D(fundamentals[ticker]['pb_ratio'])

            datum['collateral'] = positions['collateral'][ticker]

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

        collateral = defaultdict(lambda: {'put': Z, 'call': Z})
        activities = defaultdict(list)
        for option in [o for o in data if D(o['quantity']) != Z]:
            ticker = option['chain_symbol']

            uri = option['option']
            instrument = self.cached('stocks', 'instrument', uri)
            if instrument['state'] == 'expired':
                pass
            elif instrument['tradability'] == 'untradable':
                raise

            premium = Z
            if instrument['state'] != 'queued':
                premium -= D(option['quantity']) * D(option['average_price'])

            itype = instrument['type']
            activities[ticker].append("%s %s %s x%s P=%s K=%s X=%s" % (
                instrument['state'],
                option['type'],
                itype,
                util.color.qty(option['quantity'], Z),
                util.color.mulla(premium),
                util.color.mulla(instrument['strike_price']),
                instrument['expiration_date'],
            ))

            collateral[ticker][itype] += 100 * D(dict(
                put=instrument['strike_price'],
                call=option['quantity']
            )[itype])

        premiums = defaultdict(lambda: Z)
        data = self.cached('orders', 'options:all')
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
                instrument = self.cached('stocks', 'instrument', uri)

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

        return dict(
            premiums=premiums,
            activities=activities,
            collateral=collateral,
        )

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
        return util.output.ddump(f'{area}:{subarea}', self._machine(area, subarea, *args))

    @connected
    def _pickled(self, cachefile, area, subarea, *args, **kwargs):
        if cachefile.exists():
            return pickle.load(open(cachefile, 'rb'))

        with util.output.progress(
            "Cache fault on %s:%s(%d x args, %d x kwargs)" % (area, subarea, len(args), len(kwargs))
        ):
            endpoint = ROBIN_STOCKS_API[area][subarea]
            if type(endpoint) is FinnhubEndpoint:
                fn = getattr(self.finnhub, endpoint.function)
                data = fn(*args, **kwargs)
            elif type(endpoint) is PolygonEndpoint:
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
            elif type(endpoint) is MonkeyLearnEndpoint:
                if area == 'classifiers':
                    model_id = endpoint.model_id
                    news = kwargs['news']
                    data = self.ml.classifiers.classify(model_id, news)
                else:
                    raise
            else:
                data = endpoint.function(*args, **kwargs)

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                util.output.ddump(f"Generating {cachefile} from {tmp.name}")
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
FinnhubEndpoint = namedtuple('FinnhubEndpoint', ['ttl', 'function'])
MonkeyLearnEndpoint = namedtuple('MonkeyLearnEndpoint', ['ttl', 'model_id'])

ROBIN_STOCKS_API = {
    'classifiers': {
        'sentiment'      : MonkeyLearnEndpoint(14400, 'cl_pi3C7JiL')
    },
    'profiles': {
        'account'        : RobinStocksEndpoint(7200, rh.profiles.load_account_profile),
        'investment'     : RobinStocksEndpoint(7200, rh.profiles.load_investment_profile),
        'portfolio'      : RobinStocksEndpoint(7200, rh.profiles.load_portfolio_profile),
        'security'       : RobinStocksEndpoint(7200, rh.profiles.load_security_profile),
        'user'           : RobinStocksEndpoint(7200, rh.profiles.load_user_profile),
    },
    'stocks': {
        'fundamentals'   : RobinStocksEndpoint(7200,  rh.stocks.get_fundamentals),
        'instruments'    : RobinStocksEndpoint(7200,  rh.stocks.get_instruments_by_symbols),
        'instrument'     : RobinStocksEndpoint(7200,  rh.stocks.get_instrument_by_url),
        'prices'         : RobinStocksEndpoint(7200,  rh.stocks.get_latest_price),
        'target'         : FinnhubEndpoint(7200,      'price_target'),
        'recommendations': FinnhubEndpoint(7200,      'recommendation_trends'),
        'news1'          : RobinStocksEndpoint(60,    rh.stocks.get_news),
        'news2'          : IEXFinanceEndpoint(60,     'get_news'),
        'quote'          : IEXFinanceEndpoint(7200,   'get_quote'),
        'quotes'         : RobinStocksEndpoint(7200,  rh.stocks.get_quotes),
        'ratings'        : RobinStocksEndpoint(7200,  rh.stocks.get_ratings),
        'splits'         : IEXFinanceEndpoint(86400,  'get_splits'),
        'yesterday'      : IEXFinanceEndpoint(21600,  'get_previous_day_prices'),
        'marketcap'      : IEXFinanceEndpoint(86400,  'get_market_cap'),
        'losers'         : IEXFinanceEndpoint(300,    'get_market_losers'),
        'gainers'        : IEXFinanceEndpoint(300,    'get_market_gainers'),
        'stats'          : IEXFinanceEndpoint(21600,  'get_key_stats'),
        'historicals'    : RobinStocksEndpoint(7200,  rh.stocks.get_stock_historicals),
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


@click.group()
@click.option('-D', '--debug-tickers',multiple=True,  default=None)
@click.pass_context
def cli(ctx, debug_tickers):
    constants.DEBUG=debug_tickers
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

FILTERS = {
    'active': lambda d: len(d['activities']),
    'optionable': lambda d: DS(d['quantity']) - DS(d['CC.Coll']) > 100,
}

VIEWS = {
    'pie': {
        'sort_by': 'rank',
        'columns': [
            'ticker', 'percentage',
            'quantity', 'price',
            'epst', 'epst%',
            'rank', 'analyst', 'news', 'ma',
            'equity', 'equity_change', 'percent_change',
            'premium_collected', 'dividends_collected',
            'short', 'bucket',
            'rank', 'ma', 'since_close', 'since_open', 'alerts',
            'pe_ratio', 'pb_ratio', 'beta',
            'CC.Coll', 'CSP.Coll',
            'activities',
        ],
    },
    'losers': {
        'sort_by': 'rank',
        'columns': [
            'ticker',
            'marketcap',
            'rank', 'analyst', 'news', 'ma', 'epst%',
            'alerts',
            'pe_ratio', 'pb_ratio', 'beta',
            'equity', 'equity_change', 'percent_change',
            'premium_collected', 'dividends_collected',
            'crv', 'cuv',
            'CC.Coll', 'CSP.Coll',
            'activities',
        ],
    },
     'gen': {
        'sort_by': 'CC.Coll',
        'filter_by': 'optionable',
        'columns': [
            'ticker', 'percentage',
            'quantity', 'price',
            'epst', 'epst%',
            'equity', 'equity_change', 'percent_change',
            'premium_collected', 'dividends_collected',
            'rank', 'ma', 'since_close', 'since_open', 'alerts',
            'pe_ratio', 'pb_ratio', 'beta',
            'CC.Coll', 'CSP.Coll',
            'activities',
        ],
    },
    'active': {
        'sort_by': 'ticker',
        'filter_by': 'active',
        'columns': [
            'ticker', 'percentage',
            'price', 'quantity',
            'equity',
            'equity_change', 'percent_change',
            'premium_collected', 'dividends_collected',
            'activities',
        ],
    },
    'tax': {
        'sort_by': 'ticker',
        'columns': [
            'ticker',
            'equity',
            'price', 'quantity',
            'cusq', 'cusv', 'culq', 'culv',
            'crsq', 'crsv', 'crlq', 'crlv',
            'premium_collected', 'dividends_collected',
        ],
    },
}

@cli.command(help='Views')
@click.option('-v', '--view', default='pie', type=click.Choice(VIEWS.keys()))
@click.option('-s', '--sort-by', default=False, type=str)
@click.option('-r', '--reverse', default=False, is_flag=True, type=bool)
@click.option('-l', '--limit', default=-1, type=int)
@click.pass_context
def tabulize(ctx, view, sort_by, reverse, limit):
    account = ctx.obj['account']
    account.slurp()

    buckets = [ 0, 3, 5, 8, 13, 21, 34 ]
    pie = { k: [] for k in buckets }

    fundamentals = account._get_fundamentals()
    for ticker, datum in account.data.items():
        stock = account.get_stock(ticker)
        index = stock.pointer
        buy = stock[index]
        assert buy.side == 'buy'

        def alerts(marketcap):
            alerts = []
            if marketcap is not None:
                marketcap /= 1000000000
                if marketcap > 10: sizestr = util.color.colored('L', 'green')
                elif marketcap > 2:
                    if 4 < marketcap < 5:
                        sizestr = util.color.colored('M', 'magenta')
                    else:
                        sizestr = util.color.colored('M', 'blue')
                else: sizestr = util.color.colored('S', 'yellow')
            else:
                marketcap, sizestr = Z, util.color.colored('U', 'red')
            alerts.append('%s/%sB' % (sizestr, util.color.mulla(marketcap)))

            datum['marketcap'] = marketcap

            #if buy.term == 'st': alerts.append(util.color.colored('ST!', 'yellow'))
            if stock.subject2washsale: alerts.append(util.color.colored('WS!', 'yellow'))
            if datum['pe_ratio'].is_nan() or datum['pe_ratio'] < 10:
                alerts.append(util.color.colored('PE<10', 'red'))

            return alerts

        marketcap = datum['marketcap']
        datum['alerts'] = ' '.join(alerts(marketcap))

        # MA: deviation from 0 means deviation from the worst-case scenario
        # TODO: formatter function can't take arguments at this time
        prices = (datum['price'], datum['50dma'], datum['200dma'])
        p, p50, p200 = prices
        #mac = 'yellow'
        #if p > p50 > p200: mac = 'green'
        #elif p < p50 < p200: mac = 'red'
        #ma_score_color = lambda ma: util.color.colored('%0.3f' % ma, mac)
        datum['ma'] = util.numbers.growth_score(prices)

        sentiments = stock.sentiments()
        datum['analyst'] = Z
        if sentiments['analysts']:
            scores = sentiments['analysts']
            datum['analyst'] += scores['buy'] - scores['sell']

        scores = sentiments['news']
        datum['news'] = scores['Positive'] - scores['Negative']

        cbu = stock.costbasis(realized=False)
        datum['cusq'] = cbu['short']['qty']
        datum['cusv'] = cbu['short']['value']
        datum['culq'] = cbu['long']['qty']
        datum['culv'] = cbu['long']['value']

        cbr = stock.costbasis(realized=True)
        datum['crsq'] = cbr['short']['qty']
        datum['crsv'] = cbr['short']['value']
        datum['crlq'] = cbr['long']['qty']
        datum['crlv'] = cbr['long']['value']

        datum['crv'] = datum['crsv'] + datum['crlv']
        datum['crq'] = datum['crsq'] + datum['crlq']
        datum['cuv'] = datum['cusv'] + datum['culv']
        datum['cuq'] = datum['cusq'] + datum['culq']

        datum['epst'] = stock.epst(dividends=True, premiums=True)
        datum['epst%'] = datum['epst'] / p

        collateral = datum['collateral']
        datum['CC.Coll'] = collateral['call']
        datum['CSP.Coll'] = collateral['put']

        yesterday = stock.yesterday

        c = D(yesterday['close'])
        datum['since_close'] = (p - c) / c if c > Z else Z

        o = D(fundamentals[ticker]['open'])
        datum['since_open'] = (p - o) / o

        datum['beta'] = stock.beta

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
                datum['short'] = util.numbers.rnd((eq - b) / datum['price'], round_to)
                break

        yesterchange = D(yesterday['marketChangeOverTime'])
        totalchange = D(datum['percent_change'])/100
        eq = D(datum['equity'])

        # Use 'view' to multiplex 'rank' here, for now just using this one for all
        scores = {
            'epst':          (15, util.numbers.scale_and_shift(datum['epst%'], D(1))),
            'ma':            (15, util.numbers.scale_and_shift(datum['ma'], D(1))),
            'analyst':       (15, util.numbers.scale_and_shift(datum['analyst'], D(1))),
            'news':          (15, util.numbers.scale_and_shift(datum['news'], D(1))),
            'yesterchange':  (10, util.numbers.scale_and_shift(yesterchange, D(4))),
            'since_close':   (10, util.numbers.scale_and_shift(datum['since_close'], D(4))),
            'since_open':    (10, util.numbers.scale_and_shift(datum['since_open'], D(4))),
            'totalchange':   (10, util.numbers.scale_and_shift(-totalchange, D(0.25))),
        }
        datum['rank'] = sum([pct * min(100, score) for (pct, score) in scores.values()])

        if ticker in constants.DEBUG:
            print(util.output.ddump('scores', scores))


    formats = {
        'bucket': lambda b: util.color.colored('$%dk' % b, 'blue'),
        'since_open': util.color.mpct,
        'since_close': util.color.mpct,
        'CC.Coll': util.color.qty0,                   # Covered Call Collateral
        'CSP.Coll': util.color.mulla,                 # Cash-Secured Put Collateral
        'price': util.color.mulla,
        'epst': util.color.mulla,                     # Earnings-Per-Share Traded
        'epst%': util.color.mpct,                     # Earnings-Per-Share Traded as % of current stock price
        'quantity': util.color.qty0,
        'marketcap': util.color.mulla,
        'average_buy_price': util.color.mulla,
        'equity': util.color.mulla,
        'percent_change': util.color.pct,
        'equity_change': util.color.mulla,
        'pe_ratio': util.color.qty,
        'pb_ratio': util.color.qty,
        'percentage': util.color.pct,
        'beta': util.color.qty,
        'rank': int,
        'delta': util.color.qty,
        'short': util.color.qty1,
        'premium_collected': util.color.mulla,
        'dividends_collected': util.color.mulla,
        'ma': util.color.qty,
        'cuq': util.color.qty,
        'cuv': util.color.mulla,
        'crq': util.color.qty,
        'crv': util.color.mulla,
        'cusq': util.color.qty,
        'cusv': util.color.mulla,
        'culq': util.color.qty,
        'culv': util.color.mulla,
        'crsq': util.color.qty,
        'crsv': util.color.mulla,
        'crlq': util.color.qty,
        'crlv': util.color.mulla,
        'analyst': util.color.qty,
        'news': util.color.qty,
    }

    data = account.data.values()
    table = mktable(data, view, formats, sort_by=sort_by, reverse=reverse, limit=limit)
    print(table)

    if constants.DEBUG:
        print(table.rows.filter(lambda row: row['ticker'] in constants.DEBUG))

def mktable(data, view, formats, maxwidth=320, sort_by=None, reverse=False, limit=None):
    columns = VIEWS[view]['columns']

    # 0. create
    table = BeautifulTable(maxwidth=maxwidth)

    # 1. configure
    table.set_style(BeautifulTable.STYLE_GRID)
    table.columns.header = [h.replace('_', '\n') for h in columns]
    if 'activate' in columns:
        table.columns.alignment['activities'] = BeautifulTable.ALIGN_LEFT

    # 2. populate
    for datum in data:
        table.rows.append(map(lambda k: datum.get(k, 'N/A'), columns))

    # 3. filter
    filter_by = VIEWS[view].get('filter_by', None)
    if filter_by is not None:
        table = table.rows.filter(FILTERS[filter_by])

    # 4. sort
    if not sort_by:
        sort_by = VIEWS[view].get('sort_by', 'ticker')
    table.rows.sort(key=sort_by, reverse=reverse)

    # 5. limit
    if limit > 0:
        table = table.rows[:limit] if not reverse else table.rows[-limit:]

    # 6. format
    for index, column in enumerate(columns):
        columns, fn = table.columns[index], formats.get(column, None)
        if fn is not None:
            columns = map(fn, columns)
            table.columns[index] = columns

    return table


@cli.command(help='Account History')
@click.pass_context
def history(ctx):
    account = ctx.obj['account']
    account.slurp()
    for ticker, stock in account.stocks.items():
        if len(constants.DEBUG) > 0 and ticker not in constants.DEBUG: continue
        stock.summarize()

def preinitialize(repl=False):
    locale.setlocale(locale.LC_ALL, '')
    rh.helper.set_output(open(os.devnull,"w"))

    if not posixpath.exists(constants.CACHE_DIR):
        os.mkdir(constants.CACHE_DIR)

acc = None
def interact():
    print("Preparing REPL...")
    preinitialize(repl=True)

    print("Initializing your Robinhood Account...")
    global acc
    acc = Account()
    acc.slurp()

    print("Injecting Stock objects for all stocks in your portfolio...")
    module = sys.modules[__name__]
    for ticker in acc.tickers:
        key = ticker.lower()
        setattr(module, key, acc.get_stock(ticker))
        setattr(module, '_%s' % key, iex.Stock(ticker))

    print("Done! Available ticker objects:")
    print(" + rh.acc           (Local Robinhood Account object)")
    print(" + rh.acc.rh        (RobinStocksEndpoint API)")
    print(" + rh.acc.iex       (IEXFinanceEndpoint API)")
    print(" + rh.acc.yec       (YahooEarningsCalendarEndpoint API)")
    print(" + rh.acc.finhubb   (FinnhubEndpoint API)")
    print(" + rh.acc.ml        (MonkeyLearnEndpoint API)")
    print(" + rh._<ticker>     (IEXFinanceEndpoint Stock object API)")
    print(" + rh.<ticker>      (Local Stock object API multiplexor")
    print()
    print("Meta-helpers for this REPL")
    print(" + relmod()         (reload wthout having to exit the repl)")

if __name__ == '__main__':
    preinitialize()
    cli(obj={'account': Account()})
elif not hasattr(__main__, '__file__'):
    interact()
