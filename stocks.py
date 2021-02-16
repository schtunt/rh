from collections import defaultdict
from functools import reduce

import util
from util.numbers import dec as D
from constants import ZERO as Z

import cachier
import datetime

import api
import lots
import events

class Stock:
    #todo move this to api and make dynamic
    mappings = {
        'FCAU': 'STLA'
    }

    def __init__(self, account, ticker):
        self.account = account
        self.ticker = Stock.mappings.get(ticker, ticker)

        self._quantity = 0
        self._ledger = []

        self.lots = []

        self.pointer = 0
        self.events = []

        self.splits = [
            events.StockSplitEvent(
                stock=self,
                date=ss['exDate'],
                divisor=ss['fromFactor'],
                multiplier=ss['toFactor'],
            ) for ss in api.splits(self.ticker)
        ]

        # StockSplits - These splits take effect whenever the query date and the buy date
        # lie on different sides of the split date.  That means, that the buy event needs to
        # know "who's asking", or more specifically, "when's asking?", in order to answer
        # correctly.
        self._pool = self.splits[:]


    def __getitem__(self, i):
        return self.events[i]

    def __repr__(self):
        return '<Stock %s - %s x %s @ ESP of %s>' % (
            self.ticker,
            util.color.qty(self._quantity),
            util.color.mulla(self.price),
            util.color.mulla(self.esp()),
        )

    @property
    def earnings(self):
        return api.earnings(self.ticker)

    @property
    def yesterday(self):
        return api.yesterday(self.ticker)

    @property
    def marketcap(self):
        return api.marketcap(self.ticker)

    @property
    def stats(self):
        return api.stats(self.ticker)

    @property
    def beta(self):
        return api.beta(self.ticker)

    @property
    def news(self):
        return [
            '. '.join((blb['preview_text'], blb['title'])) for blb in api.news(self.ticker, 'rh')
        ] + [
            '. '.join([blb['headline'], blb['summary']]) for blb in api.news(self.ticker, 'iex')
        ]


    @property
    def _analyst_sentiments(self):
        '''
        Convert the analyst sentiment series on the stock to a score, using extremely
        questionable mathematics.
        '''

        def series():
            '''
            Looking at historic analyst sentiments, and exagerating the strong opinions, creates
            a time series, to show if those sentiments have been growing, or shrinking with time.
            '''
            series = dict(buy=[], hold=[], sell=[])
            for r in reversed(self.recommendations):
                b = r['buy'] + pow(r['strongBuy'], 2)
                s = r['sell'] + pow(r['strongSell'], 2)
                h = r['hold']
                t = sum((b, h, s))
                series['buy'].append(b/t)
                series['hold'].append(s/t)
                series['sell'].append(h/t)

            return series


        data = {
            'buy':  { 'risk': Z, 'mean': Z, 'std': Z },
            'sell': { 'risk': Z, 'mean': Z, 'std': Z },
            'hold': { 'risk': Z, 'mean': Z, 'std': Z },
        }

        ss = util.numbers.growth_score
        for key, datum in series().items():
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

    @property
    def _news_sentiments(self):
        result = api.sentiments(self.news)

        # filter out unnecessary data
        data = [e for ee in map(lambda e: e['classifications'], result) for e in ee]

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
            'news': self._news_sentiments,
            'analysts': self._analyst_sentiments,
        }

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
        return api.price(self.ticker)

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

        transactions = self.transactions
        current, last = next(transactions), None
        try:
            while True:
                last, current = current, next(transactions)
        except StopIteration:
            pass

        assert current is not None

        if util.datetime.delta(current.timestamp) <= util.datetime.A_MONTH: return True
        if last and util.datetime.delta(last.timestamp, current.timestamp) <= util.datetime.A_MONTH: return True
        return False

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

                term = util.finance.conterm(buy.timestamp, when)
                available = buy.available(when=when)
                price = buy.price(when=when)
                costbasis[term]['qty'] += available
                costbasis[term]['value'] += available * (self.price - price)

                #util.output.ddump([when, buy.timestamp, available, self.price, -price], title="stock.costbasis")

        return costbasis

    def ledger(self, transaction):
        self._pool.append(transaction)
        self._pool.sort(key=lambda e: e.timestamp)
        event = None
        while event is not transaction:
            event = self._pool.pop(0)

            if type(event) is events.TransactionEvent:
                if event.side == 'sell':
                    lot = lots.Lot(self, event)
                    self.lots.append(lot)
                    self._quantity -= event.quantity()
                else:
                    self._quantity += event.quantity()
                self.events.append(event)
            elif type(event) is events.StockSplitEvent and self._quantity != Z:
                self._quantity = event.forward(self._quantity)
                self.events.append(event)

            # This is used only for unit testing, bit expensive
            now = util.datetime.now()
            cbr = self.costbasis(realized=True, when=now)
            cbu = self.costbasis(realized=False, when=now)

            self._ledger.append({
                'cnt':  len(self.events),
                'dts':  transaction.timestamp,
                'trd':  self.traded(when=now),
                'qty':  self._quantity,
                'ptr':  self.pointer,
                'esp':  self.esp(when=now), # Effective Share Price
                'crsq': cbr['short']['qty'],
                'crsv': cbr['short']['value'],
                'crlq': cbr['long']['qty'],
                'crlv': cbr['long']['value'],
                'cusq': cbu['short']['qty'],
                'cusv': cbu['short']['value'],
                'culq': cbu['long']['qty'],
                'culv': cbu['long']['value'],
            })

        # This following data will end up in the DataFrame and saved
        now = util.datetime.now()
        cbr = self.costbasis(realized=True, when=now)
        cbu = self.costbasis(realized=False, when=now)
        return {
            'cnt':  len(self.events),
            'dts':  transaction.timestamp,
            'trd':  self.traded(when=now),
            'qty':  self._quantity,
            'ptr':  self.pointer,
            'esp':  self.esp(when=now),
            'crsq': cbr['short']['qty'],
            'crsv': cbr['short']['value'],
            'crlq': cbr['long']['qty'],
            'crlv': cbr['long']['value'],
            'cusq': cbu['short']['qty'],
            'cusv': cbu['short']['value'],
            'culq': cbu['long']['qty'],
            'culv': cbu['long']['value'],
        }

    def esp(self, when=None, dividends=False, premiums=False):
        '''
        On average, how much has each stock cost/made for the investor, per share ever traded,
        by that invester.  This number is calculated by taking into account everything that has
        happened to stocks held by the investor - capital gains, capital losses, dividends, and
        premiums collected and forfeited on the option market.
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
            divs = api.dividends()[self.ticker]
            value += sum(D(div['amount']) for div in divs)

        if premiums:
            value += self.account.positions['premiums'][self.ticker]

        return value / qty if qty > 0 else 0

    def traded(self, when=None):
        return sum(map(lambda buy: buy.quantity(when=when), self.buys))

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
