#!/usr/bin/env python3

import os, sys, locale
import csv, requests

from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import dateutil.parser as dtp
import robin_stocks as rh

mulla = lambda amount: locale.currency(amount, grouping=True)
flt = np.single
ZERO = 1e-5

class Lot:
    ident = 0

    def __init__(self, ticker, side, qty, price, timestamp, description):
        self.ident = Lot.ident
        Lot.ident += 1

        self.side = side
        self.ticker = ticker
        self.price = flt(price)
        self.qty = flt(qty)
        self.timestamp = timestamp
        self.ties = []
        self.description = description

    def __repr__(self):
        string = '#%05d %-4s:%4s[%20s] %8.2f x %10s = %10s @ %s' % (
            self.ident,
            self.ticker,
            self.side,
            self.description,
            self.qty,
            mulla(self.price),
            mulla(self.price * self.qty),
            self.timestamp,
        )

        return '<%s>' % string

    def tie(self, conn):
        self.ties.append(conn)

    @property
    def signum(self):
        return -1 if self.side == 'buy' else +1

    @property
    def unsettled(self):
        '''
        For a buy lot, this is how many remaining units are left in this lot.
        For a sell lot, this is how many outstanding units are unaccounted for.

        Different things, but the same underlying calculation.
        '''
        return self.qty - sum([conn.qty for conn in self.ties])

class LotConnector:
    @classmethod
    def settle(self, stock):
        index = stock.pointer
        fifo = stock.fifo
        sold = fifo[-1]
        assert sold.side == 'sell'

        qty = sold.qty
        while qty > ZERO:
            l = LotConnector(sold, stock, index)
            qty -= l.qty

            while index < len(fifo) and (
                fifo[index].side == 'sell' or fifo[index].unsettled <= ZERO
            ):
                index += 1

        stock.pointer = index

    def __init__(self, sold, stock, index):
        fifo = stock.fifo

        if len(fifo) == index:
            fifo.append(
                Lot(
                    stock.ticker,
                    'buy',
                    sold.unsettled,
                    0.00,
                    dtp.parse("1st Jan 2020, 12:00:00 am"),
                    'FREE',
                )
            )

        self.sold, self.bought = sold, fifo[index]
        self.qty = min((self.sold.unsettled, self.bought.unsettled))
        self.sold.tie(self)
        self.bought.tie(self)

    def __repr__(self):
        return '%s <--(x%0.5f)--> %s' % (
            self.sold,
            self.qty,
            self.bought,
        )


class StockFIFO:
    @property
    def qty(self):
        return sum([-t.signum * t.qty for t in self.fifo])

    @property
    def timestamp(self):
        return self.fifo[-1].timestamp

    @property
    def value(self):
        return sum([t.price * t.unsettled for t in self.fifo if t.side == 'buy'])

    @property
    def average(self):
        return self.value / self.qty if self.value and self.qty else 0

    def __init__(self, ticker):
        self.ticker = ticker
        self.pointer = 0
        self.fifo = []

    def __getitem__(self, i):
        return self.fifo[i]

    def __repr__(self):
        raise
        return '<StockFIFO:%-5s x %8.2f @ %s>' % (
            self.ticker,
            self.qty,
            mulla(self.average),
        )

    def push(self, qty, price, side, timestamp, description):
        self.fifo.append(
            Lot(
                self.ticker,
                side,
                qty,
                price,
                timestamp,
                description,
            )
        )

        if side == 'sell':
            LotConnector.settle(self)

    def summarize(self, fetch=False):
        print('#' * 80)

        sfmt = lambda qty, lot: '%10.5f %s' % (qty, lot)
        for lot in self.fifo:
            if lot.side == 'buy': continue

            # The buys
            cb = 0
            for tie in lot.ties:
                cb -= tie.qty * tie.bought.price
                print(sfmt(tie.qty, tie.bought))

            # The sell
            print(sfmt(lot.qty, lot))
            cb += lot.qty * lot.price

            print("Cost Basis: %s" % mulla(cb))
            print()

        # Remaining buys (without sells)
        for lot in self.fifo[self.pointer:]:
            if lot.side == 'buy':
                print(sfmt(lot.qty, lot))

        print("Cost : %10.5f x %s = %s" % (
            self.qty, mulla(self.average), mulla(self.value)
        ))
        if fetch and self.qty > ZERO:
            price = flt(rh.stocks.get_latest_price(self.ticker)[0])
            equity = self.qty * price
            print("Value: %10.5f x %s = %s" % (
                self.qty, mulla(price), mulla(equity)
            ))
            print("Position: %s" % (
                mulla(equity - self.value)
            ))

        print()


class CSVReader:
    def __init__(self, filename, importer):
        self.filename = filename
        self.importer = importer

        self.active = None
        self.robinhood = None

        if not os.path.exists(filename):
            self.importer('.', file_name=filename)

        with open(filename, newline='') as fh:
            reader = csv.reader(fh, delimiter=',', quotechar='"')
            self.header = next(reader)
            self.reader = reversed(list(reader))

    def __iter__(self):
        return self

    def __next__(self):
        self.active = next(self.reader, None)
        return self if self.active else None

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
    def description(self):
        raise NotImplementedError

    @property
    def parameters(self):
        raise NotImplementedError


class StockReader(CSVReader):
    def __init__(self, filename='stocks.csv', importer=rh.export_completed_stock_orders):
        super().__init__(filename, importer)
        self._ticker_field = 'symbol'

    @property
    def timestamp(self):
        return dtp.parse(self.get('date'))

    @property
    def description(self):
        return self.get('order_type')

    @property
    def parameters(self):
        return (
            flt(self.get('quantity')),
            flt(self.get('average_price')),
            self.side,
            self.timestamp,
            self.description,
        )


class OptionReader(CSVReader):
    def __init__(self, filename='options.csv', importer=rh.export_completed_option_orders):
        super().__init__(filename, importer)
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
    def description(self):
        side, otype = self.get('side'), self.get('option_type')
        return '%s %s' % (side, otype)

    @property
    def parameters(self):
        return (
            100 * flt(self.get('processed_quantity')),
            flt(self.get('strike_price')),
            self.side,
            self.timestamp,
            self.description,
        )


class Account:
    def __init__(self):
        self.portfolio = {}

        self.robinhood = None
        self.connect()

        self.stockReader = StockReader()
        self.optionReader = OptionReader()

    def __getitem__(self, ticker):
        if ticker not in self.portfolio:
            self.portfolio[ticker] = StockFIFO(ticker)

        return self.portfolio[ticker]

    def connect(self):
        if self.robinhood is None:
            with open(os.path.join(Path.home(), ".rhrc")) as fh:
                username, password = fh.readline().split(',')
                self.robinhood = rh.login(username, password)

        return self.robinhood

    def transactions(self):
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

    def slurp(self, tickers=None):
        for ticker, parameters in self.transactions():
            if tickers is None or ticker in tickers:
                self[ticker].push(*parameters)

    def tickers(self):
        for ticker, stock in self.portfolio.items():
            yield ticker, stock

def main():
    locale.setlocale(locale.LC_ALL, '')

    account = Account()

    tickers = None
    if len(sys.argv) > 1:
        tickers = sys.argv[1].split(',')

    account.slurp(tickers)

    for ticker, stock in account.tickers():
        stock.summarize(fetch=(tickers is not None))

if __name__ == '__main__':
    main()
