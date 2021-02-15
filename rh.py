#!/usr/bin/env python3

import os, sys, locale
import click

import pathlib

import __main__

import constants
from constants import ZERO as Z

import api
import account

import util
from util.numbers import dec as D
from util.output import ansistrip as S
DS = lambda s: D(S(s))

@click.group()
@click.option('-D', '--debug-tickers',multiple=True,  default=None)
@click.pass_context
def cli(ctx, debug_tickers):
    constants.DEBUG=debug_tickers
    ctx.ensure_object(dict)

CONTEXT_SETTINGS = dict(token_normalize_func=lambda x: x.lower())

FILTERS = {
    'active': lambda d: len(d['activities']),
    'optionable': lambda d: DS(d['quantity']) - DS(d['CC.Coll']) > 100,
    'expiry': lambda d: d != 'N/A',
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
            'since_close', 'since_open', 'alerts',
            'pe_ratio', 'pb_ratio', 'beta',
            'CC.Coll', 'CSP.Coll',
            'activities'
        ],
    },
    'losers': {
        'sort_by': 'rank',
        'columns': [
            'ticker',
            'marketcap',
            'rank', 'analyst', 'news', 'ma',
            'epst%', 'epst', 'price', 'quantity',
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
        'sort_by': 'premium_collected',
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
        'sort_by': 'ttl',
        'filter_by': 'expiry',
        'columns': [
            'ticker', 'percentage',
            'quantity', 'price', 'epst', 'rank',
            'equity',
            'equity_change', 'percent_change',
            'premium_collected', 'dividends_collected',
            'activities',
            'expiry', 'ttl'
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

    fundamentals = api.fundamentals()
    for ticker, datum in account.data.items():
        stock = account.get_stock(ticker)
        index = stock.pointer

        print(stock)
        # Say you have a limit order on a stock for the very first time:
        if len(stock.events) == 0: continue

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
        'rank': D,
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
        'expiry': lambda dt: 'N/A' if dt is None else dt.strftime('%Y-%m-%d'),
        'ttl': util.color.qty0,
    }

    data = account.data.values()
    table = util.output.mktable(
        data,
        view,
        formats,
        sort_by=sort_by,
        reverse=reverse,
        limit=limit
    )
    print(table)

    if constants.DEBUG:
        print(table.rows.filter(lambda row: row['ticker'] in constants.DEBUG))


@cli.command(help='Account History')
@click.pass_context
def history(ctx):
    account = ctx.obj['account']
    account.slurp()
    for ticker, stock in account.stocks.items():
        if len(constants.DEBUG) > 0 and ticker not in constants.DEBUG: continue
        stock.summarize()

def preinitialize(repl=False):
    api.connect()
    locale.setlocale(locale.LC_ALL, '')
    if not pathlib.posixpath.exists(constants.CACHE_DIR):
        os.mkdir(constants.CACHE_DIR)

acc = None
def interact():
    print("Preparing REPL...")
    preinitialize(repl=True)

    print("Initializing APIs...")
    global acc
    acc = account.Account()
    acc.slurp()

    #print(acc.rh.authentication.getpass.getuser())

    print("Injecting Stock objects for all stocks in your portfolio...")
    module = sys.modules[__name__]
    for ticker in acc.tickers:
        key = ticker.lower()
        setattr(module, key, acc.get_stock(ticker))
        #setattr(module, '_%s' % key, iex.Stock(ticker))

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
    cli(obj={'account': account.Account()})
elif not hasattr(__main__, '__file__'):
    interact()

'''
In [38]: [x['symbol'] for x in api.iex.get_market_losers()]
Out[38]: ['FOLD', 'SMTI', 'TUFN', 'ONCR', 'SQZ', 'GMBL', 'FDMT', 'RELI', 'ACB']

In [40]: [x['symbol'] for x in api.iex.get_market_gainers()]
Out[40]: ['WNW', 'PRTA', 'CCNC']

api.iex.get_market_most_active()

'''
