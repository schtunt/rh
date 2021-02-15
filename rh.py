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
        'sort_by': 'ticker',
        'columns': [
            'ticker', 'percentage',
            'quantity', 'price',
            'esp',
            'ma',
            'equity', 'equity_change', 'percent_change',
            'premium_collected', 'dividends_collected',
            'short', 'bucket',
            'since_close', 'since_open', 'alerts',
            'pe_ratio', 'pb_ratio', #'beta',
            #'CC.Coll', 'CSP.Coll',
            'activities'
        ],
    },
    'losers': {
        'sort_by': 'ticker',
        'columns': [
            'ticker',
            'marketcap',
            'ma', 'd200ma', 'd50ma', 'price',
            'esp',
            'quantity',
            'alerts',
            'pe_ratio', 'pb_ratio', #'beta',
            'equity', 'equity_change', 'percent_change',
            'premium_collected', 'dividends_collected',
            #'crv', 'cuv',
            #'CC.Coll', 'CSP.Coll',
            'activities',
        ],
    },
     'gen': {
        'sort_by': 'premium_collected',
        'filter_by': 'optionable',
        'columns': [
            'ticker', 'percentage',
            'quantity', 'price',
            'esp',
            'equity', 'equity_change', 'percent_change',
            'premium_collected', 'dividends_collected',
            'ma', 'since_close', 'since_open', 'alerts',
            'pe_ratio', 'pb_ratio', #'beta',
            #'CC.Coll', 'CSP.Coll',
            'activities',
        ],
    },
    'active': {
        'sort_by': 'ttl',
        'filter_by': 'expiry',
        'columns': [
            'ticker', 'percentage',
            'quantity', 'price', 'esp',
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

FORMATERS = {
    'bucket': lambda b: util.color.colored('$%dk' % b, 'blue'),
    'since_open': util.color.mpct,
    'since_close': util.color.mpct,
    'CC.Coll': util.color.qty0,                   # Covered Call Collateral
    'CSP.Coll': util.color.mulla,                 # Cash-Secured Put Collateral
    'price': util.color.mulla,
    'esp': util.color.mulla,                      # Effective Share Price
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
    'delta': util.color.qty,
    'short': util.color.qty1,
    'premium_collected': util.color.mulla,
    'dividends_collected': util.color.mulla,
    'd200ma': util.color.qty,
    'd50ma': util.color.qty,
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


@cli.command(help='Views')
@click.option('-v', '--view', default='pie', type=click.Choice(VIEWS.keys()))
@click.option('-s', '--sort-by', default=False, type=str)
@click.option('-r', '--reverse', default=False, is_flag=True, type=bool)
@click.option('-l', '--limit', default=-1, type=int)
@click.pass_context
def tabulize(ctx, view, sort_by, reverse, limit):
    account = ctx.obj['account']
    table = util.output.mktable(
        VIEWS,
        account.stocks,
        view,
        FORMATERS,
        sort_by=sort_by,
        reverse=reverse,
        limit=limit
    )
    print(table)

    if constants.DEBUG:
        print(table.rows.filter(lambda row: row['ticker'] in constants.DEBUG))

    api.measurements()


@cli.command(help='Account History')
@click.pass_context
def history(ctx):
    account = ctx.obj['account']
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

    #print(acc.rh.authentication.getpass.getuser())

    print("Injecting Stock objects for all stocks in your portfolio...")
    module = sys.modules[__name__]
    #for ticker in api.symbols():
        #key = ticker.lower()
        #setattr(module, key, acc.portfolio[ticker])
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
