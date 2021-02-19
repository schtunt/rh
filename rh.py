#!/usr/bin/env python3

import os, sys, locale
import pathlib
import click
import colored_traceback.auto

import pandas as pd

import constants
from constants import ZERO as Z

import api
import account
import fields

import util
from util.numbers import D
from util.color import strip as S
DS = lambda s: D(S(s))

@click.group()
@click.option('-D', '--debug', is_flag=True, default=False)
@click.option('-C', '--clear-cache', is_flag=True, default=False)
@click.pass_context
def cli(ctx, debug, clear_cache):
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug

    if clear_cache:
        from slurp import FEATHERS
        for fn in FEATHERS.values():
            if os.path.exists(fn):
                os.unlink(fn)


CONTEXT_SETTINGS = dict(token_normalize_func=lambda x: x.lower())

VIEWS = [
    {
        'configurations': [
            { 'title': 'movers', 'sort_by': 'change', 'filter_by': 'movers' },
        ],
        'fields': [
            'ticker',
            'marketcap',
            'ma', 'd200ma', 'd50ma', 'pcp', 'price',
            'change',
            'esp',
            'quantity',
            'alerts',
            'pe_ratio', 'pb_ratio', 'beta',
            'equity', 'equity_change', 'percent_change',
            'premium_collected', 'dividends_collected',
            'activities',
            'trd0',
        ],
    },
    {
        'configurations': [
            { 'title': 'active', 'sort_by': 'urgency', 'filter_by': 'next_expiry' },
        ],
        'fields': [
            'ticker', 'percentage',
            'quantity', 'price', 'esp',
            'equity',
            'equity_change', 'percent_change',
            'premium_collected', 'dividends_collected',
            'activities',
            'next_expiry',
            'urgency',
        ],
    },
    {
        'configurations': [
            { 'title': 'expiring', 'sort_by': 'urgency', 'filter_by': 'soon_expiring' },
            { 'title': 'urgent', 'sort_by': 'next_expiry', 'filter_by': 'urgent' },
        ],
        'fields': [
            'ticker', 'percentage',
            'quantity', 'price', 'esp',
            'equity',
            'equity_change', 'percent_change',
            'premium_collected', 'dividends_collected',
            'activities',
            'next_expiry',
            'urgency',
        ],
    },
    {
        'configurations': [
            { 'title': 'tax', 'sort_by': 'ticker' }
        ],
        'fields': [
            'ticker',
            'equity',
            'price', 'quantity',
            'cusq', 'cusv', 'culq', 'culv',
            'crsq', 'crsv', 'crlq', 'crlv',
            'premium_collected', 'dividends_collected',
        ],
    },
]

FORMATERS = {
    'bucket': lambda b: util.color.colored('$%dk' % b, 'blue'),
    'since_open': util.color.mpct,
    'since_close': util.color.mpct,
    'CC.Coll': util.color.qty0,                   # Covered Call Collateral
    'CSP.Coll': util.color.mulla,                 # Cash-Secured Put Collateral
    'price': util.color.mulla,
    'pcp': util.color.mulla,                      # Previous Close Price (PCP)
    'change': util.color.pct,                     # Change since PCP
    'esp': util.color.mulla,                      # Effective Share Price
    'quantity': util.color.qty0,
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
    'next_expiry': util.datetime.ttl,
    'urgency': util.color.mpctr,
    'trd0': util.datetime.age,
}
FORMATERS.update(fields.formaters())

FILTERS = {
    None: lambda d: d,
    'active': lambda d: len(d['activities']),
    'next_expiry': lambda d: d['next_expiry'] is not pd.NaT,
    'soon_expiring': lambda d: d['next_expiry'] is not pd.NaT and util.datetime.ttl(d['next_expiry']) < 7,
    'movers': lambda d: abs(d['change']) >= 5,
    'urgent': lambda d: d['urgency'] > 0.8,
}

_VIEWS = {}
for cfg in VIEWS:
    fields = cfg['fields']
    for view in cfg['configurations']:
        _VIEWS[view['title']] = dict(
            sort_by=view.get('sort_by', 'ticker'),
            filter_by=FILTERS[view.get('filter_by', None)],
            fields=fields,
        )

@cli.command(help='Views')
@click.option('-t', '--tickers', multiple=True, default=None)
@click.option('-v', '--view', default='expiring', type=click.Choice(_VIEWS.keys()))
@click.option('-s', '--sort-by', default=False, type=str)
@click.option('-r', '--reverse', default=False, is_flag=True, type=bool)
@click.option('-l', '--limit', default=-1, type=int)
@click.pass_context
def tabulize(ctx, view, sort_by, reverse, limit, tickers):
    debug = ctx.obj['debug']
    tickers = [t.upper() for t in tickers]
    acc = account.Account(tickers)

    columns = _VIEWS[view]['fields']
    sort_by = _VIEWS[view]['sort_by']
    filter_by = _VIEWS[view]['filter_by']

    table = util.output.mktable(
        acc.stocks,
        columns,
        FORMATERS,
        tickers=tickers,
        filter_by=filter_by,
        sort_by=sort_by,
        reverse=reverse,
        limit=limit
    )
    util.output.prtable(table)

    api.measurements()


@cli.command(help='Account History')
@click.option('-t', '--tickers', multiple=True, required=True)
@click.pass_context
def history(ctx, tickers):
    debug = ctx.obj['debug']
    acc = account.Account(tickers)

    for ticker, stock in acc.portfolio:
        print(stock)
        stock.summarize()

    api.measurements()


acc = None
def repl():
    print("Initializing APIs...")

    global acc
    acc = account.Account()

    print("Injecting Stock objects for all stocks in your portfolio...")
    module = sys.modules[__name__]

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


def preinitialize():
    api.connect()
    locale.setlocale(locale.LC_ALL, '')
    if not pathlib.posixpath.exists(constants.CACHE_DIR):
        os.mkdir(constants.CACHE_DIR)

if __name__ == '__main__':
    preinitialize()
    cli()
