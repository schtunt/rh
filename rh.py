#!/usr/bin/env python3

import os, sys, locale
import time
import pathlib
import click
import colored_traceback.auto

import pandas as pd

import constants

import api
import fields
import account

import util
from util.numbers import D, Z

upper = lambda l: list(map(lambda s: s.upper(), l))
csv_list_flatten = lambda csvl: [token for csv in csvl for token in csv.split(',')]

@click.group()
@click.option('-D', '--debug', is_flag=True, default=False)
@click.pass_context
def cli(ctx, debug):
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug


CONTEXT_SETTINGS = dict(token_normalize_func=lambda x: x.lower())

VIEWS = [
    {
        'configurations': [
            { 'title': 'all', 'sort_by': ['change'] },
            { 'title': 'active', 'sort_by': ['urgency'], 'filter_by': 'next_expiry' },
            { 'title': 'expiring', 'sort_by': ['urgency'], 'filter_by': 'soon_expiring' },
            { 'title': 'urgent', 'sort_by': ['next_expiry'], 'filter_by': 'urgent' },
        ],
        'fields': [
            'sector', 'industry', 'ticker',
            'quantity', 'equity', 'equity_change',                # Holdings
            'marketcap', 'ev', 'shoutstanding',                   # Company Info
            'pcp', 'price', 'percent_change', 'change',           # Price and Last Closing Price
            'cbps', 'cbps%',                                      # Share Performance in this PF,
            'premium_collected', 'dividends_collected',           # Share Performance in this PF
            'pe_ratio', 'pb_ratio', 'beta', 'sharpe', 'treynor',  # Ratios
            'momentum', 'ma', 'd200ma', 'd50ma',                  # Moving Avs & Score + Momentum
            'urgency', 'next_expiry', 'activities',               # Activity
        ],
    },
    {
        'configurations': [
            { 'title': 'tax', 'sort_by': ['ticker'] }
        ],
        'fields': [
            'ticker',
            'equity',
            'price', 'quantity',
            'cusq', 'cusv', 'culq', 'culv',
            'crsq', 'crsv', 'crlq', 'crlv',
            'premium_collected', 'dividends_collected',
            'trd0',
        ],
    },
]

FILTERS = {
    None: lambda row: True,
    'active': lambda row: len(row['activities']) > 0,
    'next_expiry': lambda row: row['next_expiry'] is not pd.NaT,
    'soon_expiring': lambda row: row['next_expiry'] is not pd.NaT and util.datetime.ttl(row['next_expiry']) < 14,
    'urgent': lambda row: row['urgency'] > 0.8,
}

_VIEWS = {}
for cfg in VIEWS:
    columns = cfg['fields']
    for view in cfg['configurations']:
        _VIEWS[view['title']] = dict(
            sort_by=view.get('sort_by', ['ticker']),
            filter_by=FILTERS[view.get('filter_by', None)],
            columns=columns,
        )

@cli.command(help='Refresh Data')
@click.pass_context
def refresh(ctx):
    acc = account.Account()
    batch_size = 5

    view = 'all'
    columns = _VIEWS[view]['columns']
    sort_by = _VIEWS[view]['sort_by']
    filter_by = _VIEWS[view]['filter_by']

    for tickers in util.chunk(acc.stocks.ticker.to_list(), batch_size):
        print("Processing next batch of %d (%s)..." % (batch_size, ','.join(tickers)))
        acc = account.Account(tickers)

        table = util.output.mktable(
            acc.stocks,
            columns,
            tickers=tickers,
            filter_by=filter_by,
            sort_by=sort_by,
            reverse=False,
            limit=0,
        )
        util.output.prtable(table)
        print(util.debug.measurements())

        time.sleep(60)


@cli.command(help='Views')
@click.option('-t', '--tickers', multiple=True, default=None)
@click.option('-v', '--view', default='all', type=click.Choice(_VIEWS.keys()))
@click.option('-s', '--sort-by', multiple=True, default=[])
@click.option('-r', '--reverse', default=False, is_flag=True, type=bool)
@click.option('-l', '--limit', default=0, type=int)
@click.pass_context
def tabulize(ctx, view, sort_by, reverse, limit, tickers):
    debug = ctx.obj['debug']
    tickers = upper(csv_list_flatten(tickers))
    sort_by = csv_list_flatten(sort_by)

    acc = account.Account(tickers)

    columns = _VIEWS[view]['columns']
    sort_by = _VIEWS[view]['sort_by'] if len(sort_by) == 0 else sort_by
    filter_by = _VIEWS[view]['filter_by']

    table = util.output.mktable(
        acc.stocks,
        columns,
        tickers=tickers,
        filter_by=filter_by,
        sort_by=sort_by,
        reverse=reverse,
        limit=limit,
    )
    util.output.prtable(table)

@cli.command(help='Account History')
@click.option('-t', '--tickers', multiple=True, required=True)
@click.pass_context
def history(ctx, tickers):
    debug = ctx.obj['debug']
    tickers = upper(csv_list_flatten(tickers))
    acc = account.Account(tickers)

    for ticker, stock in acc.portfolio:
        print(stock)
        stock.summarize()

    print(util.debug.measurements())


acc = None
def repl():
    print("Initializing APIs...")

    global acc
    acc = account.Account()

    print("Injecting Stock objects for all stocks in your portfolio...")
    module = sys.modules[__name__]

    print("Done! Available ticker objects:")
    print(" + api.acc                (Local Robinhood Account object)")
    print()
    print(" + api.rh                 (RobinStocksEndpoint API)")
    print(" + api.iex                (IEXFinanceEndpoint API)")
    print(" + api.yec                (YahooEarningsCalendarEndpoint API)")
    print()
    print(" + acc.stock('<ticker>')  (Local Stock API-Multiplexor object)")
    print(" + api.<ticker>           (IEXFinanceEndpoint Stock API object)")
    print(" + S                      (Stocks DataFrame)")
    print(" + T                      (Transactions DataFrame)")
    print()
    print("Meta-helpers for this REPL")
    print(" + relmod()         (reload wthout having to exit the repl)")


def preinitialize():
    util.numbers.preinitialize()
    api.connect()
    locale.setlocale(locale.LC_ALL, '')
    if not pathlib.posixpath.exists(constants.CACHE_DIR):
        os.mkdir(constants.CACHE_DIR)

if __name__ == '__main__':
    preinitialize()
    cli()
