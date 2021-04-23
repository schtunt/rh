#!/usr/bin/env python3

import locale
import os
import pathlib
import time

# import colored_traceback.auto
# import rich.traceback
import better_exceptions
import click
import pandas as pd

# import fields
import account
import api
import constants
import util

# from util.numbers import F

upper = lambda l: list(map(lambda s: s.upper(), l))
csv_list_flatten = lambda csvl: [token for csv in csvl for token in csv.split(',')]


@click.group()
@click.option('-D', '--debug', is_flag=True, default=False)
@click.pass_context
def cli(ctx, debug):
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug


CONTEXT_SETTINGS = dict(token_normalize_func=lambda x: x.lower())

FIELD_GROUPS = {
    'base': ['ticker'],
    'category': ['sector', 'industry'],
    'company': ['marketcap', 'so'],
    'ev': ['ev', 'ebit', 'ebit2ev'],
    'position': ['quantity', 'price', 'equity', 'percentage', 'cbps'],
    'ma': ['d200ma', 'd50ma'],
    'change': ['equity_change', 'percent_change'],
    'income': ['premium_collected', 'dividends_collected'],
    'performance': ['roe'],
    'daytrader': ['pcp', 'd1c%', 'd5c%', 'd30c%', 'first_traded', 'last_traded'],
    'ratios': ['p2e', 'p2b', 'p2s', 'peg', 'beta'],  # Valuation Ratios
    'ratios-ii': ['sharpe', 'treynor'],
    'you-score': ['you_score%', 'cbps%', 'dyps%', 'pcps%'],
    'stock-score': ['stock_score%', 'malps%', 'momentum', 'analyst_score'],
    'options': ['urgency', 'next_expiry', 'activities'],
}

VIEWS = {
    'all': {
        'sort_by': ['ticker'],
        'fieldgroups': ['base', 'company', 'position', 'ratios', 'ratios-ii', 'ev'],
    },
    'perf': {
        'sort_by': ['ticker'],
        'fieldgroups': [
            'base', 'position', 'you-score', 'stock-score', 'income', 'change',
            'daytrader', 'ratios', 'ratios-ii', 'ev', 'performance',
        ],
    },
    'options': {
        'sort_by': ['urgency'],
        'filter_by': 'next_expiry',
        'fieldgroups': ['base', 'position', 'options'],
    },
    'urgent': {
        'sort_by': ['next_expiry'],
        'filter_by': 'urgent',
        'fieldgroups': ['base', 'position', 'options'],
    },
}

FILTERS = {
    None: lambda row: True,
    'active': lambda row: len(row['activities']) > 0,
    'next_expiry': lambda row: row['next_expiry'] is not pd.NaT,
    'soon_expiring': lambda row: row['next_expiry'] is not pd.NaT and util.datetime.ttl(row['next_expiry']) < 14,
    'urgent': lambda row: row['urgency'] > 0.8,
}

_VIEWS = {}
for view, configuration in VIEWS.items():
    _VIEWS[view] = dict(
        sort_by=configuration.get('sort_by', ['ticker']),
        filter_by=FILTERS[configuration.get('filter_by', None)],
        columns=[f for fg in configuration['fieldgroups'] for f in FIELD_GROUPS[fg]]
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

        time.sleep(100)


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


def preinitialize():
    os.environ['BETTER_EXCEPTIONS'] = 'TRUE'
    better_exceptions.MAX_LENGTH = None
    # rich.traceback.install()

    util.numbers.preinitialize()
    api.connect()
    locale.setlocale(locale.LC_ALL, '')
    if not pathlib.posixpath.exists(constants.CACHE_DIR):
        os.mkdir(constants.CACHE_DIR)


if __name__ == '__main__':
    preinitialize()
    cli()
