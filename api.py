import os
import datetime
import pathlib

from collections import defaultdict

import cachier
import hashlib

import util
from util.numbers import dec as D
import constants

import robin_stocks as rh
import iexfinance.stocks as iex
import yahoo_earnings_calendar as yec
import polygon
import finnhub
import monkeylearn

IEX_STOCKS = {}


def iex_stock(ticker):
    global IEX_STOCKS
    if ticker in IEX_STOCKS:
        stock = IEX_STOCKS[ticker]
    else:
        stock = iex.Stock(ticker)
        IEX_STOCKS[ticker] = stock
    return stock


CONNECTIONS = {}


def connect():
    if len(CONNECTIONS) > 0: return

    CONNECTIONS['yec'] = yec.YahooEarningsCalendar()
    # use datetime.fromtimestamp() for ^^'s results

    with open(os.path.join(pathlib.Path.home(), ".rhrc")) as fh:
        (
            username, password,
            polygon_api_key,
            iex_api_key,
            finnhub_api_key,
            monkeylearn_api,
            alpha_vantage_api,
        ) = [token.strip() for token in fh.readline().split(',')]

        CONNECTIONS['rh'] = rh.login(username, password)
        CONNECTIONS['fh'] = finnhub.Client(api_key=finnhub_api_key)
        CONNECTIONS['ml'] = monkeylearn.MonkeyLearn(monkeylearn_api)

        os.environ['IEX_TOKEN'] = iex_api_key
        os.environ['IEX_OUTPUT_FORMAT'] = 'json'

    rh.helper.set_output(open(os.devnull, "w"))


news_blob_hash = lambda blob: hashlib.sha1('|'.join(blob).encode('utf-8'))
@cachier.cachier(stale_after=datetime.timedelta(hours=4), hash_params=news_blob_hash)
def sentiments(blob):
    return CONNECTIONS['ml'].classifiers.classify('cl_pi3C7JiL', blob)


@cachier.cachier(stale_after=datetime.timedelta(hours=2))
def recommendations(ticker):
    return CONNECTIONS['fh'].recommendation_trends(ticker)


@cachier.cachier(stale_after=datetime.timedelta(hours=1))
def target(ticker):
    return CONNECTIONS['fh'].price_target(ticker)


def connected(fn):
    def connected(*args, **kwargs):
        connect()
        return fn(*args, **kwargs)

    return connected


@cachier.cachier(stale_after=datetime.timedelta(hours=1))
def symbols():
    return sorted(map(
        rh.stocks.get_symbol_by_url,
        positions('stocks', 'all', info='instrument'),
    ))


@cachier.cachier(stale_after=datetime.timedelta(days=2))
def download(context):
    directory = constants.CACHE_DIR
    fn = {
        'stocks': rh.export_completed_stock_orders,
        'options': rh.export_completed_option_orders,
    }[context]
    fn(directory, '%s/%s' % (directory, context))
    return os.path.join(directory, '%s.csv' % context)


@cachier.cachier(stale_after=datetime.timedelta(hours=3))
def holdings():
    return rh.account.build_holdings()


@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
def instrument(url):
    return rh.stocks.get_instrument_by_url(url)

@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
def dividends():
    dividends = defaultdict(list)
    for datum in rh.account.get_dividends():
        uri = datum['instrument']
        ticker = instrument(uri)['symbol']
        dividends[ticker].append(datum)
    return dividends


@cachier.cachier(stale_after=datetime.timedelta(hours=2))
def positions(otype, ostate, info=None):
    return {
        'options': {
            'all': rh.options.get_all_option_positions,
            'open': rh.options.get_open_option_positions,
        },
        'stocks': {
            'all': rh.account.get_all_positions,
            'open': rh.account.get_open_stock_positions,
        },
    }[otype][ostate](info)
    #rh.options.get_aggregate_positions

@cachier.cachier(stale_after=datetime.timedelta(hours=2))
def orders(otype, ostate, info=None):
    return {
        'options': {
            'all': rh.orders.get_all_option_orders,
            'open': rh.orders.get_all_open_option_orders,
        },
        'stocks': {
            'all': rh.orders.get_all_stock_orders,
            'open': rh.orders.get_all_open_stock_orders,
        },
    }[otype][ostate](info)


@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
def ticker(uri):
    return rh.stocks.get_name_by_url(uri).upper()


PRICES = {}
@cachier.cachier(stale_after=datetime.timedelta(hours=3))
def prices():
    tickers = sorted(set(symbols()) | set(PRICES.keys()))
    prices = rh.stocks.get_latest_price(tickers)
    PRICES.update(dict(zip(tickers, map(D, prices))))
    return PRICES

@cachier.cachier(stale_after=datetime.timedelta(hours=3))
def price(ticker):
    if ticker not in PRICES:
        stock = iex_stock(ticker)
        PRICES[ticker] = D(stock.get_price())
    return PRICES[ticker]


FUNDAMENTALS = {}
@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
def fundamentals():
    fun = {}
    tickers = sorted(set(symbols()) | set(FUNDAMENTALS.keys()))
    for tickers in util.chunk(tickers, 32):
        fun.update(dict(zip(tickers, rh.stocks.get_fundamentals(tickers))))
    return fun

@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
def fundamental(ticker):
    if ticker not in FUNDAMENTALS:
        FUNDAMENTALS[ticker] = rh.stocks.get_fundamentals(ticker)[0]
    return FUNDAMENTALS[ticker]


@cachier.cachier(stale_after=datetime.timedelta(hours=2))
def events(ticker):
    return rh.stocks.get_events(ticker)


@cachier.cachier(stale_after=datetime.timedelta(days=3))
def earnings(ticker):
    return rh.stocks.get_earnings(ticker)


@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
def splits(ticker):
    stock = iex_stock(ticker)
    return stock.get_splits()


@cachier.cachier(stale_after=datetime.timedelta(hours=4))
def yesterday(ticker):
    stock = iex_stock(ticker)
    return stock.get_previous_day_prices()


@cachier.cachier(stale_after=datetime.timedelta(days=1))
def marketcap(ticker):
    stock = iex_stock(ticker)
    return stock.get_market_cap()


@cachier.cachier(stale_after=datetime.timedelta(hours=3))
def stats(ticker):
    stock = iex_stock(ticker)
    return stock.get_key_stats()


@cachier.cachier(stale_after=datetime.timedelta(hours=3))
def beta(ticker):
    stock = iex_stock(ticker)
    return stock.get_beta()


@cachier.cachier(stale_after=datetime.timedelta(hours=1))
def quote(ticker):
    stock = iex_stock(ticker)
    return stock.get_quote()


@cachier.cachier(stale_after=datetime.timedelta(hours=1))
def news(ticker, source):
    return {
        'rh': lambda: rh.stocks.get_news(ticker),
        'iex': lambda: iex_stock(ticker).get_news()
    }[source]()
