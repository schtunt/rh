from functools import wraps
from time import time
from collections import defaultdict

import util
from util.numbers import D

#import gc
#gc.disable()

MEASURED = defaultdict(lambda: dict(count=0, time=0))
def measure(func):
    @wraps(func)
    def timed(*args, **kwargs):
        start = D(time())
        try:
            return func(*args, **kwargs)
        finally:
            MEASURED[func.__name__]['count'] += 1
            MEASURED[func.__name__]['time'] += D(time()) - start
    return timed

def measurements():
    util.output.ddump(MEASURED)



import os
import datetime
import pathlib

from collections import defaultdict

import cachier
import hashlib

import util
import constants

import robin_stocks as rh
import iexfinance.stocks as iex
import yahoo_earnings_calendar as yec
import polygon
import finnhub
import monkeylearn

IEX_STOCKS = {}


CONNECTIONS = {}


@measure
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


@measure
def connected(fn):
    def connected(*args, **kwargs):
        connect()
        return fn(*args, **kwargs)

    return connected

# MonkeyLearn -={
news_blob_hash = lambda *args, **kwargs: hashlib.sha1('|'.join(args[0][0]).encode('utf-8')).digest()
@cachier.cachier(stale_after=datetime.timedelta(hours=4), hash_params=news_blob_hash)
@measure
def sentiments(blob):
    response = CONNECTIONS['ml'].classifiers.classify('cl_pi3C7JiL', blob)
    return response.body
# }=-
# Finnhub -={
@cachier.cachier(stale_after=datetime.timedelta(hours=2))
@measure
def recommendations(ticker):
    return CONNECTIONS['fh'].recommendation_trends(ticker)


@cachier.cachier(stale_after=datetime.timedelta(hours=1))
@measure
def target(ticker):
    return CONNECTIONS['fh'].price_target(ticker)
# }=-
# RH -={
@cachier.cachier(stale_after=datetime.timedelta(days=300))
@measure
def instrument2symbol(url):
    return rh.stocks.get_symbol_by_url(url)


@cachier.cachier(stale_after=datetime.timedelta(hours=1))
@measure
def symbols():
    instruments = positions('stocks', 'all', info='instrument')
    return sorted(
        (
            set(
                map(
                    instrument2symbol,
                    instruments
                )
            ) | set(
                holdings().keys()
            )
        ) - set(('AABAZZ',))
    )


@cachier.cachier(stale_after=datetime.timedelta(days=2))
@measure
def download(context):
    directory = constants.CACHE_DIR
    fn = {
        'stocks': rh.export_completed_stock_orders,
        'options': rh.export_completed_option_orders,
    }[context]
    fn(directory, '%s/%s' % (directory, context))
    return os.path.join(directory, '%s.csv' % context)


@cachier.cachier(stale_after=datetime.timedelta(hours=6))
@measure
def holdings():
    return rh.account.build_holdings()


@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
@measure
def instrument(url):
    return rh.stocks.get_instrument_by_url(url)


@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
@measure
def dividends():
    dividends = defaultdict(list)
    for datum in rh.account.get_dividends():
        uri = datum['instrument']
        ticker = instrument(uri)['symbol']
        dividends[ticker].append(datum)
    return dividends


@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
@measure
def ticker(uri):
    return rh.stocks.get_name_by_url(uri).upper()
# }=-
# RH Orders -={
@cachier.cachier(stale_after=datetime.timedelta(hours=3))
@measure
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
# }=-
# RH Positions -={
@cachier.cachier(stale_after=datetime.timedelta(hours=2))
@measure
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
# }=-
# RH Fundamentals -={
FUNDAMENTALS = {}
@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
@measure
def _fundamentals_agg():
    fun = {}
    for tickers in util.chunk(symbols(), 32):
        fun.update(dict(zip(tickers, rh.stocks.get_fundamentals(tickers))))
    return fun

@measure
def fundamentals(ticker):
    if len(FUNDAMENTALS) == 0:
        _fundamentals_agg()
    if ticker not in FUNDAMENTALS:
        FUNDAMENTALS[ticker] = rh.stocks.get_fundamentals(ticker)[0]
    return FUNDAMENTALS[ticker]
# }=-

# IEX Aggregation -={
IEX_AGGREGATOR = defaultdict(dict)
def _iex_aggregator(fn_name, ticker=None):
    #util.output.ddump(IEX_AGGREGATOR, force=True)

    agg = IEX_AGGREGATOR[fn_name]
    if len(agg) == 0:
        if fn_name in ('get_market_cap', 'get_beta'):
            # https://github.com/addisonlynch/iexfinance/issues/261
            # https://github.com/addisonlynch/iexfinance/issues/260
            print('Refusing to call API for %s' % fn_name)
            pass
        else:
            for chunk in util.chunk(symbols(), 100):
                fn = getattr(iex.Stock(chunk), fn_name)
                agg.update({
                    ticker: datum
                    for ticker, datum in fn().items()
                })

    if ticker is not None:
        if ticker not in agg:
            fn = getattr(iex.Stock(ticker), fn_name)
            datum = fn()
            if datum: agg[ticker] = fn()
        else:
            datum = agg[ticker]

        return datum

    return {ticker: datum for ticker, datum in agg.items()}


@cachier.cachier(stale_after=datetime.timedelta(hours=3))
@measure
def _price_agg(ticker=None): return _iex_aggregator('get_price', ticker)
price = lambda ticker: _price_agg(ticker)


@cachier.cachier(stale_after=datetime.timedelta(days=14))
@measure
def _beta_agg(ticker=None): return _iex_aggregator('get_beta', ticker)
beta = lambda ticker: _beta_agg(ticker)


@cachier.cachier(stale_after=datetime.timedelta(hours=12))
@measure
def _quote_agg(ticker=None): return _iex_aggregator('get_quote', ticker)
quote = lambda ticker: _quote_agg(ticker)


@cachier.cachier(stale_after=datetime.timedelta(hours=3))
@measure
def _stats_agg(ticker=None): return _iex_aggregator('get_key_stats', ticker)
stats = lambda ticker: _stats_agg(ticker)


@cachier.cachier(stale_after=datetime.timedelta(hours=2))
@measure
def _events_agg(ticker=None): return _iex_aggregator('get_events', ticker)
events = lambda ticker: _events_agg(ticker)


@cachier.cachier(stale_after=datetime.timedelta(days=3))
@measure
def _earnings_agg(ticker=None): return _iex_aggregator('get_earnings', ticker)
earnings = lambda ticker: _earnings_agg(ticker)


@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
@measure
def _splits_agg(ticker=None): return _iex_aggregator('get_splits', ticker)
splits = lambda ticker: _splits_agg(ticker)


@cachier.cachier(stale_after=datetime.timedelta(hours=4))
@measure
def _previous_day_prices_agg(ticker=None): return _iex_aggregator('get_previous_day_prices', ticker)
previous_day_prices = lambda ticker: _previous_day_prices_agg(ticker)


@cachier.cachier(stale_after=datetime.timedelta(days=7))
@measure
def _market_cap_agg(ticker=None): return _iex_aggregator('get_market_cap', ticker)
market_cap = lambda ticker: _market_cap_agg(ticker)
marketcap = market_cap


# }=-
# IEX Other -={
@measure
def iex_stock(ticker):
    global IEX_STOCKS
    if ticker in IEX_STOCKS:
        stock = IEX_STOCKS[ticker]
    else:
        stock = iex.Stock(ticker)
        IEX_STOCKS[ticker] = stock
    return stock


def __getattr__(ticker: str) -> iex.Stock:
    return iex_stock(ticker.upper())


@cachier.cachier(stale_after=datetime.timedelta(hours=1))
@measure
def news(ticker, source):
    return {
        'rh': lambda: rh.stocks.get_news(ticker),
        'iex': lambda: iex_stock(ticker).get_news()
    }[source]()
# }=-
