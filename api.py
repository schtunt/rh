import os
import sys
import typing
import pathlib
import datetime
import dataclasses

from collections import defaultdict

import util
from util.numbers import D, Z, NaN

import cachier
import hashlib

import util
import util.debug
import constants

import robin_stocks as rh
import iexfinance.stocks as iex
import yahoo_earnings_calendar as _yec
import finnhub as _fh
import yfinance as yf
import tiingo as tii
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries

from secrets import SECRETS


TICKER_CHAIN = defaultdict(tuple)
TICKER_CHAIN.update({
    'STLA': ( 'FCAU', ),
    'SIRI': ( 'P', ),
    'AABAZZ': ( 'AABA', ),
})

TICKER_BLACKLIST = set((
    'AABAZZ',
)) | set(
    tocker for chain in TICKER_CHAIN.values() for tocker in chain
)
for ticker in TICKER_BLACKLIST:
    if ticker in TICKER_CHAIN:
        TICKER_BLACKLIST |= set(TICKER_CHAIN[ticker])

is_black = lambda ticker: bool(ticker in TICKER_BLACKLIST)
is_white = lambda ticker: bool(ticker not in TICKER_BLACKLIST)
tockers4ticker = lambda ticker: set((ticker,) + TICKER_CHAIN[ticker])
def tocker2ticker(tocker):
    for ticker, tockers in TICKER_CHAIN.items():
        if tocker in tockers:
            return ticker
    return tocker


IEX_STOCKS = {}


CONNECTIONS = {}


@util.debug.measure
def connect():
    if len(CONNECTIONS) > 0: return

    module = sys.modules[__name__]

    secrets = defaultdict(lambda: None, SECRETS)
    username, password = (
        secrets['robinhood']['username'],
        secrets['robinhood']['password'],
    )
    CONNECTIONS['rh'] = rh.login(username, password)
    rh.helper.set_output(open(os.devnull, "w"))

    iex_api_key = secrets['iex_api_key']
    os.environ['IEX_TOKEN'] = iex_api_key
    os.environ['IEX_OUTPUT_FORMAT'] = 'json'

    finnhub_api_key = secrets['finnhub_api_key']
    CONNECTIONS['fh'] = _fh.Client(
        api_key=finnhub_api_key
    ) if finnhub_api_key else None
    module.fh = CONNECTIONS['fh']

    CONNECTIONS['yec'] = _yec.YahooEarningsCalendar()
    # use datetime.fromtimestamp() for ^^'s results
    module.yec = CONNECTIONS['yec']

    alpha_vantage_api_key = secrets['alpha_vantage_api_key']
    os.environ['ALPHAVANTAGE_API_KEY'] = alpha_vantage_api_key
    #ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
    #data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')

    CONNECTIONS['av'] = dict(
        fd=FundamentalData()
    )
    module.fd = CONNECTIONS['av']['fd']

    CONNECTIONS['tii'] = tii.TiingoClient(dict(
        session=True,
        api_key=secrets['tiingo_api_key']
    ))
    module.tii = CONNECTIONS['tii']

@util.debug.measure
def connected(fn):
    def connected(*args, **kwargs):
        connect()
        return fn(*args, **kwargs)

    return connected

# MonkeyLearn -={
news_blob_hash = lambda *args, **kwargs: hashlib.sha1('|'.join(args[0][0]).encode('utf-8')).digest()
@cachier.cachier(stale_after=datetime.timedelta(hours=4), hash_params=news_blob_hash)
@util.debug.measure
def sentiments(blob):
    response = CONNECTIONS['ml'].classifiers.classify('cl_pi3C7JiL', blob)
    return response.body
# }=-
# AlphaVantage -={
# fd.get_company_overview
# 'Symbol', 'AssetType', 'Name', 'Description', 'Exchange', 'Currency', 'Country', 'Sector',
# 'Industry', 'Address', 'FullTimeEmployees', 'FiscalYearEnd', 'LatestQuarter',
# 'MarketCapitalization', 'EBITDA', 'PERatio', 'PEGRatio', 'BookValue', 'DividendPerShare',
# 'DividendYield', 'EPS', 'RevenuePerShareTTM', 'ProfitMargin', 'OperatingMarginTTM',
# 'ReturnOnAssetsTTM', 'ReturnOnEquityTTM', 'RevenueTTM', 'GrossProfitTTM', 'DilutedEPSTTM',
# 'QuarterlyEarningsGrowthYOY', 'QuarterlyRevenueGrowthYOY', 'AnalystTargetPrice', 'TrailingPE',
# 'ForwardPE', 'PriceToSalesRatioTTM', 'PriceToBookRatio', 'EVToRevenue', 'EVToEBITDA', 'Beta',
# '52WeekHigh', '52WeekLow', '50DayMovingAverage', '200DayMovingAverage', 'SharesOutstanding',
# 'SharesFloat', 'SharesShort', 'SharesShortPriorMonth', 'ShortRatio', 'ShortPercentOutstanding',
# 'ShortPercentFloat', 'PercentInsiders', 'PercentInstitutions', 'ForwardAnnualDividendRate',
# 'ForwardAnnualDividendYield', 'PayoutRatio', 'DividendDate', 'ExDividendDate',
# 'LastSplitFactor', 'LastSplitDate'

_AV_STOP = False
@cachier.cachier(stale_after=datetime.timedelta(seconds=300))
def _overview(ticker):
    '''
    The free-tier of AV imposes a 500/day day-limit, and a 5/minute rate-limit.  Hence it will
    not be rare to find this function returning `None'.  The wrappers need to handle this as a
    non-exceptional event.
    '''
    global _AV_STOP

    if _AV_STOP: return None

    try:
        return CONNECTIONS['av']['fd'].get_company_overview(ticker)[0]
    except:
        _AV_STOP = True
        return None

#def sector(ticker):
#    overview = _overview(ticker)
#    return overview['Sector'] if overview is not None else 'N/A'

def industry(ticker):
    overview = _overview(ticker)
    return overview['Industry'] if overview is not None else 'N/A'

def ebit(ticker):
    '''
    Earnings Before Interest and Taxes
    Used to measure a firm's operating income.
    '''
    return

def ebt(ticker):
    '''
    Earnings Before Taxes
    While used for essentially the same purpose as EBIT, this represents a firm's operating
    income after accounting for expenses outside of the firm's control.
    '''
    return

# fd.get_balance_sheet_annual
# 'fiscalDateEnding', 'reportedCurrency', 'totalAssets',
# 'intangibleAssets', 'earningAssets', 'otherCurrentAssets',
# 'totalLiabilities', 'totalShareholderEquity',
# 'deferredLongTermLiabilities', 'otherCurrentLiabilities', 'commonStock',
# 'retainedEarnings', 'otherLiabilities', 'goodwill', 'otherAssets',
# 'cash', 'totalCurrentLiabilities', 'shortTermDebt',
# 'currentLongTermDebt', 'otherShareholderEquity',
# 'propertyPlantEquipment', 'totalCurrentAssets', 'longTermInvestments',
# 'netTangibleAssets', 'shortTermInvestments', 'netReceivables',
# 'longTermDebt', 'inventory', 'accountsPayable', 'totalPermanentEquity',
# 'additionalPaidInCapital', 'commonStockTotalEquity',
# 'preferredStockTotalEquity', 'retainedEarningsTotalEquity',
# 'treasuryStock', 'accumulatedAmortization', 'otherNonCurrrentAssets',
# 'deferredLongTermAssetCharges', 'totalNonCurrentAssets',
# 'capitalLeaseObligations', 'totalLongTermDebt',
# 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities',
# 'negativeGoodwill', 'warrants', 'preferredStockRedeemable',
# 'capitalSurplus', 'liabilitiesAndShareholderEquity',
# 'cashAndShortTermInvestments', 'accumulatedDepreciation',
# 'commonStockSharesOutstanding'

# }=-
# Finnhub -={
@cachier.cachier(stale_after=datetime.timedelta(hours=2))
@util.debug.measure
def recommendations(ticker):
    return CONNECTIONS['fh'].recommendation_trends(ticker)


@cachier.cachier(stale_after=datetime.timedelta(hours=1))
@util.debug.measure
def target(ticker):
    return CONNECTIONS['fh'].price_target(ticker)
# }=-
# RH -={
@cachier.cachier(stale_after=datetime.timedelta(days=300))
@util.debug.measure
def instrument2symbol(url):
    return rh.stocks.get_symbol_by_url(url)


@cachier.cachier(stale_after=datetime.timedelta(hours=1))
@util.debug.measure
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
        ) - TICKER_BLACKLIST
    )


@cachier.cachier(stale_after=datetime.timedelta(days=2))
@util.debug.measure
def download(context):
    directory = constants.CACHE_DIR
    fn = {
        'stocks': rh.export_completed_stock_orders,
        'options': rh.export_completed_option_orders,
    }[context]
    fn(directory, '%s/%s' % (directory, context))
    return os.path.join(directory, '%s.csv' % context)


@cachier.cachier(stale_after=datetime.timedelta(hours=6))
@util.debug.measure
def holdings():
    return rh.account.build_holdings()


@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
@util.debug.measure
def instrument(url):
    return rh.stocks.get_instrument_by_url(url)


@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
@util.debug.measure
def dividends():
    dividends = defaultdict(list)
    for datum in rh.account.get_dividends():
        uri = datum['instrument']
        ticker = instrument(uri)['symbol']
        dividends[ticker].append(datum)
    return dividends


@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
@util.debug.measure
def ticker(uri):
    return rh.stocks.get_name_by_url(uri).upper()
# }=-
# RH Orders -={
@cachier.cachier(stale_after=datetime.timedelta(hours=3))
@util.debug.measure
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
@util.debug.measure
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
# RH Ticker Endpoints -={
FUNDAMENTALS = {}
@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
@util.debug.measure
def _fundamentals_agg():
    fun = {}
    for tickers in util.chunk(symbols(), 32):
        fun.update(dict(zip(tickers, rh.stocks.get_fundamentals(tickers))))
    return fun

@util.debug.measure
def fundamentals(ticker):
    if len(FUNDAMENTALS) == 0:
        _fundamentals_agg()
    if ticker not in FUNDAMENTALS:
        FUNDAMENTALS[ticker] = rh.stocks.get_fundamentals(ticker)[0]
    return FUNDAMENTALS[ticker]
# }=-

@cachier.cachier(stale_after=datetime.timedelta(days=14))
@util.debug.measure
def events(ticker):
    return rh.stocks.get_events(ticker)



# IEX Aggregation -={
IEX_AGGREGATOR = defaultdict(dict)
def _iex_aggregator(fn_name, ticker=None):
    #util.debug.ddump(IEX_AGGREGATOR, force=True)

    agg = IEX_AGGREGATOR[fn_name]
    if len(agg) == 0:
        for chunk in util.chunk(symbols(), 100):
            fn = getattr(iex.Stock(chunk), fn_name)
            retrieved = fn()
            if type(retrieved) is list:
                data = defaultdict(list)
                for datum in retrieved:
                    data[datum['symbol']].append(datum)
            elif type(retrieved) is dict:
                data = {ticker: datum for ticker, datum in retrieved.items()}
            else:
                raise RuntimeError(
                    "Error: `%s' (retrieved) is neither dict nor list" % retrieved
                )

            agg.update(data)

    if ticker is not None:
        if ticker not in agg:
            fn = getattr(iex.Stock(ticker), fn_name)
            try:
                datum = fn()
                if datum:
                    agg[ticker] = datum
            except:
                print("Error handling ticker `%s'" % ticker)
                raise
        else:
            datum = agg[ticker]

        return datum

    return {ticker: datum for ticker, datum in agg.items()}


@cachier.cachier(stale_after=datetime.timedelta(hours=3))
@util.debug.measure
def _price_agg(ticker=None): return {
    ticker: D(price) for ticker, price in _iex_aggregator('get_price', ticker).items()
}


@cachier.cachier(stale_after=datetime.timedelta(days=1))
@util.debug.measure
def _stats_key_agg(ticker=None): return _iex_aggregator('get_key_stats', ticker)
_stats_key = lambda ticker: _stats_key_agg(ticker)

@cachier.cachier(stale_after=datetime.timedelta(days=7))
@util.debug.measure
def _stats_advanced_agg(ticker=None): return _iex_aggregator('get_advanced_stats', ticker)
_stats_advanced = lambda ticker: _stats_advanced_agg(ticker)
#'forwardPERatio', 'pegRatio', 'peHigh', 'peLow',
#'week52highDate', 'week52lowDate', 'putCallRatio',
#'week52high', 'week52low', 'week52highSplitAdjustOnly', 'week52highDateSplitAdjustOnly',
#'week52lowSplitAdjustOnly', 'week52lowDateSplitAdjustOnly', 'week52change',
#'float', 'avg10Volume', 'avg30Volume', 'day200MovingAvg', 'day50MovingAvg', 'employees',
#'ttmEPS', 'ttmDividendRate', 'dividendYield', 'nextDividendDate', 'exDividendDate',
#'nextEarningsDate', 'peRatio', 'maxChangePercent', 'year5ChangePercent', 'year2ChangePercent',
#'year1ChangePercent', 'ytdChangePercent', 'month6ChangePercent', 'month3ChangePercent',
#'month1ChangePercent', 'day30ChangePercent', 'day5ChangePercent'

IEX_KEY_STATS={
    'companyName', 'marketcap', 'week52high', 'week52low', 'week52highSplitAdjustOnly',
    'week52lowSplitAdjustOnly', 'week52change', 'sharesOutstanding', 'float', 'avg10Volume',
    'avg30Volume', 'day200MovingAvg', 'day50MovingAvg', 'employees', 'ttmEPS', 'ttmDividendRate',
    'dividendYield', 'nextDividendDate', 'exDividendDate', 'nextEarningsDate', 'peRatio', 'beta',
    'maxChangePercent', 'year5ChangePercent', 'year2ChangePercent', 'year1ChangePercent',
    'ytdChangePercent', 'month6ChangePercent', 'month3ChangePercent', 'month1ChangePercent',
    'day30ChangePercent', 'day5ChangePercent'
}
def stat(ticker, stat=None):
    if stat in IEX_KEY_STATS:
        data = _stats_key_agg(ticker)
    else:
        data = _stats_advanced_agg(ticker)

    return data if stat is None else data[stat]

def stats(ticker, key=None, advanced=False):
    if not advanced:
        data = _stats_advanced_agg(ticker)
    else:
        data = _stats_key_agg(ticker)

    return data if stat is None else data[stat]

def ebitda(ticker):
    '''
    Earnings Before Interest, Taxes, Depreciation and Amortization
    Used to measure the cash flow of a business.
    '''
    key = 'EBITDA'
    return D(stat(ticker, key))

def av(ticker, av):
    return D(stat(ticker, dict(
        a10v='avg10Volume',
        a30v='avg30Volume',
    )[av]))

def ma(ticker, ma):
    return D(stat(ticker, dict(
        d200ma='day200MovingAvg',
        d50ma='day50MovingAvg',
    )[ma]))

def cp(ticker, cp):
    return D(stat(ticker, dict(
        y5cp='year5ChangePercent',
        y2cp='year2ChangePercent',
        y1cp='year1ChangePercent',
        m6cp='month6ChangePercent',
        m3cp='month3ChangePercent',
        m1cp='month1ChangePercent',
        d30cp='day30ChangePercent',
        d5cp='day5ChangePercent',
    )[cp]))


def ev(ticker, ratio=None):
    '''
    Enterprise Value

    To calculate enterprise value, add the company's market capitalization to its outstanding
    preferred stock and all debt obligations, then subtract all of its cash and cash equivalents.
    '''

    if ratio is None:
        key = 'enterpriseValue'
    elif ratio == 'revenue':
        key = 'enterpriseValueToRevenue'

    return D(stat(ticker, key))

def beta(ticker):
    key = 'beta'
    return D(stat(ticker, key))

def shares_outstanding(ticker):
    key = 'sharesOutstanding'
    return D(stat(ticker, key))

def cash(ticker, total=True):
    if total:
        key = 'totalCash'

    return D(stat(ticker, key))

def debt(ticker, ratio=None):
    if ratio is None:
        key = 'currentDebt'
    else:
        key = 'debtToEquity'

    return D(stat(ticker, key))

def revenue(ticker, total=False, per=None):
    assert per is None or total is False

    if per == 'share':
        key = 'revenuePerShare'
    elif per == 'employee':
        key = 'revenuePerEmployee'
    elif total:
        key = 'totalRevenue'
    else:
        key = 'revenue'

    return D(stat(ticker, key))

def marketcap(ticker):
    key = 'marketcap'
    return D(stat(ticker, key))

def profit_margin(ticker):
    key = 'profitMargin'
    return D(stat(ticker, key))

def price(ticker, ratio=None):
    if ratio is None:
        return _price_agg(ticker)
    elif ratio == 'earnings2growth':
        key = 'pegRatio'
    elif ratio == 'earnings':
        key = 'peRatio'
    elif ratio == 'sales':
        key = 'priceToSales'
    elif ratio == 'book':
        key = 'priceToBook'

    return D(stat(ticker, key))


@cachier.cachier(stale_after=datetime.timedelta(days=90))
@util.debug.measure
def _sector_agg(ticker=None): return _iex_aggregator('get_sector', ticker)
sector = lambda ticker: _sector_agg(ticker)


@cachier.cachier(stale_after=datetime.timedelta(hours=12))
@util.debug.measure
def _quote_agg(ticker=None): return _iex_aggregator('get_quote', ticker)
quote = lambda ticker: _quote_agg(ticker)


@cachier.cachier(stale_after=datetime.timedelta(days=3))
@util.debug.measure
def _earnings_agg(ticker=None): return _iex_aggregator('get_earnings', ticker)
earnings = lambda ticker: _earnings_agg(ticker)


@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
@util.debug.measure
def _splits_agg(ticker=None): return _iex_aggregator('get_splits', ticker)
splits = lambda ticker: _splits_agg(ticker)


@cachier.cachier(stale_after=datetime.timedelta(hours=4))
@util.debug.measure
def _previous_day_prices_agg(ticker=None): return _iex_aggregator('get_previous_day_prices', ticker)
previous_day_prices = lambda ticker: _previous_day_prices_agg(ticker)


@cachier.cachier(stale_after=datetime.timedelta(days=1))
@util.debug.measure
def _insider_transactions_agg(ticker=None): return _iex_aggregator('get_insider_transactions', ticker)
insider_transactions = lambda ticker: _insider_transactions_agg(ticker)


# }=-
# IEX Other -={
@util.debug.measure
def iex_stock(ticker):
    global IEX_STOCKS
    if ticker in IEX_STOCKS:
        stock = IEX_STOCKS[ticker]
    else:
        stock = iex.Stock(ticker)
        IEX_STOCKS[ticker] = stock
    return stock


@cachier.cachier(stale_after=datetime.timedelta(hours=1))
@util.debug.measure
def news(ticker, source):
    return {
        'rh': lambda: rh.stocks.get_news(ticker),
        'iex': lambda: iex_stock(ticker).get_news()
    }[source]()
# }=-


@dataclasses.dataclass
class StockMultiplexor:
    iex: iex.Stock
    yf:  yf.Ticker


def __getattr__(ticker: str) -> iex.Stock:
    _ticker = ticker.upper()
    if _ticker not in symbols():
        raise NameError("name `%s' is not defined, or a valid ticker symbol" % ticker)

    return StockMultiplexor(
        iex=iex_stock(ticker),
        yf=yf.Ticker(ticker),
    )
