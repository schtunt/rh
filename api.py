import os
import sys
import typing
import pathlib
import datetime
import dataclasses
import operator

from collections import defaultdict
from functools import reduce

import util
from util.numbers import F, NaN

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

def blacklist(full=False):
    '''
    Modes:
     - full: remove expired and old tickers
     - half: remove expired tickers only
    '''
    blacklist = set((
        'AABAZZ',
    ))

    blacklist |= set(
        reduce(
            operator.or_,
            (TICKER_CHAIN.get(ticker, set()) for ticker in blacklist)
        )
    )

    if not full:
        return blacklist

    return blacklist | set(
        tocker for chain in TICKER_CHAIN.values() for tocker in chain
    )


blacklisted = lambda ticker, full=True: bool(ticker in blacklist(full))
whitelisted = lambda ticker, full=False: bool(ticker not in blacklist(full))

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
    if iex_api_key[0] == 'T':
        os.environ['IEX_API_VERSION'] = 'sandbox'

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

@cachier.cachier(stale_after=datetime.timedelta(days=100))
def industry(ticker):
    overview = _overview(ticker)
    return overview['Industry'] if overview is not None else 'N/A'

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
def symbols(remove='old+expired'):
    '''
    The `superset' flag will also add older symbols, but not expired ones.
    '''

    symbols = set(
        map(
            instrument2symbol,
            positions('stocks', 'all', info='instrument'),
        )
    ) | set(
        holdings().keys()
    )

    if remove is not None:
        remove = remove.split('+')

    if remove is None:
        pass
    elif 'expired' in remove and 'old' in remove:
        # remove expired and old
        symbols -= blacklist(full=True)
    elif 'expired' in remove:
        # remove just expired
        symbols -= blacklist(full=False)
    elif 'old' in remove:
        # remove just old (no use-case for this)
        #symbols -= (blacklist(full=True) - blacklist(full=False))
        raise RuntimeError('Why?')

    return tuple(sorted(list(symbols)))

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

def float(ticker):
    return F(fundamentals(ticker)['float'])

# }=-

@cachier.cachier(stale_after=datetime.timedelta(days=14))
@util.debug.measure
def events(ticker):
    return rh.stocks.get_events(ticker)



# IEX Aggregation -={
@cachier.cachier(stale_after=datetime.timedelta(hours=3))
@util.debug.measure
def _iex_aggregator(fn_name):
    agg = dict()
    for chunk in util.chunk(symbols(remove='expired'), 100):
        fn = getattr(iex.Stock(list(chunk)), fn_name)
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

    if fn_name == 'get_financials':
        # Yes, iexfinance is returning this mess for financials, which we need to correct
        return {
            ticker: datum.get('financials', [None])[0]
            for ticker, datum in agg.items()
        }
    else:
        return {ticker: datum for ticker, datum in agg.items()}


@cachier.cachier(stale_after=datetime.timedelta(days=1))
@util.debug.measure
def _prices_last_year_agg():
    when = util.datetime.short(util.datetime.lastyear())
    prices = pdr.data.DataReader(
        symbols, data_source='yahoo', start=when, end=when
    )['Adj Close'].values[0]
    return dict(zip(symbols(), map(F, prices)))
_prices_last_year = lambda ticker: _prices_last_year_agg()[ticker]


@cachier.cachier(stale_after=datetime.timedelta(hours=3))
@util.debug.measure
def _price_agg(): return { t: F(p) for t, p in _iex_aggregator('get_price').items() }


@cachier.cachier(stale_after=datetime.timedelta(days=1))
@util.debug.measure
def _stats_key_agg(): return _iex_aggregator('get_key_stats')
_stats_key = lambda ticker: _stats_key_agg()[ticker]

@cachier.cachier(stale_after=datetime.timedelta(days=7))
@util.debug.measure
def _stats_advanced_agg(): return _iex_aggregator('get_advanced_stats')
_stats_advanced = lambda ticker: _stats_advanced_agg()[ticker]
#'forwardPERatio', 'pegRatio', 'peHigh', 'peLow',
#'week52highDate', 'week52lowDate', 'putCallRatio',
#'week52high', 'week52low', 'week52highSplitAdjustOnly', 'week52highDateSplitAdjustOnly',
#'week52lowSplitAdjustOnly', 'week52lowDateSplitAdjustOnly', 'week52change',
#'avg10Volume', 'avg30Volume', 'day200MovingAvg', 'day50MovingAvg', 'employees',
#'ttmEPS', 'ttmDividendRate', 'dividendYield', 'nextDividendDate', 'exDividendDate',
#'nextEarningsDate', 'peRatio', 'maxChangePercent', 'year5ChangePercent', 'year2ChangePercent',
#'year1ChangePercent', 'ytdChangePercent', 'month6ChangePercent', 'month3ChangePercent',
#'month1ChangePercent', 'day30ChangePercent', 'day5ChangePercent'

#IEX_KEY_STATS={
#    'companyName', 'marketcap', 'week52high', 'week52low', 'week52highSplitAdjustOnly',
#    'week52lowSplitAdjustOnly', 'week52change', 'sharesOutstanding', 'avg10Volume',
#    'avg30Volume', 'day200MovingAvg', 'day50MovingAvg', 'employees', 'ttmEPS', 'ttmDividendRate',
#    'dividendYield', 'nextDividendDate', 'exDividendDate', 'nextEarningsDate', 'peRatio', 'beta',
#    'maxChangePercent', 'year5ChangePercent', 'year2ChangePercent', 'year1ChangePercent',
#    'ytdChangePercent', 'month6ChangePercent', 'month3ChangePercent', 'month1ChangePercent',
#    'day30ChangePercent', 'day5ChangePercent'
#}
def stats(ticker=None):
    '''
    Advanced stats are expensive, and so cached for longer periods.  The basic stats are
    a subset of the advanced stats, cheaper, and so cached for shoter periods.  Here we
    mask the two calls, take the advanced stats, and update it with the fresher, albeir,
    crapier basic "key" stats.
    '''
    advanced = _stats_advanced_agg()
    basic = _stats_key_agg()
    merged = {ticker: datum|advanced[ticker] for ticker,datum in advanced.items()}
    return merged if ticker is None else merged[ticker]

def ebitda(ticker):
    '''
    Earnings Before Interest, Taxes, Depreciation and Amortization
    Used to measure the cash flow of a business.
    '''
    key = 'EBITDA'
    return F(stats(ticker)[key])

def ebit(ticker):
    '''
    Earnings Before Interest and Taxes; also referred to as `operating earnings',
    `operating profit', and `profit before interest and taxes'.

    EBIT is used to measure a firm's operating income.

    EBIT can be calculated as revenue minus expenses excluding tax and interest.
    '''

    key = 'ebit'
    f = financials(ticker)
    return NaN if f is None else F(f[key])

def ebt(ticker):
    '''
    Earnings Before Taxes
    While used for essentially the same purpose as EBIT, this represents a firm's operating
    income after accounting for expenses outside of the firm's control.
    '''
    return

def dividends_paid(ticker):
    '''
    This is dividends paid by the company, nothing to do with Robinhood profile and individual's
    account.
    '''
    key = 'dividendsPaid'
    f = financials(ticker)
    return NaN if f is None else F(f[key])

def income(ticker):
    key = 'netIncome'
    f = financials(ticker)
    return NaN if f is None else F(f[key])

def liabilities(ticker):
    key = 'totalLiabilities'
    return F(financials(ticker)[key])
    f = financials(ticker)
    return NaN if f is None else F(f[key])

def debt(ticker):
    key = 'totalDebt'
    f = financials(ticker)
    return NaN if f is None else F(f[key])

def assets(ticker):
    key = 'totalAssets'
    f = financials(ticker)
    return NaN if f is None else F(f[key])

def cash(ticker):
    key = 'totalCash'
    f = financials(ticker)
    return NaN if f is None else F(f[key])

def equity(ticker):
    key = 'shareholderEquity'
    f = financials(ticker)
    return NaN if f is None else F(f[key])

def roc(ticker):
    '''
    Rate of Change
    '''
    p0 = _prices_last_year(ticker)
    pNow = price(ticker)
    return

def roic(ticker):
    '''
    Return On Invested Capital

    roic = (net income - dividend) / (debt + equity)
    '''
    return (
        income(ticker) - dividends_paid(ticker)
    ) / (
        debt(ticker) + equity(ticker)
    )

def av(ticker, av):
    return F(stats(ticker)[dict(
        a10v='avg10Volume',
        a30v='avg30Volume',
    )[av]])

def ma(ticker, ma):
    return F(stats(ticker)[dict(
        d200ma='day200MovingAvg',
        d50ma='day50MovingAvg',
    )[ma]])

def cp(ticker, cp):
    return F(stats(ticker)[dict(
        y5cp='year5ChangePercent',
        y2cp='year2ChangePercent',
        y1cp='year1ChangePercent',
        m6cp='month6ChangePercent',
        m3cp='month3ChangePercent',
        m1cp='month1ChangePercent',
        d30cp='day30ChangePercent',
        d5cp='day5ChangePercent',
    )[cp]])


def beta(ticker):
    key = 'beta'
    return F(stats(ticker)[key])

def shares_outstanding(ticker):
    key = 'sharesOutstanding'
    return F(stats(ticker)[key])

def cash(ticker, total=True):
    if total: key = 'totalCash'
    return F(stats(ticker)[key])

def debt(ticker, ratio=None):
    if ratio is None: key = 'currentDebt'
    else: key = 'debtToEquity'

    return F(stats(ticker)[key])

def revenue(ticker, total=False, per=None):
    assert per is None or total is False

    if per == 'share': key = 'revenuePerShare'
    elif per == 'employee': key = 'revenuePerEmployee'
    elif total: key = 'totalRevenue'
    else: key = 'revenue'

    return F(stats(ticker)[key])

def marketcap(ticker):
    key = 'marketcap'
    return F(stats(ticker)[key])

def profit_margin(ticker):
    key = 'profitMargin'
    return F(stats(ticker)[key])

def price(ticker, ratio=None):
    if ratio is None: return _price_agg()[ticker]
    elif ratio == 'peg': key = 'pegRatio'
    elif ratio == 'p2e': key = 'peRatio'
    elif ratio == 'p2s': key = 'priceToSales'
    elif ratio == 'p2b': key = 'priceToBook'

    return F(stats(ticker)[key])

def ev(ticker, ratio=None):
    '''
    Enterprise Value

    To calculate enterprise value, add the company's market capitalization to its outstanding
    preferred stock and all debt obligations, then subtract all of its cash and cash equivalents.
    '''

    numerator = 'enterpriseValue'
    denominator = None

    if ratio == 'ev2r':
        numerator = 'enterpriseValueToRevenue'
    elif ratio == 'ev2gp':
        numerator = 'enterpriseValue'
        denominator = 'grossProfit'
    elif ratio == 'ev2ebitda':
        denominator = 'EBITDA'
    elif ratio == 'ebit2ev':
        denominator = numerator
        numerator = 'EBIT'

    s = stats(ticker)
    return F(s[numerator]) / F(
        1 if denominator is None else s[denominator]
    )

@cachier.cachier(stale_after=datetime.timedelta(days=90))
@util.debug.measure
def _sector_agg(): return _iex_aggregator('get_sector')
sector = lambda ticker: _sector_agg()[ticker]


@cachier.cachier(stale_after=datetime.timedelta(weeks=4))
@util.debug.measure
def _financials_agg(): return _iex_aggregator('get_financials')
financials = lambda ticker: _financials_agg()[ticker]


@cachier.cachier(stale_after=datetime.timedelta(hours=12))
@util.debug.measure
def _quote_agg(): return _iex_aggregator('get_quote')
quote = lambda ticker: _quote_agg()[ticker]


@cachier.cachier(stale_after=datetime.timedelta(days=7))
@util.debug.measure
def _earnings_agg(): return _iex_aggregator('get_earnings')
earnings = lambda ticker: _earnings_agg()[ticker]


@cachier.cachier(stale_after=datetime.timedelta(weeks=1))
@util.debug.measure
def _splits_agg(): return _iex_aggregator('get_splits')
splits = lambda ticker: _splits_agg().get(ticker, [])


@cachier.cachier(stale_after=datetime.timedelta(hours=4))
@util.debug.measure
def _previous_day_prices_agg(): return _iex_aggregator('get_previous_day_prices')
previous_day_prices = lambda ticker: _previous_day_prices_agg()[ticker]


@cachier.cachier(stale_after=datetime.timedelta(days=4))
@util.debug.measure
def _insider_transactions_agg(): return _iex_aggregator('get_insider_transactions')
insider_transactions = lambda ticker: _insider_transactions_agg()[ticker]


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
    if _ticker not in symbols(remove=None):
        raise NameError("name `%s' is not defined, or a valid ticker symbol" % ticker)

    return StockMultiplexor(
        iex=iex_stock(ticker),
        yf=yf.Ticker(ticker),
    )
