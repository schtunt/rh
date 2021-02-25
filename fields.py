import sys
import statistics
import scipy.stats
import pandas as pd
import datetime

import typing
import dataclasses

from functools import reduce
from collections import defaultdict
from collections import OrderedDict

import api
import util
import models
from util.numbers import NaN, D, Z
from util.datetime import NaT


@dataclasses.dataclass
class FieldComplexConstructor:
    attributes: list[str]
    chain:      list[typing.Callable]

@dataclasses.dataclass
class Field:
    name: typing.AnyStr             # Field name
    getter: FieldComplexConstructor # Getter function, or otherwise, a FieldComplexConstructor
    pullcast:  typing.Callable      # How to typecast the pulled data at inress
    pushcast: typing.Callable       # How to format the output data at presentation
    description: str                # Description of the field/column
    documentation: str              # URL to Investopedia describing the field

# Field Pullcast (Data Type / Typecasting) -={
_PULLCAST = dict(
    ticker=str,
    price=D, pcp=D, quantity=D, average_buy_price=D, equity=D, percent_change=D,
    equity_change=D, pe_ratio=D, percentage=D,
    type=str, name=str,
    cnt=D, trd=D, qty=D,
    crsq=D, crsv=D, crlq=D, crlv=D, cusq=D, cusv=D, culq=D, culv=D,
    pe_ratio2=D, pb_ratio=D,
    collateral_call=D, collateral_put=D,
    next_expiry=util.datetime.datetime,
    ttl=D, urgency=D, activities=str,
)
# }=-
# Field Pushcast (Data Presentation / Formating) -={
_PUSHCAST = {
    'ticker': lambda t: util.color.colhashbold(t, t[0]),
    #'since_open': util.color.mpct,
    #'since_close': util.color.mpct,
    #'CC.Coll': util.color.qty0,           # Covered Call Collateral
    #'CSP.Coll': util.color.mulla,         # Cash-Secured Put Collateral
    'price': util.color.mulla,
    'pcp': util.color.mulla,               # Previous Close Price (PCP)
    'quantity': util.color.qty0,
    'average_buy_price': util.color.mulla,
    'equity': util.color.mulla,
    'percent_change': util.color.pct,
    'equity_change': util.color.mulla,
    'pe_ratio': util.color.qty,
    'pb_ratio': util.color.qty,
    'percentage': util.color.pct,
    'delta': util.color.qty,
    'short': util.color.qty1,
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
    'urgency': util.color.mpctr,
    'next_expiry': lambda d: util.color.qty(util.datetime.ttl(d)),
}
#. }=-
# Field Extensions -={
_apidictplucker = lambda getter, key: lambda ticker: getter(ticker)[key]

# price, current stock price
# pcp, previous closing price
# ave buy price - your average according to robinhood
# *_change - total return ppercent/equity
# type - stock or adr

def _extensions(T, S):
    return [
        Field(
            name='industry',
            getter=api.industry,
            pullcast=str,
            pushcast=util.color.colhashwrap,
            description='Industry',
            documentation='https://www.investopedia.com/terms/i/industry.asp',
        ),
        Field(
            name='sector',
            getter=api.sector,
            pullcast=str,
            pushcast=util.color.colhashwrap,
            description='Sector',
            documentation='https://www.investopedia.com/terms/s/sector.asp',
        ),
        Field(
            name='ev',
            getter=api.ev,
            pullcast=D,
            pushcast=util.color.mulla,
            description='Enterprise Value',
            documentation='https://www.investopedia.com/ask/answers/111414/whats-difference-between-enterprise-value-and-market-capitalization.asp',
        ),
        Field(
            name='so',
            getter=api.shares_outstanding,
            pullcast=D,
            pushcast=util.color.qty0,
            description='Shares Outstanding',
            documentation='https://www.investopedia.com/terms/o/outstandingshares.asp',
        ),
        Field(
            name='marketcap',
            getter=api.market_cap,
            pullcast=D,
            pushcast=util.color.mulla,
            description='Market Capitalization',
            documentation='https://www.investopedia.com/terms/m/marketcapitalization.asp',
        ),
        Field(
            name='beta',
            getter=_apidictplucker(api.stats, 'beta'),
            pullcast=D,
            pushcast=util.color.qty,
            description='Beta',
            documentation='https://www.investopedia.com/terms/b/beta.asp',
        ),
        Field(
            name='premium_collected',
            getter=None,
            pullcast=D,
            pushcast=util.color.mulla,
            description='Options Premium Collected',
            documentation='',
        ),
        Field(
            name='dividends_collected',
            getter=None,
            pullcast=D,
            pushcast=util.color.mulla,
            description='Stock Dividends Collected',
            documentation='',
        ),
        Field(
            name='d50ma',
            getter=_apidictplucker(api.stats, 'day50MovingAvg'),
            pullcast=D,
            pushcast=util.color.mulla,
            description='50-Day Moving Average',
            documentation='',
        ),
        Field(
            name='d200ma',
            getter=_apidictplucker(api.stats, 'day200MovingAvg'),
            pullcast=D,
            pushcast=util.color.mulla,
            description='200-Day Moving Average',
            documentation='',
        ),
        Field(
            name='y5cp',
            getter=_apidictplucker(api.stats, 'year5ChangePercent'),
            pullcast=D,
            pushcast=util.color.pct,
            description='5-Year Percentage Change',
            documentation='',
        ),
        Field(
            name='y2cp',
            getter=_apidictplucker(api.stats, 'year2ChangePercent'),
            pullcast=D,
            pushcast=D,
            description='2-Year Percentage Change',
            documentation='',
        ),
        Field(
            name='y1cp',
            getter=_apidictplucker(api.stats, 'year1ChangePercent'),
            pullcast=D,
            pushcast=util.color.pct,
            description='1-Year Percentage Change',
            documentation='',
        ),
        Field(
            name='m6cp',
            getter=_apidictplucker(api.stats, 'month6ChangePercent'),
            pullcast=D,
            pushcast=util.color.pct,
            description='6-Month Percentage Change',
            documentation='',
        ),
        Field(
            name='m3cp',
            getter=_apidictplucker(api.stats, 'month3ChangePercent'),
            pullcast=D,
            pushcast=util.color.pct,
            description='3-Month Percentage Change',
            documentation='',
        ),
        Field(
            name='m1cp',
            getter=_apidictplucker(api.stats, 'month1ChangePercent'),
            pullcast=D,
            pushcast=util.color.pct,
            description='1-Month Percentage Change',
            documentation='',
        ),
        Field(
            name='d30cp',
            getter=_apidictplucker(api.stats, 'day30ChangePercent'),
            pullcast=D,
            pushcast=util.color.pct,
            description='30-Day Percentage Change',
            documentation='',
        ),
        Field(
            name='d5cp',
            getter=_apidictplucker(api.stats, 'day5ChangePercent'),
            pullcast=D,
            pushcast=util.color.pct,
            description='5-Day Percentage Change',
            documentation='',
        ),
        Field(
            name='change',
            getter=_apidictplucker(api.quote, 'changePercent'),
            pullcast=D,
            pushcast=util.color.pct,
            description='Change Percent',
            documentation='',
        ),
        Field(
            name='ma',
            getter=FieldComplexConstructor(
                attributes=(
                    'd200ma', 'd50ma', 'price',
                ),
                chain=(
                    lambda R: R.values(),
                    lambda R: util.numbers.growth_score(R),
                ),
            ),
            pullcast=D,
            pushcast=util.color.qty,
            description='Moving Average',
            documentation='',
        ),
        Field(
            name='cbps',
            getter=FieldComplexConstructor(
                attributes=(
                    'ticker',
                ),
                chain=(
                    lambda R: T[T['symbol']==R['ticker']]['cbps'].values[-1],
                )
            ),
            pullcast=D,
            pushcast=util.color.mulla,
            description='Cost-Basis per Share (Excluding Option Premiums and Dividends',
            documentation='https://www.investopedia.com/terms/c/costbasis.asp',
        ),
        Field(
            name='cbps%',
            getter=FieldComplexConstructor(
                attributes=(
                    'ticker',
                ),
                chain=(
                    lambda R: T[T['symbol']==R['ticker']]['cbps'].values[-1] / (
                        S[S['ticker']==R['ticker']].price.item()
                    ),
                )
            ),
            pullcast=D,
            pushcast=util.color.pct,
            description='Cost-Basis per Share (as a percentage of share price)',
            documentation='https://www.investopedia.com/terms/c/costbasis.asp',
        ),
        Field(
            name='trd0',
            getter=FieldComplexConstructor(
                attributes=(
                    'ticker',
                ),
                chain=(
                    lambda R: T[T['symbol']==R['ticker']].date,
                    lambda dates: min(dates) if len(dates) else NaT,
                )
            ),
            pullcast=util.datetime.datetime,
            pushcast=lambda d: util.color.qty0(util.datetime.age(d)),
            description='Day-Zero Trade',
            documentation='',
        ),
        Field(
            name='momentum',
            getter=FieldComplexConstructor(
                attributes=(
                    'ticker',
                    'y5cp', 'y2cp', 'y1cp', 'm6cp', 'm3cp', 'm1cp', 'd30cp', 'd5cp',
                ),
                chain=(
                    lambda R: { R.pop('ticker'): R },
                    lambda R: [
                        scipy.stats.percentileofscore(
                            S[period].to_numpy(),
                            pd.to_numeric(percentile),
                        ) for ticker, percentiles in R.items()
                        for period, percentile in percentiles.items()
                    ],
                    statistics.mean,
                )
            ),
            pullcast=D,
            pushcast=util.color.pct,
            description='Momentum Percentile (compared to the rest of this Portfolio)',
            documentation='',
        ),
        Field(
            name='sharpe',
            getter=FieldComplexConstructor(
                attributes=(
                    'ticker',
                ),
                chain=(
                    lambda R: models.sharpe(ticker=R['ticker']),
                )
            ),
            pullcast=D,
            pushcast=util.color.pct,
            description='Sharpe Ratio',
            documentation='https://www.investopedia.com/terms/s/sharperatio.asp',
        ),
        Field(
            name='treynor',
            getter=FieldComplexConstructor(
                attributes=(
                    'ticker', 'beta'
                ),
                chain=(
                    lambda R: models.treynor(R['ticker'], R['beta']),
                )
            ),
            pullcast=D,
            pushcast=util.color.pct,
            description='Treynor Ratio',
            documentation='https://www.investopedia.com/terms/t/treynorratio.asp',
        ),
     ]

# }=-

def formatters():
    '''The formatters control how the data is displayed at presentation time'''

    formatters = dict(_PUSHCAST.items())
    formatters.update(
        {field.name: field.pushcast for field in _extensions(None, None)}
    )
    return defaultdict(lambda: str, formatters)

def _typecasters():
    '''
    The typecasters control how the data is imported.  This includes only the
    base fields.  The old style of defining base fields was through _PULLCAST.
    The new style is via Field objects that have their getters set to None.
    '''

    typecasters = dict(_PULLCAST.items())
    typecasters.update(
        {
            field.name: field.pullcast for field in _extensions(None, None)
                if field.getter is None
        }
    )
    return defaultdict(lambda: str, typecasters)

def _extend(S, field, prioritize_missing):
    figet = field.getter
    cast = field.pullcast
    try:
        if figet is None:
            S[field.name] = S[field.name].apply(cast)
        elif type(figet) is not FieldComplexConstructor:
            # prioritize_missing implies user has not requested specific stocks, i.e., a
            # targetted query (so fresh data is desired)
            series = []
            for ticker in S['ticker']:
                if field.name not in S.columns:
                    series.append(cast(figet(ticker)))
                elif prioritize_missing and S[field.name] in (None, NaN, NaT, 'N/A'):
                    series.append(cast(figet(ticker, ignore_cache=True)))
                elif not prioritize_missing: # i.e., no priority at all, do all tickers
                    series.append(cast(figet(ticker, ignore_cache=True)))
                else:
                    # TODO: Add age to everu row, this final block will refresh based on age
                    pass
            S[field.name] = series
        else:
            S[field.name] = pd.Series(list(
                reduce(
                    lambda g, f: f(g),
                    figet.chain + (field.pullcast,),
                    OrderedDict(zip(figet.attributes, components)),
                ) for components in zip(
                    *map(
                        lambda attr: getattr(S, attr),
                        figet.attributes
                    )
                )
            ), index=S.index)
    except Exception as e:
        ErrorClass = type(e)
        message = "Failed to extend field `%s'" % field.name
        exception = ErrorClass('%s; %s' % (str(e), message))
        raise exception.with_traceback(sys.exc_info()[2])


class Fields:
    def __init__(self, data, T, df=None):
        # Some APIs (free ones) will limit requests.  To make this useable, if the user has
        # requested only a handful of tickers in their query, then to to hit the API unlit
        # throttled.  Otherwise, prioritize missing ticker data over old ticker data, to avoid
        # refreshing the same tickers over and over again.  This does not apply for the field
        # type `FieldComplexConstructor', since those should not be making direct API calls,
        # but create new fields using existing ones.
        tickers_requested = len(data)
        prioritize_missing = tickers_requested > 5


        # Create new DataFrame first.  This DataFrame will be limited to the list
        # of tickers supplied by the user, if any, otherwise as inclusive as the
        # stored all-stock DataFrame.
        typecasts = _typecasters()
        S = pd.DataFrame(
            map(
                lambda row: map(
                    lambda typecast: typecast[1](row[typecast[0]]),
                    typecasts.items()
                ), data
            ),
            columns=typecasts.keys()
        )

        # Extend fields/columns
        for field in _extensions(T, S):
            _extend(S, field, prioritize_missing)

        # Sort the DataFrame by Ticker
        S.sort_values(by='ticker', inplace=True)

        if df is not None:
            df.set_index('ticker', inplace=True)
            df.update(S.set_index('ticker'))
            df.reset_index(inplace=True)
            self._S = df
        else:
            self._S = S


    @property
    def extended(self):
        return self._S
