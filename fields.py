import statistics
import scipy.stats
import pandas as pd
import datetime

from functools import reduce
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict

import api
import util
import models
from util.numbers import NaT
from util.numbers import D

FieldComplexConstructor = namedtuple(
    'FieldComplexConstructor', [
        'attributes',
        'chain',
    ]
)

Field = namedtuple(
    'DataField', [
        'name',
        'getter',
        'pullcast',
        'pushcast',
        'description',
        'documentation',
    ]
)

# Field Pullcast (Data Type / Typecasting) -={
_PULLCAST = dict(
    ticker=str,
    price=D, pcp=D, quantity=D, average_buy_price=D, equity=D, percent_change=D,
    equity_change=D, pe_ratio=D, percentage=D,
    type=str, name=str,
    cnt=D, trd=D, qty=D, esp=D,
    crsq=D, crsv=D, crlq=D, crlv=D, cusq=D, cusv=D, culq=D, culv=D,
    premium_collected=D, dividends_collected=D, pe_ratio2=D, pb_ratio=D,
    collateral_call=D, collateral_put=D,
    next_expiry=datetime.datetime,
    ttl=D, urgency=D, activities=str,
)
# }=-
# Field Pushcast (Data Presentation / Formating) -={
_PUSHCAST = {
    #'since_open': util.color.mpct,
    #'since_close': util.color.mpct,
    #'CC.Coll': util.color.qty0,                   # Covered Call Collateral
    #'CSP.Coll': util.color.mulla,                 # Cash-Secured Put Collateral
    'price': util.color.mulla,
    'pcp': util.color.mulla,                      # Previous Close Price (PCP)
    'esp': util.color.mulla,                      # Effective Share Price
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
    'premium_collected': util.color.mulla,
    'dividends_collected': util.color.mulla,
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
    'next_expiry': util.datetime.ttl,
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
            name='d50ma',
            getter=_apidictplucker(api.stats, 'day50MovingAvg'),
            pullcast=D,
            pushcast=util.color.qty,
            description='50-Day Moving Average',
            documentation='',
        ),
        Field(
            name='d200ma',
            getter=_apidictplucker(api.stats, 'day200MovingAvg'),
            pullcast=D,
            pushcast=util.color.qty,
            description='200-Day Moving Average',
            documentation='',
        ),
        Field(
            name='y5cp',
            getter=_apidictplucker(api.stats, 'year5ChangePercent'),
            pullcast=D,
            pushcast=D,
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
            pushcast=D,
            description='1-Year Percentage Change',
            documentation='',
        ),
        Field(
            name='m6cp',
            getter=_apidictplucker(api.stats, 'month6ChangePercent'),
            pullcast=D,
            pushcast=D,
            description='6-Month Percentage Change',
            documentation='',
        ),
        Field(
            name='m3cp',
            getter=_apidictplucker(api.stats, 'month3ChangePercent'),
            pullcast=D,
            pushcast=D,
            description='3-Month Percentage Change',
            documentation='',
        ),
        Field(
            name='m1cp',
            getter=_apidictplucker(api.stats, 'month1ChangePercent'),
            pullcast=D,
            pushcast=D,
            description='1-Month Percentage Change',
            documentation='',
        ),
        Field(
            name='d30cp',
            getter=_apidictplucker(api.stats, 'day30ChangePercent'),
            pullcast=D,
            pushcast=D,
            description='30-Day Percentage Change',
            documentation='',
        ),
        Field(
            name='d5cp',
            getter=_apidictplucker(api.stats, 'day5ChangePercent'),
            pullcast=D,
            pushcast=D,
            description='5-Day Percentage Change',
            documentation='',
        ),
        Field(
            name='ma',
            getter=FieldComplexConstructor(
                attributes=(
                    'd200ma', 'd50ma', 'price',
                ),
                chain=(
                    lambda data: data.values(),
                    lambda data: util.numbers.growth_score(data),
                ),
            ),
            pullcast=D,
            pushcast=util.color.qty,
            description='Moving Average',
            documentation='',
        ),
        Field(
            name='trd0',
            getter=FieldComplexConstructor(
                attributes=(
                    'ticker',
                ),
                chain=(
                    lambda data: T[T['symbol'] == data['ticker']].date,
                    lambda dates: min(dates) if len(dates) else NaT,
                )
            ),
            pullcast=D,
            pushcast=lambda d: util.color.qty0(util.datetime.age(d)),
            description='Day-Zero Trade',
            documentation='',
        ),
        Field(
            name='change',
            getter=FieldComplexConstructor(
                attributes=(
                    'price', 'pcp',
                ),
                chain=(
                    lambda data: 100 * (data['price'] / data['pcp'] - 1),
                )
            ),
            pullcast=D,
            pushcast=util.color.qty,
            description='Moving Average',
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
                    lambda data: { data.pop('ticker'): data },
                    lambda data: [
                        scipy.stats.percentileofscore(
                            S[period].to_numpy(),
                            pd.to_numeric(percentile),
                        ) for ticker, percentiles in data.items()
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
                    lambda data: models.sharpe(ticker=data['ticker']),
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
                    lambda data: models.treynor(data['ticker'], data['beta']),
                )
            ),
            pullcast=D,
            pushcast=util.color.pct,
            description='Treynor Ratio',
            documentation='https://www.investopedia.com/terms/t/treynorratio.asp',
        ),
     ]

def _extend(S, field):
    figet = field.getter
    cast = field.pullcast
    if type(figet) is not FieldComplexConstructor:
        S[field.name] = [
            cast(figet(ticker)) for ticker in S['ticker']
        ]
    else:
        S[field.name] = pd.Series([
            reduce(
                lambda g, f: f(g),
                figet.chain,
                OrderedDict(zip(figet.attributes, components)),
            ) for components in zip(
                *map(
                    lambda attr: getattr(S, attr),
                    figet.attributes
                )
            )
        ], index=S.index)
# }=-

def formatters():
    _formatters = list(_PUSHCAST.items())
    _formatters.extend((field.name, field.pushcast) for field in _extensions(None, None))
    return defaultdict(lambda: None, _formatters)

class Fields:
    @property
    def extended(self):
        return self._S

    def __init__(self, data, T):
        # The header for the rows represented in `data', keyed by column header, and the
        # value representing the data type of that column.
        global _PULLCAST
        self._header = _PULLCAST

        # The formatters control how the data is displayed at presentation time
        global _PUSHCAST
        self._formatters = _PUSHCAST

        # Create the new `Stocks' DataFrame here for the first time, using the passed in python
        # list of dicts, each representing a new row.  This is the starting point.
        S = pd.DataFrame(
            map(lambda row: map(lambda key: row[key], self._header), data),
            columns=self._header
        )

        # Extend fields/columns
        for field in _extensions(T, S):
            _extend(S, field)

        # Sort the DataFrame by Ticker
        S = S.sort_values(by='ticker')

        # Assign & Delete Excess
        self._S = S
