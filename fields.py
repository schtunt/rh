import sys
import statistics
import scipy as sp
import pandas as pd
import numpy as np
import datetime

import typing
import decimal
import dataclasses

from functools import reduce
from collections import defaultdict
from collections import OrderedDict

import api
import util
import models
from util.numbers import NaN, F
from util.datetime import NaT

def xxx(args):
    print('#'*80 + ' START')
    print()
    util.debug.ddump(args, force=True)
    print()
    print('#'*80 + ' END')
    return args

@dataclasses.dataclass
class FieldComplexConstructor:
    attributes: list[str]
    chain:      list[typing.Callable]

@dataclasses.dataclass
class Field:
    name: typing.AnyStr             # Field name
    getter: FieldComplexConstructor # Getter function, or otherwise, a FieldComplexConstructor
    pullcast: typing.Callable       # How to typecast the pulled data at inress
    pushcast: typing.Callable       # How to format the output data at presentation
    description: str                # Description of the field/column
    documentation: str              # URL to Investopedia describing the field

# Field Pullcast (Data Type / Typecasting) -={
_PULLCAST = dict(
    ticker=str,
    price=F, pcp=F, quantity=F, average_buy_price=F, equity=F, percent_change=F,
    equity_change=F,
    percentage=F,
    type=str, name=str,
    cnt=F, trd=F, qty=F,
    collateral_call=F, collateral_put=F,
    next_expiry=util.datetime.datetime,
    ttl=F, urgency=F, activities=str,
)
# }=-
# Field Pushcast (Data Presentation / Formating) -={
_PUSHCAST = {
    'ticker': lambda t: util.color.colhashbold(t, t[0]),
    'price': util.color.mulla,
    'pcp': util.color.mulla,               # Previous Close Price (PCP)
    'quantity': util.color.qty0,
    'average_buy_price': util.color.mulla,
    'equity': util.color.mulla,
    'percent_change': util.color.pct,
    'equity_change': util.color.mulla,
    'percentage': util.color.pct,
    'delta': util.color.qty,
    'short': util.color.qty1,
    'urgency': util.color.mpctr,
    'next_expiry': lambda d: util.color.qty(util.datetime.ttl(d)),
}
#. }=-
# Field Extensions -={
# price, current stock price
# pcp, previous closing price
# ave buy price - your average according to robinhood
# *_change - total return ppercent/equity
# type - stock or adr

def _extensions(T, S):
    return [
        Field(
            name='roic',
            getter=api.roic,
            pullcast=F,
            pushcast=util.color.pct,
            description='ROIC',
            documentation='https://www.investopedia.com/terms/r/returnoninvestmentcapital.asp',
        ),
        Field(
            name='ebt',
            getter=api.ebt,
            pullcast=F,
            pushcast=util.color.pct,
            description='EBT',
            documentation='https://www.investopedia.com/terms/e/ebt.asp',
        ),
        Field(
            name='ebit',
            getter=api.ebit,
            pullcast=F,
            pushcast=util.color.pct,
            description='EBIT',
            documentation='https://www.investopedia.com/terms/e/ebit.asp',
        ),
        Field(
            name='ebitda',
            getter=api.ebitda,
            pullcast=F,
            pushcast=util.color.pct,
            description='EBITDA',
            documentation='https://www.investopedia.com/terms/e/ebitda.asp',
        ),
        Field(
            name='p2e',
            getter=lambda ticker: api.price(ticker, ratio='p2e'),
            pullcast=F,
            pushcast=util.color.pct,
            description='P/E Ratio',
            documentation='https://www.investopedia.com/terms/i/industry.asp',
        ),
        Field(
            name='p2b',
            getter=lambda ticker: api.price(ticker, ratio='p2b'),
            pullcast=F,
            pushcast=util.color.pct,
            description='P/B Ratio',
            documentation='https://www.investopedia.com/terms/i/industry.asp',
        ),
        Field(
            name='p2s',
            getter=lambda ticker: api.price(ticker, ratio='p2s'),
            pullcast=F,
            pushcast=util.color.pct,
            description='P/S Ratio',
            documentation='https://www.investopedia.com/terms/i/industry.asp',
        ),
        Field(
            name='peg',
            getter=lambda ticker: api.price(ticker, ratio='peg'),
            pullcast=F,
            pushcast=util.color.pct,
            description='PEG Ratio',
            documentation='https://www.investopedia.com/terms/i/industry.asp',
        ),
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
            pullcast=F,
            pushcast=util.color.mulla,
            description='Enterprise Value',
            documentation='https://www.investopedia.com/ask/answers/111414/whats-difference-between-enterprise-value-and-market-capitalization.asp',
        ),
        Field(
            name='so',
            getter=api.shares_outstanding,
            pullcast=F,
            pushcast=util.color.qty0,
            description='Shares Outstanding',
            documentation='https://www.investopedia.com/terms/o/outstandingshares.asp',
        ),
        Field(
            name='marketcap',
            getter=api.marketcap,
            pullcast=F,
            pushcast=util.color.mulla,
            description='Market Capitalization',
            documentation='https://www.investopedia.com/terms/m/marketcapitalization.asp',
        ),
        Field(
            name='beta',
            getter=api.beta,
            pullcast=F,
            pushcast=util.color.qty,
            description='Beta',
            documentation='https://www.investopedia.com/terms/b/beta.asp',
        ),
        Field(
            name='premium_collected',
            getter=None,
            pullcast=F,
            pushcast=util.color.mulla,
            description='Options Premium Collected',
            documentation='',
        ),
        Field(
            name='dividends_collected',
            getter=None,
            pullcast=F,
            pushcast=util.color.mulla,
            description='Stock Dividends Collected',
            documentation='',
        ),
        Field(
            name='d50ma',
            getter=lambda ticker: api.ma(ticker, 'd50ma'),
            pullcast=F,
            pushcast=util.color.mulla,
            description='50-Day Moving Average',
            documentation='',
        ),
        Field(
            name='d200ma',
            getter=lambda ticker: api.ma(ticker, 'd200ma'),
            pullcast=F,
            pushcast=util.color.mulla,
            description='200-Day Moving Average',
            documentation='',
        ),
        Field(
            name='y5cp',
            getter=lambda ticker: api.cp(ticker, 'y5cp'),
            pullcast=F,
            pushcast=util.color.pct,
            description='5-Year Percentage Change',
            documentation='',
        ),
        Field(
            name='y2cp',
            getter=lambda ticker: api.cp(ticker, 'y2cp'),
            pullcast=F,
            pushcast=F,
            description='2-Year Percentage Change',
            documentation='',
        ),
        Field(
            name='y1cp',
            getter=lambda ticker: api.cp(ticker, 'y1cp'),
            pullcast=F,
            pushcast=util.color.pct,
            description='1-Year Percentage Change',
            documentation='',
        ),
        Field(
            name='m6cp',
            getter=lambda ticker: api.cp(ticker, 'm6cp'),
            pullcast=F,
            pushcast=util.color.pct,
            description='6-Month Percentage Change',
            documentation='',
        ),
        Field(
            name='m3cp',
            getter=lambda ticker: api.cp(ticker, 'm3cp'),
            pullcast=F,
            pushcast=util.color.pct,
            description='3-Month Percentage Change',
            documentation='',
        ),
        Field(
            name='m1cp',
            getter=lambda ticker: api.cp(ticker, 'm1cp'),
            pullcast=F,
            pushcast=util.color.pct,
            description='1-Month Percentage Change',
            documentation='',
        ),
        Field(
            name='d30cp',
            getter=lambda ticker: api.cp(ticker, 'd30cp'),
            pullcast=F,
            pushcast=util.color.pct,
            description='30-Day Percentage Change',
            documentation='',
        ),
        Field(
            name='d5cp',
            getter=lambda ticker: api.cp(ticker, 'd5cp'),
            pullcast=F,
            pushcast=util.color.pct,
            description='5-Day Percentage Change',
            documentation='',
        ),
        Field(
            name='change',
            getter=lambda ticker: api.quote(ticker)['changePercent'],
            pullcast=F,
            pushcast=util.color.pct,
            description='Change Percent',
            documentation='',
        ),
        Field(
            name='malps%',
            getter=FieldComplexConstructor(
                attributes=(
                    'd200ma', 'd50ma', 'price',
                ),
                chain=(
                    lambda R: list(R.values()),
                    lambda R: -util.numbers.growth_score(R) / R[-1],
                ),
            ),
            pullcast=F,
            pushcast=util.color.mpct,
            description='Moving average loss per dollar of share',
            documentation='',
        ),
        Field(
            name='cbps',
            getter=FieldComplexConstructor(
                attributes=(
                    'ticker',
                ),
                chain=(
                    lambda R: T[T['symbol']==R['ticker']],
                    lambda df: df['cbps'].values[-1],
                )
            ),
            pullcast=F,
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
                    lambda R: (
                        T[T['symbol']==R['ticker']]['cbps'].values[-1]
                    ) / (
                        S[S['ticker']==R['ticker']].price.item()
                    ),
                )
            ),
            pullcast=F,
            pushcast=util.color.mpct,
            description='Cost-Basis per Share (as a percentage of share price)',
            documentation='https://www.investopedia.com/terms/c/costbasis.asp',
        ),
        Field(
            name='dyps%',
            getter=FieldComplexConstructor(
                attributes=(
                    'ticker',
                ),
                chain=(
                    lambda R: R['ticker'],
                    lambda ticker: dict(
                        price=np.mean((
                            api.ma(ticker, ma='d200ma'),
                            api.ma(ticker, ma='d50ma'),
                            api.price(ticker),
                        )),
                        divs=list(zip(*(
                            (F(div['amount']), F(div['position']))
                            for div in api.dividends(ticker)
                        ))),
                    ),
                    lambda d: sum(
                        d['divs'][0] # dividend amount
                    ) / sum(
                        d['divs'][1] # number of stocks held at time of dividend payout
                    ) / (
                        d['price'] # recency-weighted average of share price
                    ) if len(d['divs']) > 0 else 0,
                )
            ),
            pullcast=F,
            pushcast=util.color.mpct,
            description='Dividend yield per dollar of share price',
            documentation='...',
        ),
        Field(
            name='pcps%',
            getter=FieldComplexConstructor(
                attributes=(
                    'ticker',
                ),
                chain=(
                    lambda R: R['ticker'],
                    lambda ticker: dict(
                        price=np.mean((
                            api.ma(ticker, ma='d200ma'),
                            api.ma(ticker, ma='d50ma'),
                            api.price(ticker),
                        )),
                        premcol=S[S['ticker']==ticker]['premium_collected'],
                        bought=T[T['symbol']==ticker]['bought'].tail(1).item(),
                    ),
                    lambda d: (
                        d['premcol'] # premium collected in total
                    ) / (
                        d['bought'] # number of shares ever purchased
                    ) / (
                        d['price'] # recency-weighted average of share price
                    )
                )
            ),
            pullcast=F,
            pushcast=util.color.mpct,
            description='Premium collected per dollar of share price ever purchased',
            documentation='...',
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
                    'y5cp', 'y2cp', 'y1cp', 'm6cp', 'm3cp', 'm1cp', 'd30cp', 'd5cp',
                ),
                chain=(
                    lambda R: [
                        sp.stats.percentileofscore(S[period], value)
                        for period, value in R.items()
                    ],
                    statistics.mean
                )
            ),
            pullcast=F,
            pushcast=util.color.pct,
            description='Momentum Percentile (compared to the rest of this Portfolio)',
            documentation='',
        ),
        Field(
            name='score%',
            getter=FieldComplexConstructor(
                attributes=(
                    'cbps%', 'dyps%', 'pcps%', 'malps%',  'momentum',
                ),
                chain=(
                    lambda R: [
                        sp.stats.percentileofscore(S[period], value)
                        for period, value in R.items()
                    ],
                    statistics.mean
                )
            ),
            pullcast=F,
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
            pullcast=F,
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
            pullcast=F,
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
    typecasters.update({
        field.name: field.pullcast
        for field in _extensions(None, None)
        if field.getter is None
    })
    return defaultdict(lambda: str, typecasters)

def _extend(S, field):
    figet = field.getter
    cast = field.pullcast
    try:
        if figet is None:
            S[field.name] = S[field.name].apply(cast)
        elif type(figet) is not FieldComplexConstructor:
            series = []
            for ticker in S['ticker']:
                datum = cast(figet(ticker))
                if datum in (None, NaN, NaT, 'N/A'):
                    datum = cast(figet(ticker, ignore_cache=True))
                series.append(datum)
            S[field.name] = series
        else:
            S[field.name] = pd.Series((
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
    def __init__(self, data, T, S=None):
        # 1. Create new DataFrame first.  This DataFrame will be limited to the list of tickers
        # supplied by the user, if any, otherwise as inclusive as the stored DataFrame.
        typecasts = _typecasters()
        df = pd.DataFrame(
            map(
                lambda row: map(
                    lambda typecast: typecast[1](row[typecast[0]]),
                    typecasts.items()
                ), data
            ),
            columns=typecasts.keys()
        )

        # 2. Update the stored DataFrame `df' with the fresh data in `S'
        if S is not None:
            S.set_index('ticker', inplace=True)
            S.update(df.set_index('ticker'))
            S.reset_index(inplace=True)
        else:
            S = df

        # 3. Replace all `NaN's with corresponding column means.
        means = S[S.keys()].mean(skipna=True, numeric_only=True)
        S.fillna(means, inplace=True)

        # 4. Extend fields (add new columns)
        for field in _extensions(T, S):
            _extend(S, field)

        # 5. Sort the DataFrame by Ticker
        S.sort_values(by='ticker', inplace=True)

        self._S = S

    @property
    def extended(self):
        return self._S
