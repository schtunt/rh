from collections import namedtuple

import api
import util
from util.numbers import D

Field = namedtuple(
    'DataField', [
        'getter',
        'pullcast',
        'pushcast',
        'description',
        'documentation',
    ]
)

apidictplucker = lambda getter, key: lambda ticker: getter(ticker)[key]
FIELDS = {
    'marketcap': Field(
        getter=api.market_cap,
        pullcast=D,
        pushcast=util.color.mulla,
        description='Market Capitalization',
        documentation='https://www.investopedia.com/terms/m/marketcapitalization.asp',
    ),
    'd50ma': Field(
        getter=apidictplucker(api.stats, 'day50MovingAvg'),
        pullcast=D,
        pushcast=util.color.qty,
        description='50-Day Moving Average',
        documentation='',
    ),
    'd200ma': Field(
        getter=apidictplucker(api.stats, 'day200MovingAvg'),
        pullcast=D,
        pushcast=util.color.qty,
        description='200-Day Moving Average',
        documentation='',
    ),
    'y5cp': Field(
        getter=apidictplucker(api.stats, 'year5ChangePercent'),
        pullcast=D,
        pushcast=D,
        description='5-Year Percentage Change',
        documentation='',
    ),
    'y2cp': Field(
        getter=apidictplucker(api.stats, 'year2ChangePercent'),
        pullcast=D,
        pushcast=D,
        description='2-Year Percentage Change',
        documentation='',
    ),
    'y1cp': Field(
        getter=apidictplucker(api.stats, 'year1ChangePercent'),
        pullcast=D,
        pushcast=D,
        description='1-Year Percentage Change',
        documentation='',
    ),
    'm6cp': Field(
        getter=apidictplucker(api.stats, 'month6ChangePercent'),
        pullcast=D,
        pushcast=D,
        description='6-Month Percentage Change',
        documentation='',
    ),
    'm3cp': Field(
        getter=apidictplucker(api.stats, 'month3ChangePercent'),
        pullcast=D,
        pushcast=D,
        description='3-Month Percentage Change',
        documentation='',
    ),
    'm1cp': Field(
        getter=apidictplucker(api.stats, 'month1ChangePercent'),
        pullcast=D,
        pushcast=D,
        description='1-Month Percentage Change',
        documentation='',
    ),
    'd30cp': Field(
        getter=apidictplucker(api.stats, 'day30ChangePercent'),
        pullcast=D,
        pushcast=D,
        description='30-Day Percentage Change',
        documentation='',
    ),
    'd5cp': Field(
        getter=apidictplucker(api.stats, 'day5ChangePercent'),
        pullcast=D,
        pushcast=D,
        description='5-Day Percentage Change',
        documentation='',
    ),
    'trd0': Field(
        getter=None,
        pullcast=None,
        pushcast=lambda d: util.color.qty0(util.datetime.age(d)),
        description='Day-Zero Trade',
        documentation='',
    ),
    'momentum': Field(
        getter=None,
        pullcast=None,
        pushcast=util.color.pct,
        description='Momentum Percentile (compared to the rest of this Portfolio)',
        documentation='',
    ),
}

row = lambda ticker: {header: field.getter(ticker) for header, field in FIELDS.items() if field.getter}
types = lambda: {header: field.pullcast for header, field in FIELDS.items() if field.pullcast}
formaters = lambda: {header: field.pushcast for header, field in FIELDS.items()}
