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

FIELDS = {
    'marketcap': Field(
        getter=api.market_cap,
        pullcast=D,
        pushcast=util.color.mulla,
        description='Market Capitalization',
        documentation='https://www.investopedia.com/terms/m/marketcapitalization.asp',
    )
}

row = lambda ticker: {header: field.getter(ticker) for header, field in FIELDS.items()}
types = lambda: {header: field.pullcast for header, field in FIELDS.items()}
formaters = lambda: {header: field.pushcast for header, field in FIELDS.items()}
