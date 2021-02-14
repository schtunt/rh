import pytz
import datetime
import dateutil.parser as dtp

import random

import datetime as dt
from datetime import datetime, timedelta, timezone

now = lambda: pytz.UTC.localize(datetime.now())
parse = lambda datestr: pytz.UTC.localize(dtp.parse(datestr))
epoch2datetime = lambda ts: datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M')

class jitter:
    MINUTE = 60
    HOUR   = 60 * MINUTE
    DAY    = 24 * HOUR
    WEEK   = 7  * DAY
    MONTH  = 4  * WEEK

    @classmethod
    def ish(cls, t):
        '''jitter t to something in the range of 0.9 x t to 1.4 x t'''
        return t * (0.9 + 0.5 *random.random())

    def minutish(n=1):
        return jitter.ish(n * jitter.MINUTE)

    def hourish(n=1):
        return jitter.ish(n * jitter.HOUR)

    def dayish(n=1):
        return jitter.ish(n * jitter.DAY)

    def weekish(n=1):
        return jitter.ish(n * jitter.WEEK)

    def monthish(n=1):
        return jitter.ish(n * jitter.MONTH)
