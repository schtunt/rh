import pytz
import datetime
import dateutil.parser as dtp

import pandas as pd
import datetime as dt
from datetime import datetime, timedelta, timezone

A_YEAR = timedelta(days=365, seconds=0)
A_MONTH = timedelta(days=30, seconds=0)

now = lambda: pytz.UTC.localize(dt.datetime.now())
parse = lambda datestr: pytz.UTC.localize(dtp.parse(datestr))
epoch2datetime = lambda ts: datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M')
recast = lambda dt: dt if not hasattr(dt, 'to_pydatetime') else dt.to_pydatetime()
short = lambda dt: pd.to_datetime(dt).strftime('%Y-%m-%d') if not pd.isna(dt) else pd.NaT
delta = lambda fr, to=None: (now() if to is None else recast(to)) - recast(fr)
age = lambda fr, to=now(): +delta(fr, to).days
ttl = lambda fr, to=now(): -delta(fr, to).days
lastyear = dt.datetime.now() - timedelta(days=365)

NaT = pd.NaT
def datetime(*args):
    if len(args) == 0:
        return dt.datetime.now()

    #tzinfo=util.datetime.timezone.utc
    if len(args) == 1:
        d = args[0]
        if type(d) is dt.datetime:
            return d
        elif type(d) is pd.Timestamp:
            return d
        elif type(d) is str:
            return parse(d)

    try:
        return datetime.datetime(*args)
    except:
        return NaT
