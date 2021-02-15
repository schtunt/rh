import pytz
import datetime
import dateutil.parser as dtp

import datetime as dt
from datetime import datetime, timedelta, timezone

A_YEAR = timedelta(days=365, seconds=0)
A_MONTH = timedelta(days=30, seconds=0)

now = lambda: pytz.UTC.localize(datetime.now())
parse = lambda datestr: pytz.UTC.localize(dtp.parse(datestr))
epoch2datetime = lambda ts: datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M')
recast = lambda dt: dt if not hasattr(dt, 'to_pydatetime') else dt.to_pydatetime()
delta = lambda fr, to=None: (now() if to is None else recast(to)) - recast(fr)
