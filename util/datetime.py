import pytz
import datetime
import dateutil.parser as dtp

import datetime as dt
from datetime import datetime, timedelta, timezone

now = lambda: pytz.UTC.localize(datetime.now())
parse = lambda datestr: pytz.UTC.localize(dtp.parse(datestr))
epoch2datetime = lambda ts: datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M')
