import re
import time
import json
import decimal, datetime

from pprint import pformat

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter

from beautifultable import BeautifulTable

import constants

def dprintf(fmt, *args):
    print(fmt % args)


def ddump(data, force=False):
    # Need to pass in some debug flag
    if force is False:
        return

    def _dump(obj):
        class DecimalEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, decimal.Decimal):
                    return str(o)
                elif isinstance(o, datetime.datetime):
                    return str(o)
                return super(DecimalEncoder, self).default(o)

        return highlight(
            json.dumps(obj, indent=4, cls=DecimalEncoder),
            PythonLexer(),
            Terminal256Formatter()
        )

    print(_dump(data))


class progress:
    def __init__(self, data, title=None, force=True):
        self._data = data
        self._force = force
        self._title = '%s(%d entries)' % (title, len(data)) if title else ''

    def __enter__(self):
        self._started_at = time.time()
        print('[DEBUG]-=[ %s ]=-', self._title)
        ddump(self._data, force=self._force)

    def __exit__(self, exctype, excval, exctraceback):
        self._stopped_at = time.time()
        if any((exctype, excval, exctraceback)):
            print(exctype)
            print(excval)
            print(exctraceback)
            raise exctype(excval)

        delta = self._stopped_at - self._started_at
        if delta > 3:
            print("%d seconds" % delta)

def prtable(table):
    columns = table.columns.header
    table.columns.header = [h.replace('_', '\n') for h in columns]
    print(table)
    table.columns.header = columns

def mktable(
    VIEWS,
    data,
    view,
    formats,
    maxwidth=320,
    tickers=(),
    filter_by=None,
    sort_by=None,
    reverse=False,
    limit=None
):
    columns = VIEWS[view]['columns']

    # 0. create
    table = BeautifulTable(maxwidth=maxwidth)

    # 1. configure
    table.set_style(BeautifulTable.STYLE_GRID)
    table.columns.header = columns
    if 'activate' in columns:
        table.columns.alignment['activities'] = BeautifulTable.ALIGN_LEFT

    # 2. populate
    for i, datum in data:
        table.rows.append(map(lambda k: datum.get(k, 'N/A'), columns))

    # 3. filter
    if filter_by is not None:
        table = table.rows.filter(filter_by)
    if len(tickers) > 0:
        table = table.rows.filter(lambda row: row['ticker'] in tickers)

    # 4. sort
    if not sort_by:
        sort_by = VIEWS[view].get('sort_by', 'ticker')

    table.rows.sort(key=sort_by, reverse=reverse)

    # 5. limit
    if limit > 0:
        table = table.rows[:limit] if not reverse else table.rows[-limit:]

    # 6. format
    for index, column in enumerate(columns):
        rows, fn = table.columns[index], formats.get(column, None)
        if fn is not None:
            try:
                casted_rows = list(map(fn, rows))
                table.columns[index] = casted_rows
            except:
                print("Failed to cast values in column %s" % column, list(rows))
                raise

    return table


RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
def ansistrip(s):
    return RE_ANSI.sub('', s)
