import time
import json
import decimal, datetime

import pandas as pd

from pprint import pp

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter

from beautifultable import BeautifulTable

import constants
from constants import ZERO as Z

def dprintf(fmt, *args, force=False):
    # Need to pass in some debug flag
    if force is False:
        return

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
    df,
    columns,
    formats,
    maxwidth=320,
    tickers=(),
    filter_by=None,
    sort_by=['ticker'],
    reverse=False,
    limit=0
):
    # 1. filter columns
    df = df[columns]

    # 2. sort dataframe
    df = df.sort_values(by=list(sort_by))

    # 3. limit (or butterfly-limit (-limit)) dataframe
    l = len(df)
    if l > limit > 0:
        df = df.tal(limit) if not reverse else df.head(limit)
    elif limit < 0:
        # butterfly-limit (remove from center point out to the wings/extremes)
        halflimit = -limit
        if l > 2 * halflimit:
            df = pd.concat([
                df.head(halflimit),
                df.tail(halflimit),
            ], axis=0)

    # 4. create table
    table = BeautifulTable(maxwidth=maxwidth)

    # 5. configure table
    table.set_style(BeautifulTable.STYLE_GRID)
    table.columns.header = columns
    if 'activate' in columns:
        table.columns.alignment['activities'] = BeautifulTable.ALIGN_LEFT

    # 6. populate table
    for ticker, datum in df.iterrows():
        table.rows.append(datum)

    # 7. filter table rows
    if filter_by is not None:
        table = table.rows.filter(filter_by)
    if len(tickers) > 0:
        table = table.rows.filter(lambda row: row['ticker'] in tickers)

    # 8. format table cells
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
