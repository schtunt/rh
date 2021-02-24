import time
import decimal, datetime

import pandas as pd

from . import debug

from beautifultable import BeautifulTable

import fields
from util.numbers import D, Z

class progress:
    def __init__(self, data, title=None, force=True):
        self._data = data
        self._force = force
        self._title = '%s(%d entries)' % (title, len(data)) if title else ''

    def __enter__(self):
        self._started_at = time.time()
        print('[DEBUG]-=[ %s ]=-', self._title)
        debug.ddump(self._data, force=self._force)

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
    maxwidth=380,
    tickers=(),
    filter_by=None,
    sort_by=['ticker'],
    reverse=False,
    limit=0
):
    # 1. sort dataframe
    df = df.sort_values(by=list(sort_by), ascending=not reverse)

    # 2. filter dataframe (by row)
    if len(tickers) > 0:
        df = df[df.apply(lambda row: row['ticker'] in tickers, axis=1)]

    if filter_by is not None:
        df = df[df.apply(filter_by, axis=1)]

    # 3. limit (or butterfly-limit (-limit)) dataframe
    l = len(df)
    if l > limit > 0:
        df = df.head(limit)
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
    for index, dfrow in df.iterrows():
        table.rows.append(dfrow[columns])

    # 7. format table cells
    formatters = fields.formatters()
    for index, column in enumerate(columns):
        # rows here represents all rows of a single column
        rows, fn = table.columns[index], formatters[column]
        if fn is None:
            continue

        try:
            casted_rows = list(map(fn, rows))
            table.columns[index] = casted_rows
        except:
            print("Failed to cast values in column %s" % column, list(rows))
            raise

    return table
