import json
import decimal
from time import time
import datetime
from functools import wraps
from collections import defaultdict

from pprint import pformat
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter

from util.numbers import F

from . import pandas

MEASURED = defaultdict(lambda: dict(count=0, time=0))
def measure(func):
    @wraps(func)
    def timed(*args, **kwargs):
        start = F(time())
        try:
            return func(*args, **kwargs)
        finally:
            MEASURED[func.__name__]['count'] += 1
            MEASURED[func.__name__]['time'] += F(time()) - start
    return timed

def mstart(name):
    assert 'activesince' not in MEASURED[name]
    MEASURED[name]['activesince'] = F(time())

def mstop(name):
    assert 'activesince' in MEASURED[name]
    MEASURED[name]['count'] += 1
    MEASURED[name]['time'] = F(time()) - MEASURED[name]['activesince']
    del MEASURED[name]['activesince']

def measurements():
    return _ddump(MEASURED)



def dprintf(fmt, *args, force=False):
    # Need to pass in some debug flag
    if force is False:
        return

    print(fmt % args)


def _ddump(obj, json=False):
    if not json:
        return highlight(
            pformat(obj),
            PythonLexer(),
            Terminal256Formatter(),
        )

    class DecimalEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, decimal.Decimal):
                return str(o)
            elif isinstance(o, datetime.datetime):
                return str(o)
            return super(DecimalEncoder, self).default(o)
    try:
        return highlight(
            json.dumps(obj, indent=4, cls=DecimalEncoder),
            PythonLexer(),
            Terminal256Formatter()
        )
    except TypeError:
        return highlight(
            pformat(obj),
            PythonLexer(),
            Terminal256Formatter(),
        )


def ddump(data, force=False):
    # Need to pass in some debug flag
    if force is False:
        return

    print(_ddump(data))
