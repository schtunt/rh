import json
import decimal
from time import time
from functools import wraps
from collections import defaultdict

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter

from pprint import pp

from util.numbers import D

MEASURED = defaultdict(lambda: dict(count=0, time=0))
def measure(func):
    @wraps(func)
    def timed(*args, **kwargs):
        start = D(time())
        try:
            return func(*args, **kwargs)
        finally:
            MEASURED[func.__name__]['count'] += 1
            MEASURED[func.__name__]['time'] += D(time()) - start
    return timed

def mstart(name):
    assert 'activesince' not in MEASURED[name]
    MEASURED[name]['activesince'] = D(time())

def mstop(name):
    assert 'activesince' in MEASURED[name]
    MEASURED[name]['count'] += 1
    MEASURED[name]['time'] = D(time()) - MEASURED[name]['activesince']
    del MEASURED[name]['activesince']

def measurements():
    return _ddump(MEASURED)



def dprintf(fmt, *args, force=False):
    # Need to pass in some debug flag
    if force is False:
        return

    print(fmt % args)



def _ddump(obj):
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

def ddump(data, force=False):
    # Need to pass in some debug flag
    if force is False:
        return

    print(_ddump(data))
