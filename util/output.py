import re
import time
import json
import decimal, datetime

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter

import constants

def dprintf(fmt, *args):
    print(fmt % args)


def ddump(data, force=False):
    if len(constants.DEBUG) == 0 and not force: return

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
        self._title = '%s(%d entries)' % (f'{title} ' if title else '', len(data))

    def __enter__(self):
        self._started_at = time.time()
        print('[DEBUG:%s]-=[ %s ]=-' % (','.join(constants.DEBUG), self._title))
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

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
def ansistrip(s):
    return RE_ANSI.sub('', s)
