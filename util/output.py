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


def ddump(data, title=None, force=False):
    def _dump(obj, title):
        class DecimalEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, decimal.Decimal):
                    return str(o)
                elif isinstance(o, datetime.datetime):
                    return str(o)
                return super(DecimalEncoder, self).default(o)

        return '%s\n%s' % (
            title, highlight(
                json.dumps(obj, indent=4, cls=DecimalEncoder),
                PythonLexer(),
                Terminal256Formatter()
            ),
        )

    if len(constants.DEBUG) == 0 and not force: return

    title = '%s(%d entries)' % (f'{title} ' if title else '', len(data))
    print(_dump(data, '[DEBUG:%s]-=[ %s ]=-' % (','.join(constants.DEBUG), title)))


class progress:
    def __init__(self, *args, force=True):
        self._force = force
        ddump(args, title='init', force=self._force)

    def __enter__(self, *args):
        self._started_at = time.time()

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
