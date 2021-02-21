from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter

from pprint import pp

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
