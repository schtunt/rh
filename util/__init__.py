import math, decimal

dec  = lambda n: decimal.Decimal(str(n) if n is not None else 'NaN')
rnd  = lambda a, r: round(a / r) * r
sgn  = lambda n: -1 if n <= 0 else 1



import json
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return [str(o) for o in [o]]
        return super(DecimalEncoder, self).default(o)

def dump(heading, obj):
    return '%s\n%s' % (
        heading,
        highlight(
            json.dumps(obj, indent=4, cls=DecimalEncoder),
            PythonLexer(),
            Terminal256Formatter()
        ),
    )



from . import color
from . import datetime
