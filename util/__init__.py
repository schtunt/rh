import math
import decimal

flt  = decimal.Decimal
rnd  = lambda a, r: round(a / r) * r
sgn  = lambda n: -1 if n <= 0 else 1
zero = flt(0)

from . import color
from . import datetime
