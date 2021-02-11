import constants



import math, decimal

dec  = lambda n: decimal.Decimal(str(n) if n is not None else 'NaN')
rnd  = lambda a, r: round(a / r) * r
sgn  = lambda n: -1 if n <= 0 else 1



from . import color
from . import datetime
from . import output
