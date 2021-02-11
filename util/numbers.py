import math, decimal

from functools import reduce

dec  = lambda n: decimal.Decimal(str(n) if n is not None else 'NaN')
rnd  = lambda a, r: round(a / r) * r
sgn  = lambda n: -1 if n <= 0 else 1

growth_score = lambda prices: reduce(
    lambda a,b:a*b, map(
        (lambda n: n[0]/n[1]),
        map(sorted, zip(prices, sorted(prices)))
    )
)
