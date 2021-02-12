import math, decimal
import numpy as np

from functools import reduce

dec = lambda n: decimal.Decimal(str(n) if n is not None else 'NaN')
rnd = lambda a, r: round(a / r) * r
sgn = lambda n: -1 if n <= 0 else 1
nan = decimal.Decimal('NaN')

std = lambda n: dec(np.std(n))
mean = lambda n: dec(np.mean(n))

# An increasing sequence will have a score of zero, and the further from that the series is,
# the worse the score will be.  Worst case is a reverse-sorted sequence.
growth_score = lambda series: dec(
    math.sqrt(
        sum(
            map(
                lambda n: dec(n[0] - n[1]) ** 2,
                zip(series, sorted(series))
            )
        )
    )
)
