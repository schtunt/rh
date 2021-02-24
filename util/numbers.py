import decimal
import numpy as np
import pandas as pd
import scipy as sp

def preinitialize():
    decimal.getcontext().prec = 20

Z = decimal.Decimal('0')

NaN = decimal.Decimal('NaN')
dec = lambda n: decimal.Decimal(str(n)) if n is not None else NaN
def D(*args):
    if len(args) == 0: return Z
    try: return dec(args[0])
    except: return NaN

std = lambda n: dec(np.std(n))
mean = lambda n: dec(np.mean(n))
ident = lambda n: n
rnd = lambda a, r: round(a / r) * r
sgn = lambda n: -1 if n <= 0 else 1

# Maps [-100/x .. +100/x] to [0 .. 100]
scale_and_shift = lambda p, x: (p*x+1)/2

# An increasing sequence will have a score of zero, and the further from that the series is,
# the worse the score will be.  Worst case is a reverse-sorted sequence.
growth_score = lambda series: sum(
    map(
        lambda n: (dec(n[0]) - dec(n[1])) ** 2,
        zip(series, sorted(series))
    )
).sqrt()
