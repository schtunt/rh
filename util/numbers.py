import decimal
import numpy as np
import pandas as pd
import scipy as sp

def preinitialize():
    decimal.getcontext().prec = 20

NaN = float('nan')
def isnan(n):
    try:
        return np.isnan(np.float64(n))
    except:
        return True

flt = lambda n: F(n) if n is not None else NaN

#NaN = decimal.Decimal('NaN')
#Z = decimal.Decimal('0')
#dec = lambda n: decimal.Decimal(str(n)) if n is not None else NaN
#def D(*args):
#    if len(args) == 0: return Z
#    try: return flt(args[0])
#    except: return NaN

F = np.float64

std = lambda n: flt(np.std(n))
mean = lambda n: flt(np.mean(n))
ident = lambda n: n
rnd = lambda a, r: round(a / r) * r
sgn = lambda n: -1 if n <= 0 else 1

# Maps [-100/x .. +100/x] to [0 .. 100]
scale_and_shift = lambda p, x: (p*x+1)/2

# An increasing sequence will have a score of zero, and the further from that the series is,
# the worse the score will be.  Worst case is a reverse-sorted sequence.
growth_score = lambda series: np.sqrt(sum(map(
    lambda n: (flt(n[0]) - flt(n[1])) ** 2,
    zip(series, sorted(series))
)))
