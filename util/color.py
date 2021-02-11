import locale
from termcolor import colored

from .numbers import dec as D

def colorize(f, n, d, v):
    c = d[0]
    if not n.is_nan():
        for i, v in enumerate(v):
            if v < n:
                c = d[i+1]

    return colored(f % n, c)

# Percentage
pct  = lambda p: colorize('%0.2f%%', D(p), ['red', 'yellow', 'green', 'magenta'], [D(0), D(50), D(80)])
mpct = lambda p: pct(100*p)

# Quantity
qty  = lambda q, dp=2: colorize(f'%0.{dp}f', D(q), ['yellow', 'cyan'], [D(0)])
qty0 = lambda q: qty(q, 0)
qty1 = lambda q: qty(q, 1)

# Currency
def mulla(m):
    m = D(m)
    c = 'red' if m < 0 else 'green' if m > 0 else 'yellow'
    return colored(locale.currency(D(m), grouping=True), c)
