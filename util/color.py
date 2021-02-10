import locale
from termcolor import colored

from . import dec as D

def colorize(f, n, d, v):
    n = D(n)
    c = d[0]
    for i, v in enumerate(v):
        v = D(v)
        if not n.is_nan and v < n:
            c = d[i+1]

    return colored(f % n, c)

pct  = lambda p: colorize('%0.2f%%', p, ['red', 'green', 'magenta'], [0, 70])
mpct = lambda p: pct(100*p)
qty  = lambda q, dp=2: colorize(f'%0.{dp}f', q, ['yellow', 'cyan'], [0])
qty0 = lambda q: qty(q, 0)
qty1 = lambda q: qty(q, 1)

def mulla(m):
    c = 'red' if D(m) < 0 else 'green'
    return colored(locale.currency(D(m), grouping=True), c)
