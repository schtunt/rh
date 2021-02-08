import locale
from termcolor import colored

from . import flt

pct  = lambda p: colorize('%0.2f%%', p, ['red', 'green', 'magenta'], [0, 70])
mpct = lambda p: pct(100*p)
qty  = lambda q, dp=2: colorize(f'%0.{dp}f', q, ['yellow', 'cyan'], [0])
qty0 = lambda q: qty(q, 0)
qty1 = lambda q: qty(q, 1)

def colorize(f, n, d, v):
    n = flt(n)
    c = d[0]
    for i, v in enumerate(v):
        if v < n:
            c = d[i+1]

    return colored(f % n, c)

def mulla(m):
    c = 'red' if flt(m) < 0 else 'green'
    return colored(locale.currency(flt(m), grouping=True), c)
