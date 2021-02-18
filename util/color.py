import re
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
    if not m.is_nan():
        c = 'red' if m < 0 else 'green' if m > 0 else 'yellow'
        return colored(locale.currency(m, grouping=True), c)
    else:
        return colored(m, 'red')


RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
def strip(s):
    return RE_ANSI.sub('', s)

'''
# The next 3 functions can be tested with this nested loop:

for i in [0.5, 0.85, 1, 1.15, 1.5]:
    for otype in 'short', 'long':
        for itype in 'call', 'put':
            print(
            '%3d%%'% (100 * i),
            util.color.otype(otype),
            util.color.itype(itype),
            util.color.wtm(i, otype, itype)
        )
'''

def itype(itype):
    return colored(itype, {
        'put': 'cyan',
        'call': 'yellow',
    }[itype])


def otype(itype):
    return colored(itype, {
        'short': 'red',
        'long': 'blue',
    }[itype])


def wtm(s_k_ratio, otype, itype):
    '''
    Where-The-Money?

    otype: short / long
    itype: call / put

    S: Current Price
    K: Strike Price

    S_K_Ratio: S/K
    '''

    keys = ['DITM', 'ITM', 'ATM', 'OTM', 'WOTM']
    thresholds = [ D(0.75), D(0.95), D(1.05), D(1.25) ]

    index = 0
    for i, limit in enumerate(thresholds):
        if s_k_ratio > limit:
            index = i+1
        else:
            break

    style = {
        'call': [
            ('red', ('reverse',)),
            ('red', ()),
            ('yellow', ('reverse',)),
            ('cyan', ()),
            ('cyan', ('reverse',)),
        ],
        'put': [
            ('red', ('reverse',)),
            ('red', ()),
            ('yellow', ('reverse',)),
            ('green', ()),
            ('green', ('reverse',)),
        ]
    }[itype]

    themoney = {
        'put': keys[index],
        'call': keys[len(thresholds) - index],
    }[itype]

    style = {
        'short': style,
        'long': reversed(style),
    }[otype]

    c, attrs = dict(zip(keys, style))[themoney]

    return colored(themoney, c, attrs=attrs)
