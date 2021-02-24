import re
import locale

from termcolor import colored

from util.numbers import D

def colorize(f, n, d, v):
    c = d[0]
    if not n.is_nan():
        for i, v in enumerate(v):
            if v < n:
                c = d[i+1]

    return colored(f(n), c)

# Percentage
spc = ['red', 'yellow', 'green', 'cyan', 'magenta']
pct   = lambda p: colorize(lambda n: '%0.2f%%' % n, D(p), spc, [D(0), D(50), D(65), D(80)])
mpct  = lambda p: pct(100*p)

spcr  = ['blue', 'cyan', 'green', 'yellow', 'red']
pctr  = lambda p: colorize(lambda n: '%0.2f%%' % n, D(p), spcr, [D(0), D(50), D(65), D(80)])
mpctr = lambda p: pctr(100*p)

    # ${:,.2f}
# Quantity
qty  = lambda q, dp=2: colorize(
    lambda n: ('{:,.%df}' % dp).format(n),
    D(q),
    ['yellow', 'cyan'],
    [D(0)]
)
qty0 = lambda q: qty(q, 0)
qty1 = lambda q: qty(q, 1)


# Currency
def mulla(m):
    # locale.currency(m, grouping=True)
    m = D(m)
    if m.is_nan():
        return colored(m, 'red')

    if abs(m) < 1000:
        s = '{:,.2f}'.format(abs(m))
        c = 'red' if m < 0 else 'yellow'
    elif abs(m) < 1000000:
        s = '{:,}'.format(abs(round(m)))
        c = 'red' if m < 0 else 'green' if m >= 10000 else 'yellow'
    else:
        compressors = [
            ('K', 'yellow'),
            ('M', 'green'),
            ('B', 'blue'),
            ('T', 'magenta'),
        ]
        i = -1
        while abs(m) > 1000:
            m //= 1000
            i += 1
        s = '{:,.2f}'.format(abs(round(m)))
        if i > -1:
            s += compressors[i][0]
            c = compressors[i][1]

    s = ('+$%s' if m > 0 else '-$%s') % s
    return colored(s, c)


RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
def strip(s):
    return RE_ANSI.sub('', s)

def test():
    '''
    Columns: S/R Ratio, otype, itype, TM, Urgency
    '''

    for ot in 'short', 'long':
        for it in 'call', 'put':
            for i in [0.5, 0.65, 1, 1/0.65, 1/0.5]:
                print(
                    '%3d%%'% (100 * i),
                    otype(ot),
                    itype(it),
                    wtm(i, ot, it),
                    mpctr(wtm_urgency(i, ot, it)),
                )

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


def _wtm(s_k_ratio, otype, itype):
    '''
    Where-The-Money?

    otype: short / long
    itype: call / put

    S: Current Price
    K: Strike Price

    S_K_Ratio: S/K
    '''

    keys = ['DITM', 'ITM', 'ATM', 'OTM', 'WOTM']
    thresholds = [ D(0.75), D(0.95), D(1/0.95), D(1/0.75) ]

    index = 0
    for i, limit in enumerate(thresholds):
        if s_k_ratio >= limit:
            index = i+1
        else:
            break

    style = {
        'call': [
            ('red', ('reverse',)),
            ('red', ()),
            ('yellow', ('reverse', 'blink')),
            ('cyan', ()),
            ('cyan', ('reverse',)),
        ],
        'put': [
            ('red', ('reverse',)),
            ('red', ()),
            ('yellow', ('reverse', 'blink')),
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

    # how likely it is that this contract will not expire worthless, and that it will
    # be assigned if seller, and hold intrinsic value if buyer
    urgency = {
        'put': D(1.0) / D(s_k_ratio),
        'call': D(s_k_ratio),
    }[itype]
    urgency = {
        'short': urgency,
        'long': 1/urgency,
    }[otype]

    c, attrs = dict(zip(keys, style))[themoney]
    return dict(
        style_c=c,
        style_attrs=attrs,
        urgency=urgency,
        string=themoney,
    )

def wtm_urgency(s_k_ratio, otype, itype):
    cfg = _wtm(s_k_ratio, otype, itype)
    urgency = cfg['urgency']
    return urgency

def wtm(s_k_ratio, otype, itype):
    cfg = _wtm(s_k_ratio, otype, itype)
    themoney = cfg['string']
    urgency = cfg['urgency']
    c = cfg['style_c']
    attrs = cfg['style_attrs']
    return colored(themoney, c, attrs=attrs)
