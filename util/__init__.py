import numpy as np

flt  = np.single
rnd  = lambda a, r: round(a / r) * r
sgn  = lambda n: -1 if n <= 0 else 1

import datetime

def epoch2datetime(ts) -> str:
    return datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M')
