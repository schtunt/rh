import numpy as np

flt  = np.single
rnd  = lambda a, r: round(a / r) * r
sgn  = lambda n: -1 if n <= 0 else 1

from . import datetime
from . import color
