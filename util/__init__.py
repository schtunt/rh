import constants

chunk = lambda lst, n: (lst[i:i + n] for i in range(0, len(lst), n))

def singleton(cls):
    instances = {}
    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return _singleton

from . import color
from . import numbers
from . import datetime
from . import output
from . import finance
from . import debug
