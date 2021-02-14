import constants

chunk = lambda lst, n: (lst[i:i + n] for i in range(0, len(lst), n))

from . import color
from . import numbers
from . import datetime
from . import output
from . import finance
