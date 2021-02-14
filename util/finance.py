import util.datetime
import datetime

def conterm(fr, to=None):
    '''contract term: long-term for anything over a year, otherwise short-term'''
    to = to if to is not None else util.datetime.now()
    delta = to - fr
    return 'short' if delta <= datetime.timedelta(days=365, seconds=0) else 'long'
