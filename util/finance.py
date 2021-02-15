import util.datetime

def conterm(fr, to=None):
    '''contract term: long-term for anything over a year, otherwise short-term'''
    delta = util.datetime.delta(fr, to)

    return 'short' if delta <= util.datetime.A_YEAR else 'long'
