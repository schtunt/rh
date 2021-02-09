import rh
import pytest

import unittest
from datetime import datetime, timezone

import tests.fixtures

month2date = lambda m: datetime(2020, m, 1, 0, 0, tzinfo=timezone.utc)
month2price = lambda m: [100, 110, 140, 155, 180, 210, 213, 216, 220, 280, 300, 800][m-1]

CSV = '''\
symbol,date,order_type,side,fees,quantity,average_price
AAPL,2020-12-13T13:48:29.838000Z,limit,sell,0.00,3.0000000000,99.00000000
AAPL,2020-11-11T13:12:29.618030Z,limit,sell,0.00,3.0000000000,99.00000000
AAPL,2020-10-10T11:25:49.178100Z,limit,sell,0.00,396.00000000,75.00000000
AAPL,2020-09-22T17:10:53.317000Z,market,buy,0.00,4.0000000000,55.00000000
AAPL,2020-08-09T17:10:14.617000Z,market,buy,0.00,4.0000000000,55.00000000
# <Event 7: StockSplit 4:1>
AAPL,2020-06-06T19:58:29.368490Z,limit,sell,0.00,1.0000000000,210.00000000
AAPL,2020-05-05T16:16:08.700000Z,market,buy,0.00,100.00000000,180.00000000
AAPL,2020-04-04T14:41:49.878000Z,limit,sell,0.00,100.00000000,155.00000000
AAPL,2020-03-03T17:10:54.617000Z,limit,sell,0.00,100.00000000,140.00000000
AAPL,2020-02-02T17:53:17.760000Z,market,buy,0.00,100.00000000,110.00000000
AAPL,2020-01-01T19:58:29.368490Z,market,buy,0.00,100.00000000,100.00000000
'''

ANSWER_TEMPLATE = {
    'ptr':0, 'events':0, 'qty':0,
    'crsq': 0, 'crsv': 0, 'crlq':0, 'crlv':0,
    'cusq': 0, 'cusv': 0, 'culq':0, 'culv':0,
}

def answers(month, price):
    answer = ANSWER_TEMPLATE.copy()

    answer.update([
        { 'qty': 100, 'events':1,  'cusq': 100, 'cusv': 100 * (price-100)
                                                                         },
        { 'qty': 200, 'events':2,  'cusq': 200, 'cusv': 200 * (price-105)
                                                                         },
        { 'qty': 100, 'events':3,  'cusq': 100, 'cusv': 100 * (price-110),
                                   'crsq': 100, 'crsv': 4000              },
        { 'qty':   0, 'events':4,
          'ptr':   1,              'crsq': 200, 'crsv': 8500              },
        { 'qty': 100, 'events':5,  'cusq': 100, 'cusv': 100 * (price-180),
          'ptr':   1,              'crsq': 200, 'crsv': 8500              },
        { 'qty':  99, 'events':6,  'cusq':  99, 'cusv':  99 * (price-180),
          'ptr':   4,              'crsq': 201, 'crsv': 8530              },
        { 'qty': 396, 'events':7,  'cusq':  99, 'cusv': 396 * (price-180),
          'ptr':   4,              'crsq': 201, 'crsv': 8530              },
        { 'qty': 400, 'events':8,  'cusq': 396, 'cusv': 396 * (price-180),
          'ptr':   4,              'crsq': 201, 'crsv': 8530              },
        { 'qty':   4, 'events':9,  'cusq': 396, 'cusv': 396 * (price-180),
          'ptr':   4,              'crsq': 201, 'crsv': 8530              },
        { 'qty': 396, 'events':10, 'cusq': 396, 'cusv': 396 * (price-180),
          'ptr':   4,              'crsq': 201, 'crsv': 8530              },
        { 'qty': 396, 'events':11, 'cusq': 396, 'cusv': 396 * (price-180),
          'ptr':   4,              'crsq': 201, 'crsv': 8530              },
        { 'qty': 396, 'events':12, 'cusq': 396, 'cusv': 396 * (price-180),
          'ptr':   7,              'crsq': 201, 'crsv': 8530              },
    ][month-1])

    return answer

#def test_stock_at_event_7(account):
#    aapl = account.get_stock('AAPL')
#
#    state = aapl._states[6]
#    assert state['ptr'] == 4
#    assert state['events'] == 7
#    assert state['qty'] == 396
#
#def test_stock_at_event_8(account):
#    aapl = account.get_stock('AAPL')
#
#    state = aapl._states[7]
#    assert state['ptr'] == 4
#    assert state['events'] == 8
#    assert state['qty'] == 400
#
#def test_stock_at_event_9(account):
#    aapl = account.get_stock('AAPL')
#
#    state = aapl._states[8]
#    assert state['ptr'] == 4
#    assert state['events'] == 9
#    assert state['qty'] == 4
#
#def test_stock_at_event_10(account):
#    aapl = account.get_stock('AAPL')
#
#    for i in range(len(aapl.events)):
#        e = aapl.events[i]
#        s = aapl._states[i]
#        print(e, s)
#
#    state = aapl._states[9]
#    assert state['ptr'] == 7
#    assert state['events'] == 10
#    assert state['qty'] == 1
#
#def test_stock_states():
#    account = rh.Account()
#    account.slurp()
#    aapl = account.get_stock('AAPL')
#
#    # All events from the CSV, plus 1 stock-split event
#    assert aapl.pointer == 7
#    assert len(aapl._states) == 9 + 1
#    assert aapl.quantity == 1

@pytest.fixture(scope='module', autouse=True)
def module_mock_initialize(module_mocker):
    rh.preinitialize()
    module_mocker.patch.object(rh, 'iex', unittest.mock.MagicMock())
    module_mocker.patch.object(rh.Account, 'connect', lambda _: True)
    with open('/tmp/stocks.csv', 'w') as fH:
        fH.write(CSV)

@pytest.fixture(scope='function', params=range(1, 13))
def month(mocker, request):
    month = request.param
    mocker.patch.object(rh.util.datetime, 'now', lambda: month2date(month))
    return month

@pytest.fixture(scope='function')
def price(mocker, month):
    tests.fixtures.MOCKED_APIS['stocks:prices'][0] = '%9.6f' % month2price(month)
    mocker.patch.object(rh.Account, 'cached', tests.fixtures.cached)
    return month2price(month)

def mkstonk(ticker):
    account = rh.Account()
    account.slurp()
    aapl = account.get_stock(ticker)
    return aapl

def test_stock(month, price):
    aapl = mkstonk('AAPL')
    state = aapl._states[month-1]
    answer = answers(month, price)

    assert state['ptr'] == answer['ptr']
    assert state['events'] == answer['events']
    assert state['qty'] == answer['qty']

    assert state['cbr']['short']['qty']   == answer['crsq']
    assert state['cbr']['short']['value'] == answer['crsv']
    assert state['cbr']['long']['qty']    == answer['crlq']
    assert state['cbr']['long']['value']  == answer['crlv']

    assert state['cbu']['short']['qty']   == answer['cusq']
    assert state['cbu']['short']['value'] == answer['cusv']
    assert state['cbu']['long']['qty']    == answer['culq']
    assert state['cbu']['long']['value']  == answer['culv']




