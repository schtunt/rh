import rh
import pytest

import unittest
from datetime import datetime, timezone

import util
from util import dec as D
import tests.fixtures

prices = [
    D('100'),
    D('110'),
    D('140'),
    D('155'),
    D('180'),
    D('210'),
    D('60'),
    D('55'),
    D('61.6'),
]
month2price = lambda m: prices[m-1]
month2date = lambda m: datetime(2020, m, 20, 0, 0, tzinfo=timezone.utc)

CSV = '''\
symbol,date,order_type,side,fees,quantity,average_price
AAPL,2020-09-09T17:10:53.317000Z,market,buy,0.00,100.00000000,58.00000000
AAPL,2020-08-08T17:10:14.617000Z,market,buy,0.00,4.0000000000,55.00000000
# <Event 7: StockSplit 4:1>
AAPL,2020-06-06T19:58:29.368490Z,limit,sell,0.00,1.0000000000,210.00000000
AAPL,2020-05-05T16:16:08.700000Z,market,buy,0.00,100.00000000,180.00000000
AAPL,2020-04-04T14:41:49.878000Z,limit,sell,0.00,100.00000000,155.00000000
AAPL,2020-03-03T17:10:54.617000Z,limit,sell,0.00,100.00000000,140.00000000
AAPL,2020-02-02T17:53:17.760000Z,market,buy,0.00,100.00000000,110.00000000
AAPL,2020-01-01T19:58:29.368490Z,market,buy,0.00,100.00000000,100.00000000
'''

ANSWER_TEMPLATE = {
    'ptr':0, 'cnt':0, 'qty':0,
    'crsq': 0, 'crsv': 0, 'crlq':0, 'crlv':0,
    'cusq': 0, 'cusv': 0, 'culq':0, 'culv':0,
}

def answers(month, price):
    answer = ANSWER_TEMPLATE.copy()

    answer.update([
        { 'qty': 100, 'cnt':1,  'cusq': 100, 'cusv': 100 * (price-100),
                                                                       },
        { 'qty': 200, 'cnt':2,  'cusq': 200, 'cusv': 200 * (price-105),
                                                                       },
        { 'qty': 100, 'cnt':3,  'cusq': 100, 'cusv': 100 * (price-110),
                                'crsq': 100, 'crsv': 4000              },
        { 'qty':   0, 'cnt':4,
          'ptr':   1,           'crsq': 200, 'crsv': 8500              },
        { 'qty': 100, 'cnt':5,  'cusq': 100, 'cusv': 100 * (price-180),
          'ptr':   1,           'crsq': 200, 'crsv': 8500              },
        { 'qty':  99, 'cnt':6,  'cusq':  99, 'cusv':  99 * (price-180),
          'ptr':   4,           'crsq': 201, 'crsv': 8530              },
        { 'qty': 396, 'cnt':7,  'cusq': 396, 'cusv': 396 * (price-45),
          'ptr':   4,           'crsq': 201, 'crsv': 8530              },
        { 'qty': 400, 'cnt':8,  'cusq': 400, 'cusv': 400 * (price-D('45.1')),
          'ptr':   4,           'crsq': 201, 'crsv': 8530              },
        { 'qty': 500, 'cnt':9,  'cusq': 500, 'cusv': 500 * (price-D('47.68')),
          'ptr':   4,           'crsq': 201, 'crsv': 8530              },
    ][month-1])

    return answer

@pytest.fixture(scope='module', autouse=True)
def module_mock_initialize(module_mocker):
    rh.preinitialize()
    module_mocker.patch.object(rh, 'iex', unittest.mock.MagicMock())
    module_mocker.patch.object(rh.Account, 'connect', lambda _: True)
    with open('/tmp/stocks.csv', 'w') as fH:
        fH.write(CSV)

@pytest.fixture(scope='function', params=range(1, 1+len(prices)))
def month(mocker, request):
    month = request.param
    mocker.patch.object(rh.util.datetime, 'now', lambda: month2date(month))
    return month

@pytest.fixture(scope='function')
def price(mocker, month):
    tests.fixtures.MOCKED_APIS['stocks:prices'][0] = '%9.6f' % month2price(month)
    mocker.patch.object(rh.Account, 'cached', tests.fixtures.cached)
    return month2price(month)

@pytest.fixture(scope='function', params=ANSWER_TEMPLATE.keys())
def key(request):
    return request.param


def mkstonk(ticker):
    account = rh.Account()
    account.slurp()
    aapl = account.get_stock(ticker)
    return aapl

def test_stock(month, price, key):
    #if month != 7: return

    aapl = mkstonk('AAPL')
    ledger = aapl._ledger[month-1]
    answer = answers(month, price)

    assert ledger[key] == answer[key]
