import rh
import pytest

import unittest
from datetime import datetime, timezone

import util
from util.numbers import dec as D
import tests.fixtures



@pytest.fixture(scope='module', autouse=True)
def module_mock_initialize(module_mocker):
    rh.preinitialize()
    module_mocker.patch.object(rh, 'iex', unittest.mock.MagicMock())
    module_mocker.patch.object(rh.Account, 'connect', lambda _: True)
    with open('/tmp/stocks.csv', 'w') as fH:
        fH.write(CSV)

def mkstonk(ticker):
    account = rh.Account()
    account.slurp()
    aapl = account.get_stock(ticker)
    return aapl


prices = [
    D('100'),
    D('110'),
    D('140'),
    D('155'),
    D('180'),
    D('210'),
    D('52.5'),
    D('60'),
    D('55'),
    D('61.6'),
]

index2month = lambda i: 1+i%12
index2year  = lambda i: 2020+(i//12)
index2date  = lambda i: datetime(index2year(i), index2month(i), 20, 0, 0, tzinfo=timezone.utc)
@pytest.fixture(scope='function', params=range(len(prices)))
def index(mocker, request):
    index = request.param
    mocker.patch.object(rh.util.datetime, 'now', lambda: index2date(index))
    return index

index2price = lambda i: prices[i]
@pytest.fixture(scope='function')
def price(mocker, index):
    tests.fixtures.MOCKED_APIS['stocks:prices'][0] = '%9.6f' % index2price(index)
    mocker.patch.object(rh.Account, 'cached', tests.fixtures.cached)
    return index2price(index)


CSV = '''\
symbol,date,order_type,side,fees,quantity,average_price
AAPL,2020-09-08T17:10:14.617000Z,market,buy,0.00,4.0000000000,55.00000000
# Month 8: NoEvent                                            60.00000000
# Month 7: <Event 7: StockSplit 4:1>                          52.50000000
AAPL,2020-06-06T19:58:29.368490Z,limit,sell,0.00,1.0000000000,210.00000000
AAPL,2020-05-05T16:16:08.700000Z,market,buy,0.00,100.00000000,180.00000000
AAPL,2020-04-04T14:41:49.878000Z,limit,sell,0.00,100.00000000,155.00000000
AAPL,2020-03-03T17:10:54.617000Z,limit,sell,0.00,100.00000000,140.00000000
AAPL,2020-02-02T17:53:17.760000Z,market,buy,0.00,100.00000000,110.00000000
AAPL,2020-01-01T19:58:29.368490Z,market,buy,0.00,100.00000000,100.00000000
'''

ANSWER_KEYS = [
    'cnt', 'ptr', 'trd', 'qty', 'epst', 'crsq', 'crsv', 'crlq', 'crlv', 'cusq', 'cusv', 'culq', 'culv'
]

@pytest.fixture(scope='function', params=ANSWER_KEYS)
def key(request):
    return request.param

def answers(index, price):
    p = price
    return {
        key: D([
            (1, 0,  100, 100,             0,   0,    0, 0, 0, 100,    0, 0, 0),
            (2, 0,  200, 200,             5,   0,    0, 0, 0, 200, 1000, 0, 0),
            (3, 0,  200, 100,            35, 100, 4000, 0, 0, 100, 3000, 0, 0),
            (4, 1,  200,   0,          42.5, 200, 8500, 0, 0,   0,    0, 0, 0),
            (5, 1,  300, 100,     85/D('3'), 200, 8500, 0, 0, 100,    0, 0, 0),
            (6, 4,  300,  99,    115/D('3'), 201, 8530, 0, 0,  99, 2970, 0, 0),
            (7, 4, 1200, 396,   115/D('12'), 804, 8530, 0, 0, 396, 2970, 0, 0),
            (8, 4, 1204, 400, 1447/D('120'), 804, 8530, 0, 0, 396, 5940, 0, 0),
        ][index][j]) for j, key in enumerate(ANSWER_KEYS)
    }

def test_stock(index, price, key):
    if index > 7: return

    aapl = mkstonk('AAPL')
    ledger = aapl._ledger[index]
    answer = answers(index, price)

    assert ledger[key] == answer[key]
