import rh
import pytest

from unittest import mock
from datetime import datetime, timezone

import tests.fixtures

month2date = lambda m: datetime(2020, m, 1, 0, 0, tzinfo=timezone.utc)
month2price = lambda m: [100, 110, 140, 155, 180, 210, 213, 216, 220, 280, 300, 800][m-1]

CSV = '''\
symbol,date,order_type,side,fees,quantity,average_price
AAPL,2020-11-11T13:48:29.818000Z,limit,sell,0.00,3.0000000000,300.00000000
AAPL,2020-10-10T11:25:49.178000Z,limit,sell,0.00,396.00000000,280.00000000
AAPL,2020-09-09T17:10:54.617000Z,market,buy,0.00,4.0000000000,220.00000000
# stock-split 4x1 takes place here in August
AAPL,2020-06-06T19:58:29.368490Z,limit,sell,0.00,1.0000000000,210.00000000
AAPL,2020-05-05T16:16:08.700000Z,market,buy,0.00,100.00000000,180.00000000
AAPL,2020-04-04T14:41:49.878000Z,limit,sell,0.00,100.00000000,155.00000000
AAPL,2020-03-03T17:10:54.617000Z,limit,sell,0.00,100.00000000,140.00000000
AAPL,2020-02-02T17:53:17.760000Z,market,buy,0.00,100.00000000,110.00000000
AAPL,2020-01-01T19:58:29.368490Z,market,buy,0.00,100.00000000,100.00000000
'''

@pytest.fixture(scope='module', autouse=True)
def module_mock_initialize(module_mocker):
    rh.preinitialize()
    module_mocker.patch.object(rh, 'iex', mock.MagicMock())
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

def test_stock_at_event_1(month, price):
    aapl = mkstonk('AAPL')
    state = aapl._states[0]

    assert state['ptr'] == 0
    assert state['events'] == 1
    assert state['qty'] == 100

    assert state['cbr']['short']['qty'] == 0
    assert state['cbr']['short']['value'] == 0
    assert state['cbr']['long']['qty'] == 0
    assert state['cbr']['long']['value'] == 0

    assert state['cbu']['short']['qty'] == 100
    assert state['cbu']['short']['value'] == 100 * (price - 100)
    assert state['cbu']['long']['qty'] == 0
    assert state['cbu']['long']['value'] == 0

def test_stock_at_event_2(price):
    aapl = mkstonk('AAPL')
    state = aapl._states[1]

    assert state['ptr'] == 0
    assert state['events'] == 2
    assert state['qty'] == 200

    assert state['cbr']['short']['qty'] == 0
    assert state['cbr']['short']['value'] == 0
    assert state['cbr']['long']['qty'] == 0
    assert state['cbr']['long']['value'] == 0

    assert state['cbu']['short']['qty'] == 200
    assert state['cbu']['short']['value'] == 200 * (price - 105)
    assert state['cbu']['long']['qty'] == 0
    assert state['cbu']['long']['value'] == 0

def test_stock_at_event_3(price):
    aapl = mkstonk('AAPL')
    state = aapl._states[2]

    assert state['ptr'] == 0
    assert state['events'] == 3
    assert state['qty'] == 100

    assert state['cbr']['short']['qty'] == 100
    assert state['cbr']['short']['value'] == 100 * (140-100)
    assert state['cbr']['long']['qty'] == 0
    assert state['cbr']['long']['value'] == 0

    assert state['cbu']['short']['qty'] == 100
    assert state['cbu']['short']['value'] == 100 * (price - 110)
    assert state['cbu']['long']['qty'] == 0
    assert state['cbu']['long']['value'] == 0


def test_stock_at_event_4(price):
    aapl = mkstonk('AAPL')
    state = aapl._states[3]

    assert state['ptr'] == 1
    assert state['events'] == 4
    assert state['qty'] == 0

    assert state['cbr']['short']['value'] == 100 * (140-100) + 100 * (155-110)
    assert state['cbr']['short']['qty'] == 200
    assert state['cbr']['long']['qty'] == 0
    assert state['cbr']['long']['value'] == 0

    assert state['cbu']['short']['qty'] == 0
    assert state['cbu']['short']['value'] == 0
    assert state['cbu']['long']['qty'] == 0
    assert state['cbu']['long']['value'] == 0


#def test_stock_at_event_5(account):
#    aapl = account.get_stock('AAPL')
#
#    state = aapl._states[4]
#    assert state['ptr'] == 1
#    assert state['events'] == 5
#    assert state['qty'] == 100
#
#def test_stock_at_event_6(account):
#    aapl = account.get_stock('AAPL')
#
#    state = aapl._states[5]
#    assert state['ptr'] == 4
#    assert state['events'] == 6
#    assert state['qty'] == 99
#
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
