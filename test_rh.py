import rh
import pytest

from unittest import mock
from datetime import datetime, timezone

rh.constants.CACHE_DIR='/tmp/'

month2date = lambda month: datetime(2020, month, 1, 0, 0, tzinfo=timezone.utc)

month2price = lambda month: [
    100, 110, 140, 155, 180, 210, 213, 216, 220, 280, 300, 800
][month-1]

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

MOCKED_APIS = {
    'export:stocks': None,
    'stocks:splits': [{
        'date': 1598832000000,
        'declaredDate': None,
        'description': 'Ordinary Shares',
        'exDate': '2020-08-31',
        'fromFactor': 1,
        'id': 'SPLITS',
        'key': 'AAPL',
        'ratio': 0.25,
        'refid': 6705964,
        'subkey': '6705964',
        'symbol': 'AAPL',
        'toFactor': 4,
        'updated': 1608134910000
    }],
    'account:holdings': {
        'AAPL': {
            'average_buy_price': '114.1640',
            'equity': '13739.74',
            'equity_change': '2277.674400',
            'id': '450dfc6d-5510-4d40-abfb-f633b7d9be3e',
            'name': 'Apple',
            'pe_ratio': '36.262900',
            'percent_change': '19.87',
            'percentage': '1.98',
            'price': '136.850000',
            'quantity': '100.40000000',
            'type': 'stock'
        }
    },
    'account:dividends': [{
        'account': 'https://api.robinhood.com/accounts/5RY82260/',
        'amount': '0.73',
        'drip_enabled': False,
        'id': 'f95c96d9-ee5e-432d-913c-25fa680d45e7',
        'instrument': 'https://api.robinhood.com/instruments/450dfc6d-5510-4d40-abfb-f633b7d9be3e/',
        'nra_withholding': '0',
        'paid_at': '2021-08-16T23:24:57.501321Z',
        'payable_date': '2021-08-16',
        'position': '1.0000',
        'rate': '0.7300000000',
        'record_date': '2021-08-13',
        'state': 'paid',
        'url': 'https://api.robinhood.com/dividends/f95c96d9-ee5e-432d-913c-25fa680d45e7/',
        'withholding': '0.00'
    }],
    'stocks:fundamentals': [{
        'average_volume': '133660187.900000',
        'average_volume_2_weeks': '133660187.900000',
        'ceo': 'Timothy Donald Cook',
        'description': 'Apple, Inc. engages in stuff',
        'dividend_yield': '0.601983',
        'float': '16771849119.200001',
        'headquarters_city': 'Cupertino',
        'headquarters_state': 'California',
        'high': '135.770000',
        'high_52_weeks': '145.090000',
        'industry': 'Telecommunications Equipment',
        'instrument': 'https://api.robinhood.com/instruments/450dfc6d-5510-4d40-abfb-f633b7d9be3e/',
        'low': '133.610000',
        'low_52_weeks': '53.152500',
        'market_cap': '2212054169331.267090',
        'market_date': '2021-02-03',
        'num_employees': 147000,
        'open': '135.520000',
        'pb_ratio': '34.076400',
        'pe_ratio': '36.262900',
        'sector': 'Electronic Technology',
        'shares_outstanding': '16515261828.664083',
        'symbol': 'AAPL',
        'volume': '89393460.000000',
        'year_founded': 1976,
    }],
    'stocks:stats': {
        'avg10Volume': 108492565,
        'avg30Volume': 108727355,
        'beta': 1.14690182912981,
        'companyName': 'Apple Inc',
        'day200MovingAvg': 119.64,
        'day30ChangePercent': 0.08186701958926057,
        'day50MovingAvg': 133.12,
        'day5ChangePercent': 0.0379233639767127,
        'dividendYield': 0.007378098907391369,
        'employees': 0,
        'exDividendDate': '2021-02-05',
        'float': 0,
        'marketcap': 2295940008960,
        'maxChangePercent': 51.40047511398904,
        'month1ChangePercent': 0.045449854565051906,
        'month3ChangePercent': 0.15265614082700285,
        'month6ChangePercent': 0.2488186172228486,
        'nextDividendDate': '',
        'nextEarningsDate': '2021-01-27',
        'peRatio': 35.913342858751754,
        'sharesOutstanding': 16788096000,
        'ttmDividendRate': 1.0090288065748436,
        'ttmEPS': 3.68,
        'week52change': 0.7225895271695797,
        'week52high': 142.95,
        'week52low': 55.66,
        'year1ChangePercent': 0.7190514797845529,
        'year2ChangePercent': 2.2201857328536176,
        'year5ChangePercent': 5.255546102405064,
        'ytdChangePercent': 0.03221326570660876
    },
    'stocks:instrument': 'AAPL',
    'stocks:prices': ['500.000000'],
    'stocks:marketcap': 0,
    'events:activities': [],
    'orders:stocks:open': [],
    'orders:options:all': [],
    'options:positions:all': [],
}

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
    MOCKED_APIS['stocks:prices'][0] = '%9.6f' % month2price(month)
    mocker.patch.object(
        rh.Account,
        'cached',
        lambda self, area, subarea, *args, **kwargs: MOCKED_APIS[
            '%s:%s' % (area, subarea)
        ]
    )
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
