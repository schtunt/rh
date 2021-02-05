import rh
import pytest

rh.CACHE_DIR='/tmp'
MOCKED_APIS = {
    'export:test': None,
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
    'stocks:instrument': 'AAPL',
    'stocks:prices': ['135.000000'],
    'stocks:events': [],
    'orders:stocks:open': [],
    'orders:options:all': [],
    'options:positions:all': [],
}

def cached(self, area, subarea, *args, **kwargs):
    return MOCKED_APIS['%s:%s' % (area, subarea)]


@pytest.fixture(scope='session', autouse=True)
def initialize_session():
    rh.preinitialize()

@pytest.fixture(scope='session')
def account():
    account = rh.Account(mocked=True)
    account.slurp()
    return account

@pytest.fixture(scope='session', autouse=True)
def initialize_function(session_mocker):
    session_mocker.patch.object(rh.Account, 'cached', cached)

@pytest.fixture
def debug(account):
    aapl = account.get_stock('AAPL')
    for e in aapl.events:
        print(e)

def test_account_general(account):
    aapl = account.get_stock('AAPL')
    assert aapl.ticker == 'AAPL'

def test_stock_pointer(account):
    aapl = account.get_stock('AAPL')
    assert aapl.pointer == 4

def test_stock_quantities(account):
    aapl = account.get_stock('AAPL')

    # ( grep buy test.csv |cut -d, -f 6|paste -sd+ - | bc ) 2>/dev/null
    assert aapl.bought == 550

    # ( grep sell test.csv |cut -d, -f 6|paste -sd+ - | bc ) 2>/dev/nul
    assert aapl.sold == 300

    # ( tail -n +2 test.csv |cut -d, -f 6|paste -sd+ - | bc ) 2>/dev/null
    assert aapl.traded == 850

    assert aapl.quantity == 250

#def test_stock_cost(account):
#    aapl = account.get_stock('AAPL')
#    assert aapl.cost == 34000

#def test_stock_equity(account):
#    aapl = account.get_stock('AAPL')
#    assert aapl.equity == 133600
