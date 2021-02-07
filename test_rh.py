import rh
import pytest

rh.CACHE_DIR='/tmp'
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
    'stocks:prices': ['135.000000'],
    'stocks:marketcap': 0,
    'events:activities': [],
    'orders:stocks:open': [],
    'orders:options:all': [],
    'options:positions:all': [],
}

def cached(self, area, subarea, *args, **kwargs):
    return MOCKED_APIS['%s:%s' % (area, subarea)]


@pytest.fixture(scope='session', autouse=True)
def initialize_session():
    rh.preinitialize()

@pytest.fixture(scope='session', autouse=True)
def initialize_function(session_mocker):
    session_mocker.patch.object(rh.Account, 'cached', cached)

@pytest.fixture(scope='session')
def account():
    account = rh.Account()
    account.slurp()
    return account

@pytest.fixture
def debug(account):
    aapl = account.get_stock('AAPL')
    for e in aapl.events:
        print(e)

@pytest.fixture(scope='function')
def trades():
    '''This must be 1:1 with test.csv'''
    return [
        (150,80), (100,90), (25,120), (25,130), (-150,150),
        (14,140), (12,145), (6,150) ,(-150,220),
        (202,200), (6,240), (10,350)
    ]

def test_account_general(account, trades):
    aapl = account.get_stock('AAPL')
    assert aapl.ticker == 'AAPL'

def test_stock_pointer(account, trades):
    aapl = account.get_stock('AAPL')
    assert aapl.pointer == 4

def test_stock_quantities_bought(account, trades):
    aapl = account.get_stock('AAPL')

    # sum(filter(lambda t: t>0, map(lambda t: t[0], trades)))
    # ( grep buy test.csv |cut -d, -f 6|paste -sd+ - | bc ) 2>/dev/null
    assert aapl.bought == 550

def test_stock_quantities_sold(account, trades):
    aapl = account.get_stock('AAPL')

    # sum(filter(lambda t: t>0, map(lambda t: -t[0], trades)))
    # ( grep sell test.csv |cut -d, -f 6|paste -sd+ - | bc ) 2>/dev/nul
    assert aapl.sold == 300

def test_stock_quantities_traded(account, trades):
    aapl = account.get_stock('AAPL')

    # sum(map(lambda t: abs(t[0]), trades))
    # ( tail -n +2 test.csv |cut -d, -f 6|paste -sd+ - | bc ) 2>/dev/null
    assert aapl.traded == 850

def test_stock_quantities_held(account, trades):
    aapl = account.get_stock('AAPL')

    # sum(map(lambda t: t[0], trades))
    assert aapl.quantity == 250

#def test_stock_cost(account):
#    aapl = account.get_stock('AAPL')
#    assert aapl.cost == 34000

#def test_stock_equity(account):
#    aapl = account.get_stock('AAPL')
#    assert aapl.equity == 133600
