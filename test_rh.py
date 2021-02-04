import rh
import pytest

rh.CACHE_DIR='/tmp'

def mock_cached(self, area, subarea, *args, **kwargs):
    if (area, subarea) == ('export', 'test'):
        pass
    elif (area, subarea) == ('stocks', 'splits'):
        return [{
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
        }]
    elif (area, subarea) == ('stocks', 'events'):
        return []
    elif (area, subarea) == ('account', 'holdings'):
        return {'AAPL': {
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
        }}
    elif (area, subarea) == ('stocks', 'prices'):
        return ['135.000000']
    elif (area, subarea) == ('account', 'dividends'):
        return [{
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
        }]
    elif (area, subarea) == ('stocks', 'fundamentals'):
        return [{
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
        }]

    elif (area, subarea) == ('orders', 'stocks:open'):
        return []
    elif (area, subarea) == ('orders', 'options:all'):
        return []
    elif (area, subarea) == ('options', 'positions:all'):
        return []
    else:
        print('ERROR', area, subarea, args, kwargs)

def test_transactions(mocker):
    rh.preinitialize()
    mocker.patch.object(rh.Account, 'cached', mock_cached)
    account = rh.Account(mocked=True)
    account.slurp()

    aapl = account.get_stock('AAPL')
    assert aapl.cost == 34000
    assert aapl.quantity == 334
    assert aapl.equity == 133600
    assert aapl.average == 100
    assert aapl.average == 100
    assert aapl.pointer == 10

