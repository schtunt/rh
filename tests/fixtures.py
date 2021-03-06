import rh
rh.constants.CACHE_DIR='/tmp/'

FIXTURES = {
    'iex': {
        'get_beta': '1.00000',
        'get_shares_outstanding': {'AAPL':1000000000000},
        'get_market_cap': 2295940008960,
        'get_splits': [{
            'date': 1598832000000,
            'declaredDate': None,
            'description': 'Ordinary Shares',
            'exDate': '2020-07-07',
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
        'get_key_stats': {
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
        'get_quote': {
            'AAPL': {
                'symbol': 'AAPL',
                'previousClose': 130.84,
                'changePercent': '0.12345',
            }
        }
    },
    'av': {
        'fd': {
            'ev': 0,
            'sector': 'N/A',
            'industry': 'N/A',
            'ebitda': 0,
        }
    },
    'rh': {
        'get_events': [],
        'build_user_profile': {
            'equity': 0,
            'extended_hours_equity': 0,
            'cash': 0,
            'dividend_total': 0,
        },
    },
    'rh.account': {
        'build_holdings': {
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
            },
        },
    },
    'rh.stocks': {
        'get_fundamentals': [{
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
    },
}
