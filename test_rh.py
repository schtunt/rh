# Testing, no interwebz allowed
import socket
def suckit(*args, **kwargs):
    raise Exception("Nice try, but no Interwebz for you!")
socket.socket = suckit

import unittest
import unittest.mock

# Decorators must be mocked prior to they being imported anywhere
import cachier
dontcachier = lambda *args, **kwargs: lambda fn: fn
unittest.mock.patch(target='cachier.cachier', new=dontcachier).start()

import pytest

# Follow-ups for mocking in Python, if things get more complicated
# + https://alexmarandon.com/articles/python_mock_gotchas/
# + https://docs.pylonsproject.org/projects/venusian/en/latest/

import os
from datetime import datetime, timezone

import rh
import api
import util
import slurp
import models
import fields
import account
from util.numbers import D

import tests.fixtures


@pytest.fixture(scope='module', autouse=True)
def module_mock_initialize(module_mocker):
    cachefile = '/tmp/stocks.csv'
    with open(cachefile, 'w') as fH:
        fH.write(CSV)

    slurp.FEATHER_BASE = '/tmp/tests'
    if not os.path.exists(slurp.FEATHER_BASE):
        os.mkdir(slurp.FEATHER_BASE)

    module_mocker.patch.object(models, 'treynor', unittest.mock.MagicMock(**dict(
        return_value=D(1)
    )))

    module_mocker.patch.object(models, 'sharpe', unittest.mock.MagicMock(**dict(
        return_value=D(1)
    )))

    module_mocker.patch.object(api, 'download', unittest.mock.MagicMock(
        return_value=cachefile
    ))

    module_mocker.patch.object(api, 'symbols', unittest.mock.MagicMock(
        return_value=['AAPL']
    ))

    module_mocker.patch.object(api, 'connect', unittest.mock.MagicMock())

    module_mocker.patch.object(api, 'FundamentalData', unittest.mock.MagicMock(**{
        'return_value.%s.return_value' % fn: response
            for fn, response in tests.fixtures.FIXTURES['av']['fd'].items()
    }))

    module_mocker.patch.object(api, 'rh', unittest.mock.MagicMock(**{
        '%s.return_value' % fn: response
            for fn, response in tests.fixtures.FIXTURES['rh'].items()
    }))

    module_mocker.patch.object(api.rh, 'stocks', unittest.mock.MagicMock(**{
        '%s.return_value' % fn: response
            for fn, response in tests.fixtures.FIXTURES['rh.stocks'].items()
    }))

    module_mocker.patch.object(api.rh, 'account', unittest.mock.MagicMock(**{
        '%s.return_value' % fn: response
            for fn, response in tests.fixtures.FIXTURES['rh.account'].items()
    }))

    module_mocker.patch.object(api, 'iex', unittest.mock.MagicMock(**{
        'Stock.return_value.%s.return_value' % fn: response
            for fn, response in tests.fixtures.FIXTURES['iex'].items()
    }))

    rh.preinitialize()


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

index2month = lambda i: 1 + i % 12
index2year = lambda i: 2020 + (i // 12)
index2date = lambda i: datetime(index2year(i), index2month(i), 20, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope='function', params=range(len(prices)))
def index(mocker, request):
    index = request.param

    return index


index2price = lambda i: prices[i]


@pytest.fixture(scope='function')
def price(mocker, index):
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
    'cnt', 'ptr', 'trd', 'qty', 'cbps',
    'crsq', 'crsv', 'crlq', 'crlv', 'cusq', 'cusv', 'culq', 'culv'
]


@pytest.fixture(scope='function', params=ANSWER_KEYS)
def key(request):
    return request.param


def answers(index):
    return {
        key: D([
            (1, 0, 100,  100,               0,   0,    0, 0, 0, 100,    0, 0, 0),
            (2, 0, 200,  200,               5,   0,    0, 0, 0, 200, 1000, 0, 0),
            (3, 0, 200,  100,              35, 100, 4000, 0, 0, 100, 3000, 0, 0),
            (4, 1, 200,    0,            42.5, 200, 8500, 0, 0,   0,    0, 0, 0),
            (5, 1, 300,  100,     85 / D('3'), 200, 8500, 0, 0, 100,    0, 0, 0),
            (6, 4, 300,   99,    115 / D('3'), 201, 8530, 0, 0,  99, 2970, 0, 0),
            (7, 4, 1200, 396,   115 / D('12'), 804, 8530, 0, 0, 396, 2970, 0, 0),
            (8, 4, 1204, 400, 1447 / D('120'), 804, 8530, 0, 0, 396, 5940, 0, 0),
        ][index][j]) for j, key in enumerate(ANSWER_KEYS)
    }

@unittest.mock.patch('util.datetime.now')
@unittest.mock.patch('api.rh.stocks.get_latest_price')
@unittest.mock.patch('api.price')
@unittest.mock.patch('api._price_agg')
def test_stock(
    iex_price_agg, iex_price, rh_get_latest_price, now,
    index, price, key
):
    iex_price.return_value = price
    iex_price_agg.return_value = {'AAPL': price}
    rh_get_latest_price.return_value = [price]

    _PULLCAST = fields._PULLCAST.items()
    _PUSHCAST = fields._PUSHCAST.items()

    now.return_value = index2date(index)
    assert util.datetime.now() == index2date(index)

    if index > 7: return
    acc = account.Account(tickers=['AAPL'])

    stock = acc.get_stock('AAPL')
    assert len(api.splits('AAPL')) == 1
    assert len(stock.splits) == 1

    ledger = stock._ledger[index]
    answer = answers(index)
    assert ledger[key] == answer[key]
