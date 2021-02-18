import os

from collections import defaultdict
from functools import reduce
import pandas as pd
import datetime

import api
import constants
import util
import events
from progress.bar import ShadyBar

from util.numbers import D
from util.color import strip as S
from constants import ZERO as Z


FEATHERS = {
    'transactions': '/tmp/transactions.feather',
    'stocks': '/tmp/stocks.feather',
}


def embelish(obj, attributes, column, chain):
    obj[column] = [
        reduce(
            lambda g, f: f(g),
            chain,
            components,
        ) for components in zip(*map(lambda attr: getattr(obj, attr), attributes))
    ]


def transactions():
    '''
    DataFrame Update 1 - Stock information pulled from Robinhood CSV

    DataFrame Update 2 - Options information pulled from Robinhood
    '''
    feather = FEATHERS['transactions']
    if os.path.exists(feather):
        print("0. Returned DataFrame from %s and return immediately" % feather)
        df = pd.read_parquet(feather)
        return df

    print("1. Download Stock Transactions from Robinhood...")
    imported = api.download('stocks')

    print("2. Import Stock Transaction data to Pandas DataFrame")
    header = pd.read_csv(imported, comment='#', nrows=0).columns
    df = pd.read_csv(imported, comment='#', converters=dict(
        date=util.datetime.dtp.parse,
        fees=D,
        quantity=D,
        average_price=D,
    ))

    print("3. Update Stocks Transaction history with position-altering Options Contracts...")
    # Update the dataframe with stocks bought and sold via the options market:
    symbols = api.symbols()
    with ShadyBar('Processing', max=len(symbols)) as bar:
        for ticker in symbols:
            for se in api.events(ticker):
                for ec in se['equity_components']:
                    df.append(
                        dict(
                            symbol=ticker,
                            date=se['updated_at'],
                            order_type=se['type'],
                            side=ec['side'],
                            fees=Z,
                            quantity=ec['quantity'],
                            average_price=ec['price'],
                        ), ignore_index=True
                    )
            bar.next()

    print("4. Sort DataFrame")
    df = df.sort_values(by='date')

    if 'test' not in feather:
        print("5. Dump DataFrame to Feather")
        df.to_parquet(feather)

    return df


def _create_and_link_python_stock_objects_and_slurp_ledger_to_dataframe(df, portfolio):
    '''
    Create local Python Stock objects, and links between them, and their corresponding
    Lots and LotConnectors

    DataFrame Update 3 - Ledger Data
    '''
    with ShadyBar('Linking', max=len(df)) as bar:
        for i, dfrow in df.iterrows():
            ticker = dfrow.symbol
            if ticker not in portfolio:
                bar.next()
                continue

            stock = portfolio[ticker]
            transaction = events.TransactionEvent(stock, dfrow)
            for key, val in stock.ledger(transaction).items():
                dfrow[key] = val

            bar.next()


def _stock_orders():
    orders = defaultdict(list)
    data = api.orders('stocks', 'open')
    for order in data:
        uri = order['instrument']
        ticker = api.ticker(uri)
        orders[ticker].append("%s %s x%s @%s" % (
            order['type'],
            order['side'],
            util.color.qty(order['quantity'], Z),
            util.color.mulla(order['price']),
        ))

    return stocks,


def stocks(transactions, portfolio, portfolio_is_complete):
    '''
    DataFrame Update 4 - Addtional columns added here from Stock Object calls or other APIs
    '''

    feather = FEATHERS['stocks']
    cache_exists = os.path.exists(feather)
    if not (portfolio_is_complete and cache_exists):
        # The `write' flag implies that the portfolio dataset is not partial (only contains
        # select tickers).  If a cache file does not exists already, that means this is the
        # time to create it.  For that to happen, the expensive linking process needs to take
        # place here.
        print("0. Create and link Python Stock objects, then slurp Stock ledgers to DataFrame...")
        _create_and_link_python_stock_objects_and_slurp_ledger_to_dataframe(
            transactions,
            portfolio
        )
    elif cache_exists:
        print("0. Returned DataFrame from %s and return immediately" % feather)
        df = pd.read_parquet(feather)
        return df

    now = util.datetime.now()

    print("1. Pull holdings")
    holdings = api.holdings()

    print("2. Pull Prices")
    prices = api.prices()

    print("3. Pull Option Positions (collaterals, next_expiry, and an `opened' blob)")
    data = _option_positions(prices)
    collaterals = data['collaterals']
    next_expiries = data['next_expiries']
    opened = data['opened']
    urgencies = data['urgencies']
    del data

    print("4. Pull Options Orders (premiums and a `closed' blob)")
    data = _option_orders()
    premiums = data['premiums']
    closed = data['closed']
    del data

    print("5. Pull Stock Orders (blob)")
    stock_orders = _stock_orders()

    print("6. Pull Dividends data")
    dividends = api.dividends()

    print("7. Pull Fundamentals")
    fundamentals = api.fundamentals()

    print("8. Per-Ticker Operations...")
    header = dict(
        ticker=str,
        price=D, # current stock price
        quantity=D,
        average_buy_price=D, # your average according to robinhood
        equity=D,
        percent_change=D, # total return
        equity_change=D, # total return
        type=str, # stock or adr
        name=str,
        pe_ratio=D,
        percentage=D,

        cnt=D,
        trd=D,
        qty=D,
        esp=D,

        crsq=D,
        crsv=D,
        crlq=D,
        crlv=D,
        cusq=D,
        cusv=D,
        culq=D,
        culv=D,

        premium_collected=D,
        dividends_collected=D,
        pe_ratio2=D,
        pb_ratio=D,
        collateral_call=D,
        collateral_put=D,
        next_expiry=datetime.datetime,
        ttl=D,
        marketcap=D,

        d50ma=D,
        d200ma=D,

        y5cp=D,
        y2cp=D,
        y1cp=D,
        m6cp=D,
        m3cp=D,
        m1cp=D,
        d30cp=D,
        d5cp=D,

        urgency=D,
        activities=str,
    )

    data = []
    with ShadyBar('Processing', max=len(portfolio)) as bar:
        for ticker, stock in portfolio.items():
            #print(f"8.1. [{ticker}] Update the dataframe")
            if ticker not in holdings:
                #print("8.1.1 Skipping stock no longer held - %s" % ticker)
                bar.next()
                continue

            holding = holdings[ticker]

            #print(f"8.2. [{ticker}] Cost Basis (realized)")
            cbr = stock.costbasis(realized=True, when=now)
            crsq = cbr['short']['qty']
            crsv = cbr['short']['value']
            crlq = cbr['long']['qty']
            crlv = cbr['long']['value']

            #print(f"8.3. [{ticker}] Cost Basis (unrealized)")
            cbu = stock.costbasis(realized=False, when=now)
            cusq = cbu['short']['qty']
            cusv = cbu['short']['value']
            culq = cbu['long']['qty']
            culv = cbu['long']['value']

            #print(f"8.4. [{ticker}] Fundamentals (if missing)")
            if ticker not in fundamentals:
                fundamentals[ticker] = api.fundamental(ticker)

            #print(f"8.5. [{ticker}] Prices (if missing)")
            if ticker not in prices:
                prices[ticker] = api.price(ticker)

            #print("8.6. [{ticker}] Activities blob")
            _opened = '\n'.join(opened.get(ticker, []))
            _closed = '\n'.join(closed.get(ticker, []))
            _collateral = D(collaterals[ticker]['call'])
            _optionable = (D(holding['quantity']) - _collateral) // 100
            activities_data = []
            if _opened: activities_data.append(_opened)
            if _closed: activities_data.append(_closed)
            if _optionable > 0: activities_data.append(
                '%sx100 available' % (util.color.qty0(_optionable))
            )
            activities = ('\n%s\n' % ('─' * 32)).join(activities_data)

            collateral = collaterals[ticker]
            premium = premiums[ticker]
            fundamental = fundamentals[ticker]
            dividend = dividends[ticker]
            next_expiry = next_expiries.get(ticker, None)
            ttl = util.datetime.delta(now, next_expiry).days if next_expiry else -1
            data.append(dict(
                ticker=ticker,
                price=prices[ticker],
                quantity=holding['quantity'],
                average_buy_price=holding['average_buy_price'],
                equity=holding['equity'],
                percent_change=holding['percent_change'],
                equity_change=holding['equity_change'],
                type=holding['type'],
                name=holding['name'],
                pe_ratio=holding['pe_ratio'],
                percentage=holding['percentage'],
                cnt=len(stock.events),
                trd=stock.traded(),
                qty=stock._quantity,
                esp=stock.esp(),
                crsq=crsq,
                crsv=crsv,
                crlq=crlq,
                crlv=crlv,
                cusq=cusq,
                cusv=cusv,
                culq=culq,
                culv=culv,
                premium_collected=premium,
                dividends_collected=sum(D(div['amount']) for div in dividend),
                pe_ratio2=D(fundamental['pe_ratio']),
                pb_ratio=D(fundamental['pb_ratio']),
                collateral_call=D(collateral['call']),
                collateral_put=D(collateral['put']),
                ttl=ttl,
                marketcap=stock.marketcap,
                d50ma=D(stock.stats['day50MovingAvg']),
                d200ma=D(stock.stats['day200MovingAvg']),
                y5cp=D(stock.stats['year5ChangePercent']),
                y2cp=D(stock.stats['year2ChangePercent']),
                y1cp=D(stock.stats['year1ChangePercent']),
                m6cp=D(stock.stats['month6ChangePercent']),
                m3cp=D(stock.stats['month3ChangePercent']),
                m1cp=D(stock.stats['month1ChangePercent']),
                d30cp=D(stock.stats['day30ChangePercent']),
                d5cp=D(stock.stats['day5ChangePercent']),
                urgency=urgencies[ticker],
                activities=activities,
                next_expiry=next_expiry,
            ))

            bar.next()

    print("9. Import Python data to Pandas DataFrame")
    df = pd.DataFrame(
        [[row[key] for key in header] for row in data],
        columns=header,
    )

    print("10. Sort the DataFrame")
    df = df.sort_values(by='ticker')

    if not cache_exists and portfolio_is_complete and 'test' not in feather:
        print("11. Dump to feather")
        df.to_parquet(feather)
    elif not portfolio_is_complete and cache_exists:
        # Update the main Feather DataFrame (on-disk) with the partial data we just
        # retrieved.
        print("11. Partial update and dump to feather")
        #dfm = pd.read_parquet(feather)
        #columns = list(dfm.columns)
        #dfm.set_index(keys=columns, inplace=True)
        #dfm.update(df.set_index(columns))
        #dfm.reset_index()
        #df.to_parquet(feather)

    return df


def _option_positions(prices):
    next_expiries = {}
    data = api.positions('options', 'all')
    collaterals = defaultdict(lambda: {'put': Z, 'call': Z})
    opened = defaultdict(list)
    urgencies = defaultdict(D)
    for option in [o for o in data if D(o['quantity']) != Z]:
        ticker = option['chain_symbol']
        price = prices[ticker]

        uri = option['option']
        instrument = api.instrument(uri)

        premium = Z
        if instrument['state'] != 'queued':
            premium -= D(option['quantity']) * D(option['average_price'])

        itype = instrument['type']
        otype = option['type']

        s_k_ratio = price / D(instrument['strike_price'])
        urgencies[ticker] = max((
            urgencies[ticker],
            util.color.wtm_urgency(s_k_ratio, otype, itype)
        ))

        opened[ticker].append("%s %s %s x%s P=%s K=%s X=%s %s" % (
            instrument['state'],
            util.color.otype(otype),
            util.color.itype(itype),
            util.color.qty(option['quantity'], Z),
            util.color.mulla(premium),
            util.color.mulla(instrument['strike_price']),
            instrument['expiration_date'],
            util.color.wtm(s_k_ratio, otype, itype)
        ))
        expiry = util.datetime.parse(instrument['expiration_date'])
        if next_expiries.get(ticker) is None:
            next_expiries[ticker] = expiry
        else:
            next_expiries[ticker] = min(next_expiries[ticker], expiry)

        collaterals[ticker][itype] += 100 * D(dict(
            put=instrument['strike_price'],
            call=option['quantity']
        )[itype])

    return dict(
        collaterals=collaterals,
        urgencies=urgencies,
        next_expiries=next_expiries,
        opened=opened,
    )

def _option_orders():
    closed = defaultdict(list)
    premiums = defaultdict(lambda: Z)
    data = api.orders('options', 'all')
    for option in [o for o in data if o['state'] not in ('cancelled')]:
        ticker = option['chain_symbol']

        util.output.ddump(option)

        strategies = []
        o, c = option['opening_strategy'], option['closing_strategy']
        if o:
            tokens = o.split('_')
            strategies.append('o[%s]' % ' '.join(tokens))
        if c:
            tokens = c.split('_')
            strategies.append('c[%s]' % ' '.join(tokens))

        legs = []
        premium = Z
        for leg in option['legs']:
            uri = leg['option']
            instrument = api.instrument(uri)

            util.output.ddump(instrument)

            legs.append('%s to %s K=%s X=%s' % (
                leg['side'],
                leg['position_effect'],
                util.color.mulla(instrument['strike_price']),
                instrument['expiration_date'],
            ))

            premium += sum([
                100 * D(x['price']) * D(x['quantity']) for x in leg['executions']
            ])

        premium *= -1 if option['direction'] == 'debit' else +1
        premiums[ticker] += premium

        closed[ticker].append("%s %s %s x%s P=%s K=%s" % (
            option['state'],
            option['type'],
            '/'.join(strategies),
            util.color.qty(option['quantity'], Z),
            util.color.mulla(premium),
            util.color.mulla(instrument['strike_price']),
        ))

        for l in legs:
            closed[ticker].append(' + l:%s' % l)

    return dict(
        closed=closed,
        premiums=premiums,
    )
