import os

import pandas_datareader as pdr
import pandas as pd

import datetime
from collections import defaultdict

import api
import constants
import util
import events
import fields
from progress.bar import ShadyBar

from util.numbers import D
from util.color import strip as S
from constants import ZERO as Z

FEATHER_BASE='/tmp'
def feathers():
    return [
        featherfile(base) for base in ('transactions', 'stocks')
    ]

def featherfile(base, ticker=None):
    featherbase = {
        'transactions': f'/{FEATHER_BASE}/transactions',
        'stocks': f'/{FEATHER_BASE}/stocks',
    }[base]

    if ticker is None:
        featherfile = f'{featherbase}.feather'
        return featherfile

    if not os.path.exists(featherbase):
        os.mkdir(featherbase)

    return f'{featherbase}/{ticker}.feather'


def update(column, data):
    '''
    data: dict() keyed by ticker
    '''
    feather = featherfile('stocks')
    if not os.path.exists(feather):
        return False

    df = pd.read_parquet(feather)
    dfcol = pd.DataFrame(data.items(), columns=('ticker', column))
    df[column] = dfcol[column].combine_first(df[column])
    df.reset_index(inplace=True)
    df.to_parquet(feather)
    return True


def transactions():
    '''
    DataFrame Update 1 - Stock information pulled from Robinhood CSV

    DataFrame Update 2 - Options information pulled from Robinhood
    '''
    feather = featherfile('transactions')
    if os.path.exists(feather):
        df = pd.read_parquet(feather)
        return df

    symbols = api.symbols()
    with ShadyBar('%32s' % 'Building Transactions', max=len(symbols)+4) as bar:
        # 1. Download Stock Transactions from Robinhood...
        imported = api.download('stocks')
        bar.next()

        # 2. Import Stock Transaction data to Pandas DataFrame
        header = pd.read_csv(imported, comment='#', nrows=0).columns
        df = pd.read_csv(imported, comment='#', converters=dict(
            date=util.datetime.dtp.parse,
            fees=D,
            quantity=D,
            average_price=D,
        ))
        bar.next()

        # -. Update Stocks Transaction history with position-altering Options Contracts...
        #    (stocks bought/sold via the options market (assigments and excercises)
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

        # 3. Sort
        df = df.sort_values(by='date')
        bar.next()

        # 4. Dump
        if 'test' not in feather:
            df.to_parquet(feather)
        bar.next()

    return df


def _create_and_link_python_stock_objects_and_slurp_ledger_to_dataframe(df, portfolio):
    '''
    Create local Python Stock objects, and links between them, and their corresponding
    Lots and LotConnectors

    DataFrame Update 3 - Ledger Data
    '''
    with ShadyBar('%32s' % 'Building Dependency Graph', max=len(df)) as bar:
        for i, dfrow in df.iterrows():
            # The DataFrame was created from transactions (export downloaded from Robinhood.
            # That means it will contain some symbols no longer in service, which we call
            # tockers.  It's necessary then, to attribute these old-symbol transactions to their
            # new ticker symbol.
            tocker = dfrow.symbol
            ticker = api.tocker2ticker(tocker)

            # If we no longer hold this symbol, we skip this iteration of the loop
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

def stock_historic_prices(ticker, years=7):
    '''
    Historic Stock Prices

    '''
    feather = featherfile('stocks', ticker)
    if os.path.exists(feather):
        df = pd.read_parquet(feather)
        return df

    tNow = util.datetime.now()
    today = util.datetime.short(tNow)

    # Start 7 years from today
    tm7y = tNow - 7 * util.datetime.timedelta(days=365)
    yesteryear = util.datetime.short(tm7y)

    df = pdr.data.DataReader(ticker, data_source='yahoo', start=tm7y, end=tNow)
    df.to_parquet(feather)

    return df

def stocks(transactions, portfolio, portfolio_is_complete):
    '''
    DataFrame Update 4 - Addtional columns added here from Stock Object calls or other APIs
    '''

    feather = featherfile('stocks')
    cache_exists = os.path.exists(feather)
    if not (portfolio_is_complete and cache_exists):
        # The `write' flag implies that the portfolio dataset is not partial (only contains
        # select tickers).  If a cache file does not exists already, that means this is the
        # time to create it.  For that to happen, the expensive linking process needs to take
        # place here.
        _create_and_link_python_stock_objects_and_slurp_ledger_to_dataframe(
            transactions,
            portfolio
        )
    elif cache_exists:
        df = pd.read_parquet(feather)
        return df

    now = util.datetime.now()

    with ShadyBar('%32s' % 'Building Stocks', max=len(portfolio)+9) as bar:
        # 1. Pull holdings
        holdings = api.holdings()
        bar.next()

        # 2. Pull Prices
        prices = { ticker: D(price) for ticker, price in api._price_agg().items() }
        bar.next()

        # 3. Pull Betas
        betas  = { ticker: D(beta) for ticker, beta in api._beta_agg().items() }
        bar.next()

        # 4. Pull Option Positions
        data = _option_positions(prices)
        collaterals = data['collaterals']
        next_expiries = data['next_expiries']
        urgencies = data['urgencies']
        opened = data['opened']
        del data
        bar.next()

        # 5. Pull Options Orders
        data = _option_orders()
        premiums = data['premiums']
        closed = data['closed']
        del data
        bar.next()

        # 6. Pull Stock Orders (blob)
        stock_orders = _stock_orders()
        bar.next()

        # 7. Pull Dividends data
        dividends = api.dividends()
        bar.next()

        # -. Add all rows to Python list first
        data = []
        for ticker, stock in portfolio.items():
            if ticker not in holdings:
                bar.next()
                continue

            holding = holdings[ticker]

            cbr = stock.costbasis(realized=True, when=now)
            crsq = cbr['short']['qty']
            crsv = cbr['short']['value']
            crlq = cbr['long']['qty']
            crlv = cbr['long']['value']

            cbu = stock.costbasis(realized=False, when=now)
            cusq = cbu['short']['qty']
            cusv = cbu['short']['value']
            culq = cbu['long']['qty']
            culv = cbu['long']['value']

            if ticker not in prices:
                prices[ticker] = api.price(ticker)

            if ticker not in betas:
                betas[ticker] = api.beta(ticker)

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
            dividend = dividends[ticker]
            next_expiry = next_expiries.get(ticker, pd.NaT)
            ttl = util.datetime.delta(now, next_expiry).days if next_expiry else -1
            fundamentals = api.fundamentals(ticker)
            quote = api.quote(ticker)
            row = dict(
                ticker=ticker,
                price=prices[ticker],
                pcp=D(quote['previousClose']),
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
                pe_ratio2=D(fundamentals['pe_ratio']),
                pb_ratio=D(fundamentals['pb_ratio']),
                collateral_call=D(collateral['call']),
                collateral_put=D(collateral['put']),
                ttl=ttl,
                urgency=urgencies[ticker],
                activities=activities,
                next_expiry=next_expiry,
            )
            data.append(row)
            bar.next()

        # 8. Fields Extensions
        flds = fields.Fields(data, transactions)
        df = flds.extended
        bar.next()

        # 9. Dump to file, conditionally, fully or partially
        if not cache_exists and portfolio_is_complete and 'test' not in feather:
            df.to_parquet(feather)
        elif not portfolio_is_complete and cache_exists:
            # Update the main Feather DataFrame (on-disk) with the partial data we just
            # retrieved.
            dfm = pd.read_parquet(feather)
            columns = list(dfm.columns)
            dfm.set_index(keys=columns, inplace=True)
            dfm.update(df.set_index(columns))
            dfm.reset_index(inplace=True)
            dfm.to_parquet(feather)
        bar.next()

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
