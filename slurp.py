import os
import datetime

import pandas_datareader as pdr
import pandas as pd

from collections import defaultdict

from progress.bar import ShadyBar

import constants
import api
import util
import events
import fields

from util.numbers import D, Z

FEATHER_BASE='/tmp'
def feathers():
    return [
        featherfile(base) for base in ('transactions', 'stocks')
    ]

def featherfile(base, ticker=None):
    featherbase = {
        'transactions': f'/{FEATHER_BASE}/transactions',
        'stocks': f'{FEATHER_BASE}/stocks',
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


def transactions():
    '''
    DataFrame Update 1 - Stock information pulled from Robinhood CSV

    DataFrame Update 2 - Options information pulled from Robinhood
    '''

    symbols = api.symbols()
    with ShadyBar('%48s' % 'Building Transactions', max=len(symbols)+6) as bar:
        # 1. Download Stock Transactions from Robinhood...
        feather = featherfile('transactions')
        imported = '/tmp/.cached/stocks.csv'
        if not os.path.exists(imported):
            # The `ignore_cache' flag here is eaten up by the cachier decorator
            imported = api.download('stocks', ignore_cache=True)
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

        # 3. Update Stocks Transaction history with position-altering Options Contracts...
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

        # 4. Sort
        df.sort_values(by='date', inplace=True)
        bar.next()

        # 5. Reset the Index
        df.reset_index(inplace=True, drop=True)
        bar.next()

        cbtally = defaultdict(list)
        def _costbasis(row):
            data = cbtally[row.symbol]

            signum = {'buy':-1, 'sell':+1}[row.side]
            datum = ['trade', D(signum) * D(row.quantity), D(row.average_price), row.date]

            if len(data) > 0:
                _,_,_,d0 = data[-1]
                _,_,_,d2 = datum
                splits = api.splits(row.symbol)
                for split in splits:
                    d1 = util.datetime.parse(split['exDate'])
                    if not d0 < d1 < d2:
                        continue

                    for existing in data:
                        existing[1] *= D(split['toFactor'])
                        existing[1] /= D(split['fromFactor'])
                        existing[2] *= D(split['fromFactor'])
                        existing[2] /= D(split['toFactor'])

            data.append(datum)
            _,_,price,_ = datum

            bought = -sum(qty for cls,qty,pps,date in data if qty < 0)
            sold   = +sum(qty for cls,qty,pps,date in data if qty > 0)
            held   = bought - sold

            return pd.Series([
                D(bought), D(sold), D(held), D(
                    (
                        sum(qty * pps for cls,qty,pps,date in data) + held * price
                    )/bought
                ) if bought > 0 else Z
            ])

        # 6. Cost Basis
        df[['bought', 'sold', 'held', 'cbps']] = df.apply(_costbasis, axis=1)
        bar.next()

        # 7. Dump
        if 'test' not in feather:
            df.to_parquet(feather, index=False)
        bar.next()

    return df


def _create_and_link_python_stock_objects_and_slurp_ledger_to_dataframe(df, portfolio):
    '''
    Create local Python Stock objects, and links between them, and their corresponding
    Lots and LotConnectors

    DataFrame Update 3 - Ledger Data
    '''
    with ShadyBar('%48s' % 'Reading Robinhood Transactions History', max=len(df)) as bar:
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

def _pull_processed_holdings_data(portfolio, T):
    now = util.datetime.now()
    data = []

    with ShadyBar('%48s' % 'Refreshing Robinhood Portfolio Data', max=len(portfolio)+5) as bar:
        # 1. Pull holdings
        holdings = api.holdings()
        bar.next()

        # 2. Pull Option Positions
        _data = _option_positions(api._price_agg())
        collaterals = _data['collaterals']
        next_expiries = _data['next_expiries']
        urgencies = _data['urgencies']
        opened = _data['opened']
        del _data
        bar.next()

        # 3. Pull Options Orders
        _data = _option_orders()
        premiums = _data['premiums']
        closed = _data['closed']
        del _data
        bar.next()

        # 4. Pull Stock Orders (blob)
        stock_orders = _stock_orders()
        bar.next()

        # 5. Pull Dividends data
        dividends = api.dividends()
        bar.next()

        # 6. Add all rows to Python list first
        _timers = {}
        for ticker, stock in portfolio.items():
            bar.next()

            if ticker not in holdings:
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
                price=api.price(ticker),
                pcp=D(quote['previousClose']),
                beta=api.beta(ticker),
                quantity=holding['quantity'],
                average_buy_price=holding['average_buy_price'],
                equity=holding['equity'],
                percent_change=holding['percent_change'],
                equity_change=holding['equity_change'],
                type=holding['type'],
                name=holding['name'],
                percentage=holding['percentage'],
                cnt=len(stock.events),
                trd=stock.traded(),
                qty=stock._quantity,
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
                collateral_call=D(collateral['call']),
                collateral_put=D(collateral['put']),
                ttl=ttl,
                urgency=urgencies[ticker],
                activities=activities,
                next_expiry=next_expiry,
            )
            data.append(row)

    return data


def stocks(transactions, portfolio, portfolio_is_complete):
    '''
    DataFrame Update 4 - Addtional columns added here from Stock Object calls or other APIs
    '''

    feather = featherfile('stocks')
    cache_exists = os.path.exists(feather)

    if not (portfolio_is_complete and cache_exists):
        _create_and_link_python_stock_objects_and_slurp_ledger_to_dataframe(
            transactions,
            portfolio
        )

    df = None
    if cache_exists:
        df = pd.read_parquet(feather)
        if portfolio_is_complete:
            return df

    data = _pull_processed_holdings_data(portfolio, transactions)

    # 8. Field Extensions
    with ShadyBar('%48s' % 'Refreshing Extended DataFrame and Cache', max=2) as bar:
        util.debug.mstart('FieldExtensions')
        # NOTE: This will modify df (if it's not None) in-place.  If that ever changes, the
        # following commented-out logic must be used:
        #    df.set_index('ticker', inplace=True)
        #    df.update(flds.extended.set_index('ticker'))
        #    df.reset_index(inplace=True)
        flds = fields.Fields(data, transactions, df=df)
        if df is None:
            df = flds.extended

        util.debug.mstop('FieldExtensions')
        bar.next()

        if 'test' not in feather:
            # Emergency Sanitization
            #for column in df.columns:
            #    df[column] = df[column].map(lambda n: D(n) if type(n) in (str, float, int) else n)
            try:
                df.to_parquet(feather, index=False)
            except:
                print("Amended DataFrame can't be dumped")
                util.debug.ddump({
                    column:str(set(map(type, df[column]))) for column in df.columns
                }, force=True)
                raise
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
        price = D(prices[ticker])

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
