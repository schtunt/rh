import csv

from util.numbers import dec as D


class CSVReader:
    def __init__(self, cachefile):
        '''
        CSV file is expected to be in reverse time sort order (that's just
        what the robinhood API happens to return.  No further sorting is done
        to account for the case where the CSV is not sorted in exactly this
        manner!
        '''
        self.active = None

        self.cachefile = cachefile
        with open(self.cachefile, newline='') as fh:
            reader = csv.reader(fh, delimiter=',', quotechar='"')
            self.header = next(reader)
            self.reader = reversed([line for line in reader if not line[0].startswith('#')])

    def __iter__(self):
        return self

    def __next__(self):
        self.active = next(self.reader, None)
        if self.active: return self
        raise StopIteration

    def get(self, field):
        return self.active[self.header.index(field)]

    @property
    def timestamp(self):
        raise NotImplementedError

    @property
    def ticker(self):
        return self.get('symbol')

    @property
    def side(self):
        return self.get('side')

    @property
    def otype(self):
        raise NotImplementedError

    @property
    def parameters(self):
        raise NotImplementedError


class StockReader(CSVReader):
    def __init__(self, cachefile):
        super().__init__(cachefile)
        self._ticker_field = 'symbol'

    @property
    def timestamp(self):
        return self.get('date')

    @property
    def otype(self):
        return self.get('order_type')

    @property
    def parameters(self):
        return (
            D(self.get('quantity')),
            D(self.get('average_price')),
            self.side,
            self.timestamp,
            self.otype,
        )
