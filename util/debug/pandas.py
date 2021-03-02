import pandas as pd

seriesSideBySide = lambda s1, s2: pd.merge(s1, s2, right_index=True, left_index=True)

datatypes = lambda df: {
    column: set(map(type, df[column])) for column in df.columns
}

nullrows = lambda df: df[df.isnull().any(axis=1)]

def shownans(df, index='ticker'):
    '''Returns a table showing only rows and columns that contain NaNs'''
    df = df.set_index(index)
    return df.loc[:, df.isna().any()][df.isna().any(axis=1)]
