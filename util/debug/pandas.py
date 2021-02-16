import pandas as pd

seriesSideBySide = lambda s1, s2: pd.merge(s1, s2, right_index=True, left_index=True)
