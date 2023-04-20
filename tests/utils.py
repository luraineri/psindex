from functools import reduce
from typing import Dict

import pandas as pd


def make_series(counts: Dict) -> pd.Series:
    series = pd.Series(
        reduce(lambda a, b: a + b, [[elem] * count for elem, count in counts.items()])
    )

    return series
