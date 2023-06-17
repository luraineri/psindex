from itertools import chain
from typing import Dict

import pandas as pd


def make_series(counts: Dict) -> pd.Series:
    elems = [[elem] * count for elem, count in counts.items()]
    return pd.Series(chain(*elems))
