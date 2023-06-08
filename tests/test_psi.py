import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from psindex.psi import psi_cat

from .utils import make_series

# testar com nan
# testar com uma categoria a mais e uma a menos


@pytest.mark.parametrize(
    ["a", "b", "exp_result"],
    [
        (
            make_series({"a": 20, "b": 30, "c": 40, "d": 50}),
            make_series({"a": 10, "b": 30, "c": 40, "d": 50}),
            0.045702,
        ),
        (
            make_series({"a": 2, "b": 3, "c": 4, "d": 5}),
            make_series({"a": 3, "b": 4, "c": 5, "d": 6}),
            0.005825,
        ),
    ],
)
def test_psi_cat_samples(a, b, exp_result):
    assert_almost_equal(psi_cat(a, b), exp_result, decimal=6)


def test_same_distribution_series():
    s = make_series({"a": 10, "b": 20, "c": 30})
    assert_almost_equal(psi_cat(s, pd.concat([s] * 10)), 0)
