import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from psindex.psi import psi_cat, psi_cont

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


@pytest.mark.parametrize(
    ["a", "b", "exp_result"],
    [
        (
            make_series({"a": 3, "b": 2, "c": 3, "d": 1}),
            make_series({"a": 2, "b": 2, "c": 3, np.nan: 2}),
            0.565913,
        ),
        (
            make_series({"a": 2, "b": 2, "c": 3, np.nan: 2}),
            make_series({"a": 3, "b": 2, "c": 3, "d": 1}),
            0.047231,
        ),
        (
            make_series({"a": 3, "b": 2, "c": 3, "d": 1, np.nan: 2}),
            make_series({"a": 3, "b": 2, "c": 3, "d": 1, np.nan: 2}),
            0,
        ),
    ],
)
def test_missing_values_cat(a, b, exp_result):
    assert_almost_equal(psi_cat(a, b), exp_result, decimal=6)


def test_same_proportion_cont():
    a = make_series({1: 3, 5: 3, 10: 3})
    b = make_series({1: 1, 5: 1, 10: 1})

    assert_almost_equal(psi_cont(a, b, nbins=3), 0, decimal=6)


@pytest.mark.parametrize(
    ["a", "b", "exp_result", "dropna"],
    [
        (
            make_series({1: 3, 5: 3, 10: 3, np.nan: 10}),
            make_series({1: 1, 5: 1, 10: 1}),
            0,
            True,
        ),
        (
            make_series({1: 1, 5: 1, 10: 1}),
            make_series({1: 3, 5: 3, 10: 3, np.nan: 10}),
            0,
            True,
        ),
        (
            make_series({1: 1, 5: 1, 10: 1, np.nan: 1000}),
            make_series({1: 3, 5: 3, 10: 3, np.nan: 10}),
            0,
            True,
        ),
        (
            make_series({1: 3, 5: 3, 10: 3, np.nan: 10}),
            make_series({1: 1, 5: 1, 10: 1}),
            3.684847,
            False,
        ),
        (
            make_series({1: 1, 5: 1, 10: 1}),
            make_series({1: 3, 5: 3, 10: 3, np.nan: 10}),
            0.393270,
            False,
        ),
        (
            make_series({1: 1, 5: 1, 10: 1, np.nan: 1000}),
            make_series({1: 3, 5: 3, 10: 3, np.nan: 10}),
            2.684731,
            False,
        ),
    ],
)
def test_missing_values_cont(a, b, exp_result, dropna):
    assert_almost_equal(psi_cont(a, b, nbins=3, dropna=dropna), exp_result, decimal=6)
