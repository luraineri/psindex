from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.pandas import range_indexes, series
from numpy.testing import assert_almost_equal

from psindex.psi import psi_cat


@given(
    series(
        elements=st.integers(min_value=0, max_value=3),
        index=range_indexes(min_size=20, max_size=50),
    ).filter(lambda s: s.nunique() == 4),
    series(
        elements=st.integers(min_value=0, max_value=3),
        index=range_indexes(min_size=20, max_size=50),
    ).filter(lambda s: s.nunique() == 4),
)
def test_commutative_property(a, b):
    assert_almost_equal(psi_cat(a, b), psi_cat(b, a))


@given(
    series(
        elements=st.integers(min_value=0, max_value=3),
        index=range_indexes(min_size=20, max_size=50),
    ),
)
def test_identical_samples_property(a):
    assert_almost_equal(psi_cat(a, a), 0)
