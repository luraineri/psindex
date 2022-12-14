import numpy as np
import pandas as pd
from psindex.psi import psi_cat

# testar com nan
# testar com uma categoria a mais e uma a menos

dfa = pd.DataFrame(
    {
        "cidade": [
            "Bariri",
            "Bariri",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
        ]
    }
)

dfb = pd.DataFrame(
    {
        "cidade": [
            "Bariri",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Jau",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
            "Sao Paulo",
        ]
    }
)

sample_a = dfa["cidade"]
sample_b = dfb["cidade"]


def test_psi_cat_same_sample():
    assert psi_cat(sample_a, sample_a) == 0


def test_psi_cat_samples():
    assert round(psi_cat(sample_a, sample_b), 2) == 0.25
