import numpy as np
import pandas as pd


MISSING_CATEGORY_FREQ = 0.001


def calc_freq(sample_a, sample_b, dropna):
    freq_a = sample_a.value_counts(normalize=True, dropna=dropna)
    freq_b = sample_b.value_counts(normalize=True, dropna=dropna)

    # categories with zero count appear in value_counts result if the series
    # come from pd.cut. Here we remove them
    freq_a = freq_a[freq_a > 0]
    freq_b = freq_b[freq_b > 0]

    categories_a = freq_a.index
    categories_b = freq_b.index

    missing_categories = set(categories_a) - set(categories_b)

    for cat in missing_categories:
        freq_b[cat] = MISSING_CATEGORY_FREQ

    freq_b = freq_b[categories_a.to_list()]

    return freq_a, freq_b


def psi_cat(sample_a, sample_b, dropna=True):
    freq_a, freq_b = calc_freq(sample_a, sample_b, dropna)

    psi = sum((freq_a - freq_b) * np.log(freq_a / freq_b))

    return psi


def psi_cont(sample_a, sample_b, nbins: int, dropna=True):

    _, bins_edges = pd.cut(sample_a, bins=nbins, retbins=True)

    bucket_a = pd.cut(sample_a, bins=bins_edges)
    bucket_b = pd.cut(sample_b, bins=bins_edges)

    return psi_cat(bucket_a, bucket_b, dropna)
