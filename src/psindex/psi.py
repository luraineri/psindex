import numpy as np
import pandas as pd


def psi_cat(sample_a, sample_b):
    # calc nai, nbi, Na, Nb - considerando categorias de 'a'
    categories_a = sample_a.unique()

    nai = sample_a.value_counts()
    nbi = sample_b.value_counts()

    ai = [nai[cat] / len(sample_a) for cat in categories_a]
    bi = [nbi[cat] / len(sample_b) for cat in categories_a]
    ai = np.array(ai)
    bi = np.array(bi)

    # calc PSI
    psi = sum((ai - bi) * np.log(ai / bi))

    return psi


def psi_cont(sample_a, sample_b, nbins: int):

    _, bins_edges = pd.cut(sample_a, bins=nbins, retbins=True)

    bucket_a = pd.cut(sample_a, bins=bins_edges)
    bucket_b = pd.cut(sample_b, bins=bins_edges)

    return psi_cat(bucket_a, bucket_b)
