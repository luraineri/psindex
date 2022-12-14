import numpy as np


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
    psi = sum((ai - bi) * np.log10(ai / bi))

    return psi
