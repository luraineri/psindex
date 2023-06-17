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
    psi = sum((ai - bi) * np.log(ai / bi))

    return psi


def psi_cont(sample_a, sample_b, bins):
    nai = np.histogram(sample_a, bins)[0]
    nbi = np.histogram(sample_b, bins)[0]

    ai = nai / nai.sum() 
    bi = nbi / nbi.sum()

    psi = sum((ai - bi) * np.log(ai / bi))

    return psi

def psi_cont2(sample_a, sample_b, bucket_type, nbins):
    d = {'sample_a': sample_a, 'sample_b': sample_b}
    df = pd.DataFrame(data=d)
    
    if bucket_type == 'bins':
        ser, bins = pd.cut(df['sample_a'], bins=nbins, retbins=True)
    elif bucket_type == 'quantiles':
        ser, bins = pd.qcut(df["sample_a"], q=nbins, retbins=True)
    
    df['bucket_a'] = pd.cut(df.sample_a, bins=bins)
    df['bucket_b'] = pd.cut(df.sample_b, bins=bins)

    return psi_cat(df['bucket_a'] , df['bucket_b'])