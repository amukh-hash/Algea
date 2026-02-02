import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

def get_ivp(cov):
    # Inverse Variance Portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def get_cluster_var(cov, c_items):
    # Variance of a cluster
    cov_slice = cov[c_items][:, c_items]
    w = get_ivp(cov_slice).reshape(-1, 1)
    c_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
    return c_var

def get_quasi_diag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]) # sort_ix.append(df0)
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def get_rec_bipart(cov, sort_ix):
    # Recursive Bisection
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
    return w

def compute_hrp_weights(returns: pd.DataFrame) -> dict:
    """
    Computes HRP weights.
    returns: DataFrame of asset returns (T x N).
    """
    if returns.empty:
        return {}

    # 1. Correlation & Covariance
    corr = returns.corr().fillna(0)
    cov = returns.cov().fillna(0)

    # 2. Distance Matrix
    dist = np.sqrt((1 - corr) / 2)
    # Handle floating point errors
    dist = np.clip(dist, 0, 1)

    # 3. Clustering
    # squareform requires symmetric matrix with 0 diag
    dist_arr = dist.values
    np.fill_diagonal(dist_arr, 0)
    condensed_dist = squareform(dist_arr)

    link = sch.linkage(condensed_dist, 'single')

    # 4. Sorting
    sort_ix = get_quasi_diag(link)
    sort_ix = corr.index[sort_ix].tolist()

    # 5. Allocation
    # Rearrange cov to match sort order
    cov_sorted = cov.loc[sort_ix, sort_ix].values

    weights_series = get_rec_bipart(cov_sorted, range(len(sort_ix)))
    weights = pd.Series(weights_series.values, index=sort_ix)

    return weights.to_dict()
