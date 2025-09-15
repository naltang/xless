import numpy as np
from scipy.spatial.distance import pdist, squareform

def jsd(p, q):
    p = p + 1e-12; q = q + 1e-12
    p /= p.sum(); q /= q.sum()
    m = 0.5*(p+q)
    return 0.5*np.sum(p*np.log(p/m)) + 0.5*np.sum(q*np.log(q/m))

# Assume `data` is an (N x K) array of histograms
dist_mat = squareform(pdist(data, metric=jsd))
