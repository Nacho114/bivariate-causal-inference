"""

Code adapted from

https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py

"""

import numpy as np


def median_heuristic(a):
    """Median heuristic to compute estimate for sigma
    """

    diff = a[np.newaxis, :] - a[:, np.newaxis]
    diff = np.triu(diff)
    diff = np.abs(diff[diff != 0])
    sigma = np.median(diff)

    return sigma

def get_sigma(x, y, sigma=.5):

    if sigma is None:
        x_y = np.vstack((x,y))
        sigma = median_heuristic(x_y) 

    return sigma

### Quadratic-time MMD with Gaussian RBF kernel
# Biased version
def rbf_mmd2(x, y, sigma=None, biased=True):
    sigma = get_sigma(x, y, sigma)
    gamma = 1 /  sigma**2
    
    XX = x.dot(x.T)
    XY = x.dot(y.T)
    YY = y.dot(y.T)
    
    X_sqnorms = np.diagonal(XX)
    Y_sqnorms = np.diagonal(YY)

    K_XY = np.exp(-gamma * (
            -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    K_XX = np.exp(-gamma * (
            -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
    K_YY = np.exp(-gamma * (
            -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

    if biased:
        mmd1 = K_XX.mean() - 2*K_XY.mean() + K_YY.mean()
        return mmd1

    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())

        return mmd2


### Linear-time MMD with Gaussian RBF kernel

# Estimator and the idea of optimizing the ratio from:
#    Gretton, Sriperumbudur, qSejdinovic, Strathmann, and Pontil.
#    Optimal kernel choice for large-scale two-sample tests. NIPS 2012.

def rbf_mmd2_streaming(X, Y, sigma=0.5):
    # n = (T.smallest(X.shape[0], Y.shape[0]) // 2) * 2
    n = (X.shape[0] // 2) * 2
    gamma = 1 / (2 * sigma**2)
    rbf = lambda A, B: np.exp(-gamma * ((A - B) ** 2).sum(axis=1))
    mmd2 = (rbf(X[:n:2], X[1:n:2]) + rbf(Y[:n:2], Y[1:n:2])
          - rbf(X[:n:2], Y[1:n:2]) - rbf(X[1:n:2], Y[:n:2])).mean()
    return mmd2, mmd2


# Median heuristic
def sigma_estimate():
    pass 