import numpy as np
import stats

import mmd

def l1_score(res_i, res_j, bins, bounds):
    delta = stats.to_density(res_i, bins, bounds=bounds) - stats.to_density(res_j, bins, bounds=bounds)
    score = np.linalg.norm(delta, ord=1)

    return score

def mmd_score(res_i, res_j, sigma):

    res_i = res_i.reshape(-1, 1)
    res_j = res_j.reshape(-1, 1)
    return mmd.rbf_mmd2(res_i, res_j, sigma)


def get_score(residuals, pair, bins=None, bounds=None, metric_name='l1'):
    res_i = residuals[pair[0]]
    res_j = residuals[pair[1]]

    if metric_name == 'l1':
        return l1_score(res_i, res_j, bins, bounds)

    elif metric_name == 'mmd':
        return mmd_score(res_i, res_j, sigma=.5)

    elif metric_name == 'mmd_median_heuristic':
        return mmd_score(res_i, res_j, sigma=None)

    raise NameError('Unkown metric name: ' + metric_name)