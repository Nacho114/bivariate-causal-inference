import numpy as np
import stats

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

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


def unwrap_score_info(score_info):

    y_true = []
    y_score = []
    sample_weight = []

    for info in score_info:
        score = -info['score'] + info['score_reverse']
        y_score.append(score)
        y_true.append(info['target'])
        sample_weight.append(info['weight'])

    return y_true, y_score, sample_weight

def auc_score_from_score_info(score_info):

    y_true, y_score, sample_weight = unwrap_score_info(score_info)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_score, sample_weight=sample_weight)

    return roc_auc

def au_precision_recall_score(score_info):

    y_true, y_score, sample_weight = unwrap_score_info(score_info)
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score, sample_weight=sample_weight)
    precision_recall_score = auc(recall, precision)

    return precision_recall_score