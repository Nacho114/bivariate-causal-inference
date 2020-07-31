from sklearn.cluster import KMeans
import numpy as np
import math
import sys

import stats
import metrics

###################################
# Partition data based on kmeans
###################################

def fraction_for_size(n):
    # gives best fraction values for kmeans 
    if n <= 1000:
        return 0.3897341 - 0.0005041345*n + 2.16147e-7*n*n
    elif n <= 6000:
        return 0.12 - 0.00002166667*n + 1.666667e-9*n*n
    return .05


def determine_partition_size(x, max_n_clusters=10):
    # Note, make max_n_clusters also as a fn of n

    n = len(x)
    for n_c in range(2, max_n_clusters):
        
        kmeans = KMeans(n_clusters=n_c, random_state=0).fit(x.reshape(-1, 1)) 
        labels = kmeans.labels_

        densities = [len(x[labels==i].reshape(-1, 1))/n for i in range(n_c)]

        if fraction_for_size(n) > min(densities):
            # Require at least 2 partitions
            return max(n_c - 1, 2) 
        
    return max_n_clusters

def sort_labels(kmeans):

    labels = kmeans.labels_

    centers = kmeans.cluster_centers_
    centers = np.squeeze(centers, axis=1)

    l_map = np.argsort(centers)
    labels_ = np.zeros_like(labels)
    for new_label, old_label in enumerate(l_map):
        labels_[labels == old_label] = new_label
    return labels_

def partition_data(x, y, n_clusters=None, sorted_labels=True):

    if n_clusters is None:
        n_clusters = determine_partition_size(x)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x.reshape(-1, 1))
    labels = kmeans.labels_

    if sorted_labels:
        labels = sort_labels(kmeans)

    X_ = []
    Y_ = []

    for i in range(n_clusters):
        X_.append(x[labels==i].reshape(-1, 1))
        Y_.append(y[labels==i])

    return X_, Y_, labels


# fit data
def fit_partitions(X_, Y_, model_params):

    models = []

    for x, y in zip(X_, Y_):
        # Fit model
        # model = stats.model_selection(x, y)
        model = stats.get_model(x, y, model_params)

        models.append(model)

    return models


###################################
# Identical test functions
###################################

def stugers(n):
    # Using Stuger's rule, k ~ 1 + log(n)
    
    k = 3
    
    if n < 500:
        k = 1 + np.log(n)
        
    elif n < 1000:
        k = 2 * (1 + np.log(n))
        
    elif n < 5000:
        k = 3 * (1 + np.log(n))
    elif n < 8000:
        k = 4 * (1 + np.log(n))
        
    else:
        k = 40
        
    return int(k)

def determine_bin_size(residuals):
    min_r_size = min([len(r) for r in residuals])
    return stugers(min_r_size)

def find_residual_bounds(residuals):
    max_res_vals = [max(r) for r in residuals]
    min_res_vals = [min(r) for r in residuals]
    
    return min(min_res_vals), max(max_res_vals)


def get_score(residuals, bins=None, metric_name='l1', score_name='max_discrep'):

    score = None

    if score_name == 'max_discrep':

        score = find_max_discrp(residuals, bins=bins, metric_name=metric_name)


    elif score_name == 'distrib_var':

        score = find_res_distrib_var(residuals, bins=bins)

    else:
        raise NameError(score_name)


    return score



def find_max_discrp(residuals, bins=None, metric_name='l1'):

    if bins is None:
        bins = determine_bin_size(residuals)

    pairs = stats.get_pairs(len(residuals))

    best_pair = 0, 0
    max_score = -math.inf

    # find the range for the density discretisationg
    # bounds = min(residuals), max(residuals)
    bounds = find_residual_bounds(residuals)

    for pair in pairs:
        score = metrics.get_score(residuals, pair, bins, bounds, metric_name)
        if score > max_score:
            max_score = score
            best_pair = pair

    return max_score, best_pair

def find_res_distrib_var(residuals, bins=None):
    if bins is None:
        bins = determine_bin_size(residuals)

    # find the range for the density discretisationg
    # bounds = min(residuals), max(residuals)
    bounds = find_residual_bounds(residuals)

    nb_res = len(residuals)

    mean_residual_density = sum([stats.to_density(r, bins, bounds=bounds) for r in residuals]) / nb_res

    total_diff = 0
    for r in residuals:
        delta = stats.to_density(r, bins, bounds=bounds) - mean_residual_density
        total_diff += np.linalg.norm(delta, ord=1)
     

    return total_diff / nb_res, None

###################################
# Main functions
###################################

def estimate_partitioned_models(x, y, n_clusters=None, model_params=None):
    """x is partitioned into n_clusters using kmeans (if k_means=None n_clusters is estimated
    based on the len(x). For each cluster we fit a model using the stats model selection method.
    """
    # We first partition the data
    X_, Y_, _ = partition_data(x, y, n_clusters)

    # For each partition we compute a model
    models = fit_partitions(X_, Y_, model_params)

    # We compute the residuals
    residuals = stats.compute_residuals(X_, Y_, models)

    return residuals, X_, Y_, models


def estimate_effect(x, y, n_clusters=None, bins=None, min_cluster_rule=False, metric_name='l1', return_scores=False, model_params=None, score_name=None):
    """Estimates the causal effect: Returns 1 if X -> Y and 0 if Y -> X"""

    if score_name is None:
        score_name = 'max_discrep'

    if min_cluster_rule:
        size_x = determine_partition_size(x)
        size_y = determine_partition_size(y)
        n_clusters = max(size_x, size_y)
    
    # We fit x->y
    residuals, _, _, _ = estimate_partitioned_models(x, y, n_clusters, model_params=model_params)
    score, _ = get_score(residuals, bins, metric_name, score_name=score_name)

    # We fit y->x
    residualsr, _, _, _ = estimate_partitioned_models(y, x, n_clusters, model_params=model_params)
    scorer, _ = get_score(residualsr, bins, metric_name, score_name=score_name)

    direction = score < scorer

    if return_scores:
        return score, scorer, direction

    return direction
