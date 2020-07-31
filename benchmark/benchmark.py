import dataset
import random
import pandas as pd
import numpy as np

import cdt
from cdt.causality.pairwise import IGCI, ANM

import sys
sys.path.append("../twintest")

import causality
import metrics

def run_benchmark(data, print_progress=True, metric_name='l1', model_params=None, cdt_name=None, n_clusters=None, score_name=None):
    W = data.get_total_weight()
    acc = 0
    score_info = []

    for idx, (x, y, target, w) in enumerate(data):
        # method = get_cdt(cdt_name)
        # df = to_df(x, y)
        # score = method.predict(df)
        # if cdt_name == 'IGCI':
        #     score = score[0][0]
        # if cdt_name == 'ANM':
        #     score = score[0]
        # pred = score > 0
        # s = score
        # s_r = 0
        # assert 1 == 2

        try:
            if not cdt_name:
                s, s_r, pred = causality.estimate_effect(x, y, metric_name=metric_name, return_scores=True, model_params=model_params, n_clusters=n_clusters, score_name=score_name)
            else:
                method = get_cdt(cdt_name)
                df = to_df(x, y)
                score = method.predict(df)

                if cdt_name == 'IGCI':
                    score = score[0][0]
                if cdt_name == 'ANM':
                    score = score[0]

                pred = score > 0
                s = score
                s_r = 0



        except:
            print('except')
            pred = random.uniform(0, 1) > .5
            s = 0
            s_r = 0

        info = {
            'score': s,
            'score_reverse': s_r,
            'weight': w,
            'target': target,
            'idx': idx
        }

        score_info.append(info)

        if print_progress:
            print('Running: {}. pred: {}, actual {}'.format(idx, int(pred), target))

        acc += int(pred == target) * w

    perf = acc / W

    auc_score = metrics.au_precision_recall_score(score_info)

    if print_progress:
        print('\nAcc score: {:.3f}, AUC: {:.3f}'.format(perf, auc_score))

    return perf, auc_score

def get_benchmark_perf(data, print_progress=True, metric_name='l1'):
    W = data.get_total_weight()
    acc = 0

    scores = []
    for idx, (x, y, target, w) in enumerate(data):

        
        s, s_r, pred = causality.estimate_effect(x, y, metric_name=metric_name, return_scores=True)

        info = {
            'score': s,
            'score_reverse': s_r,
            'weight': w,
            'target': target,
            'idx': idx
        }

        scores.append(info)

        if print_progress:
            print('Running: {}. pred: {}, actual {}'.format(idx, int(pred), target))

        acc += int(pred == target) * w


    perf = acc / W

    if print_progress:
        print('\nacc: {:.3f}'.format(perf))

    return scores

def to_df(x, y):
    df = pd.DataFrame({'A':[np.asarray(x)]})
    df['B'] = [np.asarray(y)]
    return df

def get_cdt(name):

    if name == 'IGCI':
        method = IGCI()

    if name == 'ANM':
        method = ANM()

    return method


if __name__ == '__main__':

    db_names = ['CE-Tueb', 'CE-Gauss', 'CE-Cha', 'CE-Multi', 'CE-Net']
    # db_names = ['CE-Tueb']

    acc_list = []

    score_name = 'distrib_var'

    n_clusters = None

    cdt_name = None
    # cdt_name = 'ANM'

    metric_name = 'l1'
    # metric_name = 'mmd_median_heuristic'

    model_type = 'PolyRegreg'
    # model_type = 'NeuralNet'
    lr = [1, .1, .001]
    model_params = {'model_type': model_type, 'norm': False, 'learning_rate':lr, 'epochs': 400, 'H': 100}

    print(score_name)
    print(cdt_name)
    print(model_params)
    print(metric_name)

    for name in db_names:
        data = dataset.load_dataset(name)
        print('Running ', name)
        acc, auc_score = run_benchmark(data, print_progress=True, metric_name=metric_name, model_params=model_params, cdt_name=cdt_name, n_clusters=n_clusters, score_name=score_name)
        acc_list.append( (acc, auc_score) ) 
        print('Done.\n')

    print('Performance with ' + metric_name + ' metric:')
    for name, (acc, auc_score) in zip(db_names, acc_list):
         print('\n'+name+' Acc score: {:.3f}, AUC: {:.3f}'.format(acc, auc_score))

    print(cdt_name)
    print(model_params)
    print(metric_name)
    print(n_clusters)
    print(score_name)
    print('Other comments:')

