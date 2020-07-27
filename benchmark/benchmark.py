import dataset

import sys
sys.path.append("../twintest")

import causality
import metrics

def run_benchmark(data, print_progress=True, metric_name='l1'):
    W = data.get_total_weight()
    acc = 0
    score_info = []

    for idx, (x, y, target, w) in enumerate(data):

        s, s_r, pred = causality.estimate_effect(x, y, metric_name=metric_name, return_scores=True)

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

    

if __name__ == '__main__':

    db_names = ['CE-Tueb', 'CE-Gauss', 'CE-Cha', 'CE-Multi', 'CE-Net']

    acc_list = []

    metric_name = 'l1'#'mmd_median_heuristic'

    for name in db_names:
        data = dataset.load_dataset(name)
        print('Running ', name)
        acc, auc_score = run_benchmark(data, print_progress=True, metric_name=metric_name)
        acc_list.append( (acc, auc_score) ) 
        print('Done.\n')

    print('Performance with ' + metric_name + ' metric:')
    for name, (acc, auc_score) in zip(db_names, acc_list):
         print('\n'+name+' Acc score: {:.3f}, AUC: {:.3f}'.format(acc, auc_score))

