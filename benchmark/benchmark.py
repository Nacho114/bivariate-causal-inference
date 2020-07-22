import dataset

import sys
sys.path.append("../twintest")

import causality

def run_benchmark(data, print_progress=True, metric_name='l1'):
    W = data.get_total_weight()
    acc = 0

    for idx, (x, y, target, w) in enumerate(data):

        pred = causality.estimate_effect(x, y, metric_name=metric_name)
        if print_progress:
            print('Running: {}. pred: {}, actual {}'.format(idx, int(pred), target))

        acc += int(pred == target) * w

    perf = acc / W

    if print_progress:
        print('\nacc: {:.3f}'.format(perf))

    return perf


if __name__ == '__main__':

    db_names = ['CE-Tueb', 'CE-Gauss', 'CE-Cha', 'CE-Multi', 'CE-Net']

    acc_list = []

    metric_name = 'mmd_median_heuristic'

    for name in db_names:
        data = dataset.load_dataset(name)
        print('Running ', name)
        acc = run_benchmark(data, print_progress=True, metric_name=metric_name)
        acc_list.append(acc)
        print('Done.\n')

    print('Accuracies:\n')
    for name, acc in zip(db_names, acc_list):
        print(name + ' acc: {:.3f}'.format(acc))

