import tueb_data as data

import sys
sys.path.append("../twintest")

import causality


if __name__ == '__main__':

    meta_data = data.get_metadata()
    W = data.get_total_weight(meta_data)
    acc = 0

    for idx, data_info in enumerate(meta_data):
        x, y = data.load_sample(data_info)
        w = data_info['weight']

        pred = causality.estimate_effect(x, y)
        print('Running: {}. pred: {}, actual {}'.format(idx, int(pred), data_info['causality']))

        acc += int(pred == data_info['causality']) * w

    print('\nacc:', acc / W)




