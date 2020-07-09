import tueb_data as data

import sys
sys.path.append("../twintest")

import causality


if __name__ == '__main__':

    meta_data = data.get_metadata()

    idx = 46
    data_info = meta_data[idx]
    x, y = data.load_sample(data_info)

    pred = causality.estimate_effect(x, y)

    print('Running: {}. pred: {}, actual {}'.format(idx, int(pred), data_info['causality']))





