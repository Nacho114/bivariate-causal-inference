import dataset

import sys
sys.path.append("../twintest")

import causality


if __name__ == '__main__':

    # 'CE-Tueb', 'CE-Gauss', 'CE-Cha', 'CE-Multi', 'CE-Net'
    
    data = dataset.load_dataset('CE-Tueb')

    idx = 0
    x, y, target, w = data[idx]

    print(len(data))

    pred = causality.estimate_effect(x, y)

    print('Running: {}. pred: {}, actual {}'.format(idx, int(pred), target))


