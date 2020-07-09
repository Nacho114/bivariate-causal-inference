import glob
import os
import pandas as pd
import numpy as np

DB_NAMES = ['CE-Gauss', 'CE-Cha', 'CE-Multi', 'CE-Net']


def get_file_paths(db_name):

    assert db_name in DB_NAMES

    pair_relative_path = './data/CE-Synth'

    file_names = ''

    if db_name == 'CE-Gauss':
        file_names = glob.glob(os.path.join(pair_relative_path, '*.csv'))
    else:
        pair_relative_path = os.path.join(pair_relative_path, 'Pairwise datasets')
        file_names = glob.glob(os.path.join(pair_relative_path, '*.csv'))
        file_names = [file_n for file_n in file_names if db_name in file_n]

    file_1, file_2 = file_names

    if 'pairs' in file_1:
        return file_1, file_2

    return file_2, file_1


def str_to_float_array(str):

    arr = np.array((str.split()))
    return arr.astype(np.float)
