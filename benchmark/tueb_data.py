import glob
import os
import pandas as pd
import numpy as np

def get_metadata(only_bivariate=True):
    pair_relative_path = './data/CE-Tueb'

    pair_file_names = glob.glob(os.path.join(pair_relative_path, '*.txt'))

    def is_metadata(name):
        meta_names = ['_des.txt', 'pairmeta.txt']
        for meta_n in meta_names:
            if meta_n in name:
                return True
            
        return False

    data_filenames = [name for name in pair_file_names if not is_metadata(name)]
    data_filenames = sorted(data_filenames)

    header = ['number of pair',  '1st column of cause', 'last column of cause', '1st column of effect', 'last column of effect', 'dataset weight']

    meta_data_file_name = 'pairmeta.txt'
    df = pd.read_csv(os.path.join(pair_relative_path, meta_data_file_name), names=header, sep=" ", index_col=0)

    meta_matrix = df.to_numpy()

    def bivariate_type(row):

        if row[0] == 1 and row[1] == 1 and row[2] == 2 and row[3] == 2:
            return 1

        if row[0] == 2 and row[1] == 2 and row[2] == 1 and row[3] == 1:
            return 0

        # Not bivariate
        return -1
        
    simple_meta = []
    for idx, row in enumerate(meta_matrix):
        causality_type = bivariate_type(row)
        data_info = {
            'pair_number': idx + 1,
            'causality': causality_type,
            'weight': row[4],
            'file_path': data_filenames[idx]

        }
        if not only_bivariate or (only_bivariate and causality_type != -1):
            simple_meta.append(data_info)

    return simple_meta 

def get_total_weight(meta_data):
    W = 0
    for data_info in meta_data:
        W += data_info['weight']
    
    return W

def unpack_data(data_matrix):
    x = data_matrix[:, 0]
    y = data_matrix[:, 1]

    return x, y

def load_sample(data_info):
    # regex sep to use any type of space/tabs as deliminator as data is inconsistent
    df = pd.read_csv(data_info['file_path'], sep=r"\s+|\t+|\s+\t+|\t+\s+", header=None, engine='python')
    data_matrix = df.to_numpy()

    x, y = unpack_data(data_matrix)
    
    return x, y

def print_statistics():

    meta_data = get_metadata(only_bivariate=False)
    lens = []
    ctr = 0
    for data_info in meta_data:

        if not data_info['causality'] == -1:
            x, y = load_sample(data_info)
            lens.append(len(x))
            assert len(x) == len(y)
            ctr += 1

    print('Basic stats:\n') 
    print('Minimum nb samples: {}\nMean nb samples: {} \nMax nb samples: {}'.format(min(lens), int(np.mean(lens)), max(lens)))
    print('\nnb bivariate datasets:', ctr)
    print('Total nb of datasets:', len(meta_data))
