
import pandas as pd

import tueb_data
import synth_data

class Dataset:
    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def get_total_weight(self):
        pass

    def get_name(self, idx):
        pass


class TuebDataset(Dataset):

    def __init__(self, only_bivariate=True):

        self.meta_data = tueb_data.get_metadata(only_bivariate)
        self.W = tueb_data.get_total_weight(self.meta_data)

    def __getitem__(self, idx):

        data_info = self.meta_data[idx]

        x, y = tueb_data.load_sample(data_info)
        w = data_info['weight']
        target = data_info['causality']

        return x, y, target, w

    def __len__(self):
        return len(self.meta_data)

    def get_total_weight(self):
        return self.W

    def get_name(self, idx):
        return self.meta_data[idx]['pair_number']

class SynthDataset(Dataset):

    def __init__(self, db_name):

        pairs_path, targets_path = synth_data.get_file_paths(db_name)

        pairs_df = pd.read_csv(pairs_path)
        targets_df = pd.read_csv(targets_path)

        self.pairs = pairs_df.to_numpy()
        self.targets = targets_df.to_numpy()

    def __getitem__(self, idx):

        _, x, y = self.pairs[idx, :]

        x = synth_data.str_to_float_array(x)
        y = synth_data.str_to_float_array(y)

        _, target = self.targets[idx]

        target = int(target == 1)

        return x, y, target, 1

    def __len__(self):
        return self.pairs.shape[0]

    def get_total_weight(self):
        return self.__len__()

    def get_name(self, idx):
        return idx


def load_dataset(name):
    """Available data sets:
        'CE-Tueb', 'CE-Gauss', 'CE-Cha', 'CE-Multi', 'CE-Net'
    """
    if name == 'CE-Tueb':
        return TuebDataset()

    if name in synth_data.DB_NAMES:
        return SynthDataset(name)

    print('Unkown dataset')
    return None