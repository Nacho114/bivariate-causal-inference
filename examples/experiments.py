import numpy as np
import timeit

import synth_gen

import sys
sys.path.append("../twintest")

import causality


class DataGen:

    def __init__(self, noise, gamma=10, target=True):
        self.gamma = gamma
        self.target = target
        self.Nx = noise['Nx']
        self.Ny = noise['Ny']


    def random_fun(self):
        gps = synth_gen.GPsampler(self.gamma)
        f = gps.get_func()

        return f

    def generate(self, nb_samples):

        f = self.random_fun()
        Nx = self.Nx
        Ny = self.Ny

        binary_am = synth_gen.BinaryAM(Nx, Ny, f, nb_samples)
        x = binary_am.x
        y = binary_am.y

        target = self.target

        return x, y, target



def average_run(data, causal_estimator, nb_samples, nb_runs):

    acc = 0

    for _ in range(nb_runs):

        x, y, target = data.generate(nb_samples)
        _, _, pred  = causal_estimator(x, y)
        acc +=  int(pred == target)

    return acc / nb_runs


def run_experiment(causal_estimator, data, sample_range, nb_runs):

    results = []

    for nb_samples in sample_range:
        print('Currently running :', nb_samples)
        acc = average_run(data, causal_estimator, nb_samples, nb_runs)

        results.append(acc)

    return results


def noise_settings_gen():

    N1 = {
        'Nx': { 'name': 'uniform', 'var': 10},
        'Ny': {'name': 'normal', 'var': 0.2}
    }

    N2 = {
        'Nx': { 'name': 'exp', 'var': 10},
        'Ny': {'name': 'normal', 'var': 0.1}
    }

    noise_settings = [N1, N2]

    return noise_settings


def get_causal_inference(metric_name, model_params):

    causal_estimator = lambda x, y: causality.estimate_effect(x, y, metric_name=metric_name, return_scores=True, model_params=model_params)
    return causal_estimator

def get_file_name(metric_name, model_params, noise):

    return str(metric_name) + str(model_params['model_type']) + str(noise['Nx']['name']) + str(noise['Ny']['name']) + '.npy'


if __name__ == '__main__':

    # Params


    # estimator params

    metric_name = 'l1'

    model_type = 'PolyRegreg'
    model_params = {'model_type': model_type}
    causal_estimator = get_causal_inference(metric_name, model_params)

    # data gen
    gamma = 10
    noise_settings = noise_settings_gen()
    noise = noise_settings[0]

    data = DataGen(noise=noise, gamma=gamma)

    sample_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    nb_runs = 1000

    start_time = timeit.default_timer()
    results = run_experiment(causal_estimator, data, sample_range, nb_runs)
    time_taken = timeit.default_timer() - start_time

    print(time_taken)

    print(results)

    config = {
        'model_params': model_params,
        'sample_range': sample_range,
        'noise': noise, 
        'nb_runs': nb_runs,
        'results': results
    }

    loc = 'examples_data/'
    file_name = get_file_name(metric_name, model_params, noise)
    np.save(loc + file_name, config)

    # loaded_config = np.load(loc + file_name, allow_pickle=True)
    # print(loaded_config.item().get('noise'))

