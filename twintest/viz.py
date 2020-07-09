import numpy as np
import matplotlib.pyplot as plt

CONFIG = {
    'alpha': .4
}

def pretty_scatter(x, y):
    plt.scatter(x, y, alpha=CONFIG['alpha'])

def color_map(k):
    def int_map(i):
        d_map = {
            0: -2, 1:3, 2: 4, 3:30, 4:2, 5: 9
        }

        if i not in d_map:
            return i

        return d_map[i]

    return plt.get_cmap('Set3')(int_map(k))

def plot_models(X_, Y_, models, inter_len=100):
    plot_scatters(X_, Y_)

    for i, x_ in enumerate(X_):
        a = min(x_)
        b = max(x_)
        x_inter = np.linspace(a, b, inter_len)
        y_est = models[i].predict(x_inter)
        plt.plot(x_inter, y_est, color='black')

def plot_scatters(X_, Y_):
    for i, (x_, y_) in enumerate(zip(X_, Y_)):
        plt.scatter(x_, y_, color=color_map(i))

def plot_residuals(residuals, figsize=(10,5), res_filter=None, title=''):
    nb_res = len(residuals)

    if res_filter is None:
        res_filter = range(nb_res)
    else:
        nb_res = len(res_filter)

    fig, axs = plt.subplots(1, nb_res, figsize=figsize)

    for i in range(nb_res): 
        k = res_filter[i]
        axs[i].hist(residuals[k], density=True, color=color_map(k))
        axs[i].set_title('Residual {}'.format(k))


    for ax in axs.flat:
        ax.set(xlabel='r', ylabel='frequency')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.suptitle(title)

