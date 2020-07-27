import numpy as np
import matplotlib.pyplot as plt

import causality

CONFIG = {
    'alpha': .7
}

def pretty_scatter(x, y, x_label=None, y_label=None, fname=None):
    plt.scatter(x, y, alpha=CONFIG['alpha'])

    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)

    if fname:
        plt.savefig(fname)


def color_map(k, lighten=None):

    def int_map(i):
        d_map = {
            0: -2, 1:3, 2: 4, 3:10, 4:2, 5: 9
        }

        if i not in d_map:
            return i
        # if i == 1:
        #     i = 5

        # i += 1
        return d_map[i]

    c = plt.get_cmap('Set3')(int_map(k))

    if not lighten:
        return c

    return lighten_color(c, lighten)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Taken from: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def plot_models(X_, Y_, models, inter_len=100, x_label=None, y_label=None, fname=None):
    plot_scatters(X_, Y_)

    for i, x_ in enumerate(X_):
        a = min(x_)
        b = max(x_)
        x_inter = np.linspace(a, b, inter_len)
        y_est = models[i].predict(x_inter)
        plt.plot(x_inter, y_est, color='black')

    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)

    if fname:
        plt.savefig(fname)

def plot_scatters(X_, Y_, x_label=None, y_label=None, fname=None):
    for i, (x_, y_) in enumerate(zip(X_, Y_)):
        plt.scatter(x_, y_, color=color_map(i), alpha=CONFIG['alpha'])

    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)

    if fname:
        plt.savefig(fname)

def plot_residuals(residuals, figsize=(10,5), res_filter=None, title='', bins=None, fname=None):

    if bins is None:
        bins = causality.determine_bin_size(residuals)

    nb_res = len(residuals)

    if res_filter is None:
        res_filter = range(nb_res)
    else:
        nb_res = len(res_filter)

    fig, axs = plt.subplots(1, nb_res, figsize=figsize)

    for i in range(nb_res): 
        k = res_filter[i]
        axs[i].hist(residuals[k], density=True, color=color_map(k), bins=bins)
        axs[i].set_title('Residual {}'.format(k))


    for ax in axs.flat:
        ax.set(xlabel='r', ylabel='frequency')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.suptitle(title)


    if fname:
        plt.savefig(fname)

