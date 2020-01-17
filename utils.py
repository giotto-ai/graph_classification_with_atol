import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = old_stdout
        # pass the return value of the method back
        return value

    return func_wrapper


def plot_diagram(diagram, padding=0):
    hom_dims = np.unique(diagram[:, -1]).astype(int)
    diagram = diagram[np.any(diagram[:, :-1]!=padding, axis=1)]
    colour_dictionary = {0: 'r', 1: 'b', 2: 'g', 3: 'c'}
    nb_times = max(len(hom_dims) // 4, 1)
    for time in range(nb_times):
        plt.figure()
        for hom_dim in np.arange(int(time*4), int(time*4+4)):
            diag_points = diagram[diagram[:, -1]==hom_dim]
            plt.plot(diag_points[:, 0], diag_points[:, 1], '.'+colour_dictionary[int(hom_dim-time*4)],
                     label=f'dim{hom_dim}')
    min_ = np.min(diagram[:, :-1])
    max_ = np.max(diagram[:, :-1])
    plt.plot([min_, max_], [min_, max_], 'k')
    plt.xlabel('birth')
    plt.ylabel('death')
    plt.legend()
    plt.show()



