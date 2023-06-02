import math
from collections import defaultdict

import sys
import numpy as np

from tqdm import tqdm, trange
import typing as t
import numpy.typing as npt

import KDTree2
import BallTree
import coreset as CORESET
import utils
from rounding import rand_round

def mult_weight_upd(gamma, N, k, features, colors, kis, epsilon):
    """
    uses the multiplicative weight update method to
    generate an integer solution for the LP
    :param gamma: the minimum distance to optimize for
    :param N: the number of elements in the dataset
    :param k: total number of points selected
    :param features: dataset's features
    :param colors: matching colors
    :param kis: the color->count mapping
    :param epsilon: allowed error value
    :return: a nx1 vector X of the solution or None if infeasible
    """
    assert(k > 0)

    # TODO: update epsilon if needed
    h = np.full((N, 1), 1.0 / N) # weights
    X = np.zeros((N, 1))         # Output

    T = (8 * k) / (math.pow(epsilon, 2)) * math.log(N, math.e) # iterations
    # for now, we can recreate the structure in advance
    struct = KDTree2.create(features)

    # NOTE: should this be <= or was it 1 indexed?
    for t in trange(math.ceil(T), desc='MWU Loop'):

        S = np.empty((0, features.shape[1]))  # points we select this round
        W = 0                                 # current weight sum

        # weights to every point
        w_sums = KDTree2.get_weight_ranges(struct, h, gamma / 2.0)

        # compute minimums per color
        for color in kis.keys():
            # need this to reverse things
            color_sums_ind = (color == colors).nonzero()[0] # tuple for reasons

            # get minimum points as indices
            color_sums = w_sums[color_sums_ind]
            partition = np.argpartition(color_sums, kis[color] - 1)
            arg_mins = partition[:kis[color]]
            min_indecies = color_sums_ind[arg_mins]

            # add 1 to X[i]'s that are the minimum indices
            X[min_indecies] += 1
            # add points we've seen to S
            S = np.append(S, features[min_indecies], axis=0)
            # add additional weight to W
            W += np.sum(h[min_indecies])

        if W >= 1:
            return None

        # get counts of points in each ball in M
        M = np.zeros_like(h)
        Z = BallTree.create(S)

        Cs = BallTree.get_counts_in_range(Z, features, gamma / 2.0)
        for i, c in enumerate(Cs):
            M[i] = (1.0 / k) * ((-1 * c) + 1)

        # update H
        oldH = np.copy(h)
        # TODO: check sign of value
        h = h * (1 - ((epsilon / 4.0) * M))
        h /= np.sum(h)

        # print X for testing
        file = f"test_{kis['red']}_{kis['blue']}_{gamma}_{epsilon}_output.txt"
        if t == 0:
            # zero file on first cycle
            open(file, 'w').close()

        with open(file, "a") as f:
            with np.printoptions(linewidth=np.inf):
                print((X / (t+1)).flatten(), file=f)

        # If things aren't changing, stop early
        # TODO: figure out a better way to stop early
        if np.allclose(h, oldH, atol=1e-08):
            print(f'Exiting early on iteration {t+1}')
            break

        #tqdm.write(f'\th distance: {np.linalg.norm(h - oldH)}', end='')

    # TODO: check if X changed or not
    X = X / (t + 1)
    return X


if __name__ == '__main__':
    # File fields
    allFields = [
        "x",
        "y",
        "color",
    ]

    # fields we care about for parsing
    color_field = ['color']
    feature_fields = {'x', 'y'}

    # variables for running LP bin-search
    # keys are appended using underscores
    kis = {
        'blue': 2,
        'red': 2,
    }
    k = sum(kis.values())
    # binary search params
    epsilon = np.float64("0.001")

    # coreset params
    # Set the size of the coreset
    coreset_size = 30000

    # start the timer
    timer = utils.Stopwatch("Parse Data")

    colors, features = utils.read_CSV("./datasets/mwutest/example.csv", allFields, color_field, '_', feature_fields)
    assert (len(colors) == len(features))

    """
    # "normalize" features
    # Should happen before coreset construction
    # TODO: remove normalization
    features = features / features.max(axis=0)

    timer.split("Coreset")

    print("Number of points (original): ", len(features))
    d = len(feature_fields)
    m = len(kis.keys())
    features, colors = CORESET.Coreset_FMM(features, colors, k, m, d, coreset_size).compute()
    print("Number of points (coreset): ", len(features))
    
    """

    N = len(features)

    timer.split("One MWU round")

    # TODO: remove hardcoding
    gamma = 5

    X = mult_weight_upd(gamma, N, k, features, colors, kis, 0.01)

    res = timer.stop()
    print('Timings! (seconds)')
    for name, delta in res:
        print(f'{name + ":":>40} {delta}')

    if X is None:
        print('Infeasible!')
    else:
        print(X)