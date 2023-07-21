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

import gurobipy as gp
from gurobipy import GRB

from lpsolve import solve_lp


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

    scaled_eps = epsilon / (1.0 + (epsilon / 4.0))

    # for calculating error
    mu = k - 1

    h = np.full((N, 1), 1.0 / N, dtype=np.longdouble) # weights
    X = np.zeros((N, 1))         # Output

    T = ((8 * mu) / (math.pow(scaled_eps, 2))) * math.log(N, math.e) # iterations
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
            W += np.sum(w_sums[min_indecies])

        if W >= 1:
            return None

        # get counts of points in each ball in M
        M = np.zeros_like(h)
        Z = BallTree.create(S)

        Cs = BallTree.get_counts_in_range(Z, features, gamma / 2.0)
        for i, c in enumerate(Cs):
            M[i] = (1.0 / mu) * ((-1 * c) + 1)

        # update H
        oldH = np.copy(h)
        # TODO: check sign of value
        h = h * (np.ones_like(M) - ((scaled_eps / 4.0) * M))
        h /= np.sum(h)

        """
        # print X for testing
        file = f"test_{kis['red']}_{kis['blue']}_{gamma}_{epsilon}_output.txt"
        if t == 0:
            # zero file on first cycle
            open(file, 'w').close()

        with open(file, "a") as f:
            with np.printoptions(linewidth=np.inf):
                #print(f'{np.sum(X[0:4] / (t + 1))}', file=f)
                #print(f'{W}', file=f, end="\n")
                print((X / (t+1)).flatten(), file=f, end="\t")
                print(f'\th distance: {np.linalg.norm(h - oldH)}', file=f)
        """

        # If things aren't changing, stop early
        # TODO: figure out a better way to stop early

        #if np.allclose(h, oldH, atol=1e-08):
        #    print(f'Exiting early on iteration {t+1}')
        #    break

        # check directly if X is a feasible solution
        if t % 50 == 0:
            X_weights = KDTree2.get_weight_ranges(struct, X / (1 + t), gamma / 2.0)
            if not np.any(X_weights > 1 + epsilon):
                print(f'Breaking early due to feasible solution on iteration {t+1}!')
                break


        #tqdm.write(f'\th distance: {np.linalg.norm(h - oldH)}', end='')

    # TODO: check if X changed or not
    X = X / (t + 1)
    return X


def epsilon_falloff(features, colors, coreset_size, k, a, mwu_epsilon, falloff_epsilon):
    """
    starts at a high bound (given by the corset estimate) and repeatedly falls off by 1-epsilon
    :param features: the data set
    :param colors:   color labels for the data set
    :param coreset_size: the number of points to use in the corset
    :param k:        total K to pick from the graph
    :param a:        fraction of K acceptable to lose
    :param mwu_epsilon: epsilon for the MWU method (static error)
    :param falloff_epsilon: epsilon for the falloff system (fraction to reduce by each cycle)
    :return:
    """

    # get the colors
    color_names = np.unique(colors)

    timer = utils.Stopwatch("Normalization")

    # "normalize" features
    # Should happen before coreset construction
    means = features.mean(axis=0)
    devs = features.std(axis=0)
    features = (features - means) / devs

    timer.split("Coreset")

    d = len(feature_fields)
    m = len(color_names)
    coreset = CORESET.Coreset_FMM(features, colors, k, m, d, coreset_size)
    features, colors = coreset.compute()

    N = len(features)

    timer.split("Building Kis")

    kis = utils.buildKisMap(colors, k, a)

    timer.split("Falloff")

    gamma = coreset.compute_gamma_upper_bound()

    X = mult_weight_upd(gamma, N, k, features, colors, kis, mwu_epsilon)
    while X is None:
        gamma = gamma * (1 - falloff_epsilon)
        X = mult_weight_upd(gamma, N, k, features, colors, kis, mwu_epsilon)

    timer.split("Randomized Rounding")

    # we need to flatten X since it expects an array rather than a 1D vector
    S = rand_round(gamma / 2.0, X.flatten(), features, colors, kis)

    _, total_time = timer.stop()

    # build up stats to return
    selected_count = len(S)
    solution = features[S]

    diversity = utils.compute_maxmin_diversity(solution)
    print(f'{k} solved!')
    print(f'Diversity: {diversity}')
    print(f'Time (S):  {total_time}')

    return selected_count, diversity, total_time

if __name__ == '__main__':
    """
    # Testing field
    # File fields
    allFields = [
        "x",
        "y",
        "color",
    ]

    # fields we care about for parsing
    color_field = ['color']
    feature_fields = ['x', 'y']

    # variables for running LP bin-search
    # keys are appended using underscores
    kis = {
        'blue': 2,
        'red': 1,
    }
    k = sum(kis.values())
    """

    # File fields
    allFields = [
        "age",
        "workclass",
        "fnlwgt",  # what on earth is this one?
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "yearly-income",
    ]

    # fields we care about for parsing
    color_field = ['race', 'sex']
    feature_fields = {'age', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt', 'education-num'}

    colors, features = utils.read_CSV("./datasets/ads/adult.data", allFields, color_field, '_', feature_fields)
    assert (len(colors) == len(features))

    # testing!
    results = []

    for k in range(10, 201, 5):
        selected, div, time = epsilon_falloff(
            features=features,
            colors=colors,
            coreset_size=10000,
            k=k,
            a=0,
            mwu_epsilon=0.75,
            falloff_epsilon=0.1,
        )
        results.append((k, selected, div, time))

    print('\n\nFINAL RESULTS:')
    print('k\tselected\tdiversity\ttime')
    for k, selected, div, time in results:
        print(f'{k},\t{selected},\t{div},\t{time},')