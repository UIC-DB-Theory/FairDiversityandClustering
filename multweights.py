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

# TODO: early stop on X vector in slack
    # try pre-processing ball query distances

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

    #TODO: group formulas => max(1, (1-a) * k n_c / n)
    # a = 0.2
    # n = number of input points in color c
    # (k * n_c / n) <- is all we need!

    # variables for running LP bin-search
    # keys are appended using underscores
    kis = {
        'White_Male': 15,
        'White_Female': 35,
        'Asian-Pac-Islander_Male': 55,
        'Asian-Pac-Islander_Female': 35,
        'Amer-Indian-Eskimo_Male': 15,
        'Amer-Indian-Eskimo_Female': 35,
        'Other_Male': 15,
        'Other_Female': 35,
        'Black_Male': 15,
        'Black_Female': 35,
    }
    k = sum(kis.values())
    print(k)

    # binary search params
    #epsilon = np.float64("0.001")

    # coreset params
    # Set the size of the coreset
    coreset_size = 10000

    # start the timer
    timer = utils.Stopwatch("Parse Data")

    colors, features = utils.read_CSV("./datasets/ads/adult.data", allFields, color_field, '_', feature_fields)
    assert (len(colors) == len(features))

    # "normalize" features
    # Should happen before coreset construction
    means = features.mean(axis=0)
    devs  = features.std(axis=0)
    features = (features - means) / devs

    timer.split("Coreset")

    print("Number of points (original): ", len(features))
    d = len(feature_fields)
    m = len(kis.keys())
    upper_bound = CORESET.Coreset_FMM(features, colors, k, m, d, coreset_size).compute_gamma_upper_bound()
    print(upper_bound)
    features, colors = CORESET.Coreset_FMM(features, colors, k, m, d, coreset_size).compute()
    print("Number of points (coreset): ", len(features))


    N = len(features)

    timer.split("One MWU round")

    # TODO: adult, 2 colors (sex) and all colors, K ~ 100 in total, gamma ~ 3 ([2-4]), epsilon try things
    gamma = 1.31

    # epsilon (last parameter) should be around .1-.5
    X = mult_weight_upd(gamma, N, k, features, colors, kis, .999)
    res = None
    if X is not None:
        timer.split("Randomized Rounding")
        print(X.shape)

        # do we want all points included or just the ones in S?
        S = rand_round(gamma / 2.0, X.flatten(), features, colors, kis)

        res, total = timer.stop()

        colors, chosen_colors = np.unique(colors[S], return_counts=True)

        for color, count in zip(colors, chosen_colors):
            print(f'Color: {color}    count: {count}')

        # calculate diversity
        solution = features[S]
        diversity = utils.compute_maxmin_diversity(solution)
        print(f'Solved diversity is {diversity}')

    else:
        res, total = timer.stop()
        print('MWU: N/A')

    """
    to check X is correct
        sum [0-3] <=  1 + e
        sum [5-8] <=  1 + e
        sum [8-11] <= 1 + e
    """

    print('Timings! (seconds)')
    for name, delta in res:
        print(f'{name + ":":>40} {delta}')

    """
        timer.split("LP Solve of same problem")

        # LP solve approach as well
        m = gp.Model(f"feasibility model")  # workaround for OOM error?
        m.Params.method = 2
        m.Params.SolutionLimit = 1

        variables = m.addMVar(N, name="x", vtype=GRB.CONTINUOUS)
        # create data structure
        data_struct = KDTree2.create(features)

        feasible = solve_lp(data_struct, kis, m, gamma, variables, colors, features)

        if feasible:
            vars = m.getVars()
            X = np.array(m.getAttr("X", vars))
            names = np.array(m.getAttr("VarName", vars))
            print(f'LP: {X}')
        else:
            print("LP: N/A")
    """



