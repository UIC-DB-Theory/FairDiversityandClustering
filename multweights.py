import math
from collections import defaultdict

import sys
import numpy as np

from tqdm import tqdm, trange
import typing as t
import numpy.typing as npt

import KDTree2 as algo
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
    # TODO: update epsilon if needed
    h = np.full((N, 1), 1.0 / N) # weights
    X = np.zeros((N, 1))         # Output

    T = (8 * k) / (math.pow(epsilon, 2)) * math.log(N, math.e) # iterations
    # for now, we can recreate the structure in advance
    struct = algo.create(features)

    # NOTE: should this be <= or was it 1 indexed?
    print(T)
    for t in trange(math.ceil(T), desc='MWU Loop'):

        S = np.empty_like(features)  # points we select this round
        W = 0                        # current weight sum

        # weights to every point
        w_sums = algo.get_weight_ranges(struct, h, gamma / 2.0)

        # compute minimums per color
        for color in kis.keys():
            # need this to reverse things
            color_sums_ind = (color == colors).nonzero()[0] # tuple for reasons

            # get minimum points as indices
            color_sums = w_sums[color_sums_ind]
            mins = np.argpartition(color_sums, kis[color])[:kis[color]]
            min_indecies = color_sums_ind[mins]

            # add to X's
            X[min_indecies] += 1
            # add points to set we've seen
            S = np.append(S, features[min_indecies], axis=0)
            # add additional weight
            W += np.sum(h[min_indecies])

        if W > 1:
            return None


if __name__ == '__main__':
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
    # binary search params
    epsilon = np.float64("0.001")

    # coreset params
    # Set the size of the coreset
    coreset_size = 10000

    # start the timer
    timer = utils.Stopwatch("Parse Data")

    colors, features = utils.read_CSV("./datasets/ads/adult.data", allFields, color_field, '_', feature_fields)
    assert (len(colors) == len(features))

    # "normalize" features
    # Should happen before coreset construction
    features = features / features.max(axis=0)

    timer.split("Coreset")

    print("Number of points (original): ", len(features))
    d = len(feature_fields)
    m = len(kis.keys())
    features, colors = CORESET.Coreset_FMM(features, colors, k, m, d, coreset_size).compute()
    print("Number of points (coreset): ", len(features))

    N = len(features)

    timer.split("One MWU round")

    # first we need to find the high value
    print('Solving for high bound')
    high = 1
    gamma = high * epsilon

    mult_weight_upd(gamma, N, k, features, colors, kis, 0.5)

    res = timer.stop()
    print('Timings! (seconds)')
    for name, delta in res:
        print(f'{name + ":":>40} {delta}')
