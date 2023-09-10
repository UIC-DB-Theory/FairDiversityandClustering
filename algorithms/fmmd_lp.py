# lpsolve - simple lp solving based approach to the problem
import math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

import sys
import numpy as np

from tqdm import tqdm
import typing as t
import numpy.typing as npt

import time

import datastructures.KDTree2 as algo
from datastructures.WeightedTree import WeightedTree
import datastructures.BallTree as BallTree
from algorithms.rounding import rand_round
import algorithms.coreset as CORESET
import algorithms.utils as utils


def solve_lp(dataStruct, kis, m: gp.Model, gamma: np.float64, variables: npt.NDArray[gp.MVar], colors: npt.NDArray[t.AnyStr],
             features: npt.NDArray[np.float64]) -> bool:
    """
    solve_lp

    builds and solves the LP for the given gamma, returning true if the model is feasible

    variables, colors, and features should all have the same length (features can be multi-dimensional?)

    :param m: the initialized model (named, really; added for compatibility with moving pieces outside of this to reduce re-initalization)
    :param gamma: current minimum distance between points
    :param variables: np array of model variables
    :param colors: np array of "colors" of each point
    :param features: np array of data points per model
    :return:
    """
    # reset model, and remove constraints if they exist
    m.reset()
    m.remove(m.getConstrs())

    # set up parameters since reset resets it
    m.Params.SolutionLimit = 1
    m.Params.OutputFlag = 0

    N = len(features)

    """
    THIS DOES NOT WORK
    # if we are going to a big dataset, use memory reducing settings
    if N >= 10000:
        m.Params.PreSparsify = 2
        m.Params.Method = 1
        """

    # we could probably build up the color constraints once and repeatedly add them
    # this builds up the lhs of the constraints in exprs
    exprs = defaultdict()
    exprs.default_factory = lambda: gp.LinExpr()  # this shouldn't be necessary but it is!
    for i, color in tqdm(enumerate(colors), desc="Building color constraints", unit=" elems", total=N, disable=True):
        exprs[color.item()].addTerms(1.0, variables[i].item())

    # reset the default factory to make this into a normal dictionary (error if we use the wrong key)
    exprs.default_factory = None

    # we need at least ki of each color so build the final constraints here
    for key in kis.keys():
        m.addConstr(exprs[key] >= kis[key])

    # we need at most one point in the ball
    # This is slow

    # build a constraint for every point
    for v, p in tqdm(zip(variables, features), desc="Building ball constraints", unit=" elems", total=N, disable=True):
        indices = algo.get_ind_range(dataStruct, np.float_(gamma / 2.0), p)

        # add constraint that all the variables need to add up
        other_vars = variables[indices]
        count = other_vars.shape[0]

        # a bit of workaaounds to get from MVar to var here
        in_rad = gp.LinExpr([1.0] * count, other_vars.tolist())

        m.addConstr(in_rad <= 1)

    m.update()
    m.optimize()

    # if we're feasible, return true otherwise return false
    # model is passed back automatically
    if m.status == GRB.INFEASIBLE or m.status == GRB.INF_OR_UNBD:
        #print(f'Model for {gamma} is infeasible')
        return False
    elif m.status == GRB.OPTIMAL or m.status == GRB.SOLUTION_LIMIT:
        #print(f'Model for {gamma} is feasible')
        return True
    else:
        print(f'\n\n\n***ERROR: Model returned status code {m.status}***')
        print(f'Exiting')
        exit(-1)


def bin_lpsolve(features, colors, k, epsilon, a):
    # all colors made by combining values in color_fields
    color_names = np.unique(colors)

    timer = utils.Stopwatch("Normalization")

    # "normalize" features
    # Should happen before coreset construction
    means = features.mean(axis=0)
    devs  = features.std(axis=0)
    features = (features - means) / devs

    timer.split("Coreset")

    d = len(feature_fields)
    m = len(color_names)
    features, colors = CORESET.Coreset_FMM(features, colors, k, m, d, coreset_size).compute()

    N = len(features)

    timer.split("Building Kis")

    kis = utils.buildKisMap(colors, k, a)

    # keys are appended using underscores
    # TODO: group formulas => max(1, (1-a) * k n_c / n)
    # NOTE: DO THIS AFTER CORESET
    # n = number of input points in color c
    # (k * n_c / n) <- is all we need!

    timer.split("Tree Creation")

    # create data structure
    data_struct = algo.create(features)

    timer.split("High Bound Search")

    # first we need to find the high value
    print('Solving for high bound')
    high = 20  # I'm assuming it's a BIT larger than 0.01
    gamma = high * epsilon
    m = gp.Model(f"feasibility model")  # workaround for OOM error?
    m.Params.SolutionLimit = 1
    m.Params.OutputFlag = 0
    m.Params.LogFile = ""
    m.Params.LogToConsole = 0

    variables = m.addMVar(N, name="x", vtype=GRB.CONTINUOUS)

    feasible = solve_lp(data_struct, kis, m, np.float_(gamma), variables, colors, features)
    while feasible:
        high *= 2
        gamma = high * epsilon

        feasible = solve_lp(data_struct, kis, m, np.float_(gamma), variables, colors, features)

    # binary search over multiples of epsilon
    timer.split("Binary Search")
    low = 1
    assert (low < high)

    multiple = math.ceil((low + high) / 2.0)
    while low < high:
        # solve model once for current gamma
        gamma = multiple * epsilon

        feasible = solve_lp(data_struct, kis, m, np.float_(gamma), variables, colors, features)

        # if it's feasible, we have to search for larger multiples
        # if it's not, we want smaller multiples
        if feasible:
            # our high will be the first failure
            low = multiple
        else:
            high = multiple - 1

        print(low, high)
        multiple = math.ceil((low + high) / 2.0)

    gamma = multiple * epsilon

    while not solve_lp(data_struct, kis, m, np.float_(gamma), variables, colors, features):
        multiple -= 1
        gamma = multiple * epsilon

    print()
    if m.status == GRB.INFEASIBLE or m.status == GRB.INF_OR_UNBD:
        print(f'Model for {gamma} is infeasible')
        exit(-1)
    elif m.status == GRB.OPTIMAL or m.status == GRB.SOLUTION_LIMIT:
        print(f'Model for {gamma} is feasible')
    else:
        print(f'\n\n\n***ERROR: Model returned status code {m.status}***')
        print(f'Exiting')
        exit(-1)

    timer.split("Get Results")
    # get results of the LP
    vars = m.getVars()
    X = np.array(m.getAttr("X", vars))
    names = np.array(m.getAttr("VarName", vars))
    assert(len(X) == N)

    count=0
    print(f'Nonzero variables:')
    for x, n in zip(X, names):
        if x != 0:
            #print(f"{n} = {x}")
            count+=1
    print(f'Total number: {count}')
    print()

    timer.split("Randomized Rounding")

    # do we want all points included or just the ones in S?
    S = rand_round(gamma / 2, X, features, colors, kis)

    """
    print(f'Final Solution (len = {len(S)}):')
    print(S)

    print('Points:')
    res = list(zip(features[S], colors[S]))
    for r in res:
        print(r[0], r[1])
    """

    print('Solution Stats:')
    for color in kis.keys():
        print(f'{color}: {sum(colors[S] == color)}')

    print()
    tree = algo.create(features[S])
    for i in S:
        p = features[i]
        dists, indices = algo.get_ind(tree, 2, p)

        if dists[0][1] < gamma / 2.0:
            print('ERROR: invalid distance between points')
            print(dists)
            print(features[i])
            print(features[S][indices[0]][1])
            print()

    # time!
    res, total = timer.stop()
    print(f'Timings! ({total} total seconds)')
    for name, delta in res:
        print(f'{name + ":":>40} {delta}')

    # compute diversity value of solution
    solution = features[S]

    diversity = utils.compute_maxmin_diversity(solution)
    print(f'Solved diversity is {diversity}')

    return diversity, total


def epsilon_falloff(features, colors, upper_gamma, kis, epsilon, deltas=False):
    """
    starts at a high bound from the corest and repeatedly falls off by 1-epsilon
    :param features: the points in the dataset
    :param colors: corresponding color labels for each point
    :param kis: the map of points to select per color
    :param epsilon: fallof value
    :param deltas: whether to return delta between requested and selected
    :return:
    """

    N = len(features)


    timer = utils.Stopwatch("Tree Creation")

    # create data structure
    data_struct = algo.create(features)

    timer.split("Exponential falloff")

    # build model
    m = gp.Model("feasibility model")  # workaround for OOM error?
    m.Params.SolutionLimit = 1
    m.Params.OutputFlag = 0
    m.Params.LogFile = ""
    m.Params.LogToConsole = 0


    gamma = upper_gamma

    variables = m.addMVar(N, name="X", vtype=GRB.CONTINUOUS)

    # fall off until solving
    while not solve_lp(data_struct, kis, m, np.float_(gamma), variables, colors, features):
        gamma = gamma * (1 - epsilon)

    timer.split("Randomized Rounding")
    # get results of the LP
    vars = m.getVars()
    X = np.array(m.getAttr("X", vars))
    #names = np.array(m.getAttr("VarName", vars))
    assert(len(X) == N)

    # do we want all points included or just the ones in S?
    S = rand_round(gamma / 2.0, X, features, colors, kis)

    _, total_time = timer.stop()

    selected_count = len(S)

    # compute diversity value of solution
    solution = features[S]
    diversity = utils.compute_maxmin_diversity(solution)

    if deltas:
        delta_vals = utils.check_returned_kis(colors, kis, S)

        return S, diversity, delta_vals, total_time
    else:
        return S, diversity, total_time


if __name__ == '__main__':
    # setup
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

    # coreset params
    # Set the size of the coreset

    colors, features = utils.read_CSV("./datasets/ads/adult.data", allFields, color_field, '_', feature_fields)
    assert (len(colors) == len(features))

    # run some tests!
    results = []

    # first for the proper 100
    for k in range(10, 201, 5):
        # compute coreset of size
        coreset_size = 500 * k
        # all colors made by combining values in color_fields
        color_names = np.unique(colors)

        # "normalize" features
        # Should happen before coreset construction
        means = features.mean(axis=0)
        devs = features.std(axis=0)
        features = (features - means) / devs

        d = len(feature_fields)
        m = len(color_names)
        coreset = CORESET.Coreset_FMM(features, colors, k, m, d, coreset_size)
        # can't overwrite our original dataset
        core_features, core_colors = coreset.compute()

        # starting estimate
        upper_gamma = coreset.compute_gamma_upper_bound()

        # ki generation
        kis = utils.buildKisMap(colors, k, 0.1)

        # run LP
        print(f'Running LP for {k}')
        selected, div, deltas, time = epsilon_falloff(
            features=core_features,
            colors=core_colors,
            upper_gamma=upper_gamma,
            kis=kis,
            epsilon=.1,
            deltas=True,
        )
        print(f'Finished in {time}! (diversity of {div})')
        results.append((k, selected, div, deltas, time))

    # TODO: double check how many of each color compared to K map
    # TODO: run another algorithm to compare diversity

    print('\n\nFINAL RESULTS:')
    print('k\tselected\tdiversity\ttime', end='\n')
    #for color in deltas.keys():
        #print(f'{color}', end='\t')
    #print('', end='\n')

    for k, selected, div, deltas, time in results:
        print(f'{k}\t{selected}\t{div}\t{time}', end='\t')
        for delta in deltas.values():
            print(f'{delta}', end='\t')

        print('', end='\n')
