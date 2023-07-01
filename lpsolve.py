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

import KDTree2 as algo
import coreset as CORESET
import utils
from rounding import rand_round

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
    # maybe this makes it faster?
    m.tune()

    # reset model, removing constraints if they exist
    m.reset()
    m.remove(m.getConstrs())

    N = len(features)

    sys.stdout.flush()
    print(f'testing feasability of gamma={gamma}', flush=True)
    # we could probably build up the color constraints once and repeatedly add them
    # this builds up the lhs of the constraints in exprs
    exprs = defaultdict()
    exprs.default_factory = lambda: gp.LinExpr()  # this shouldn't be necessary but it is!
    for i, color in tqdm(enumerate(colors), desc="Building color constraints", unit=" elems", total=N):
        exprs[color.item()].addTerms(1.0, variables[i].item())
    exprs.default_factory = None

    # we need at least ki of each color so build the final constraints here
    for key in kis.keys():
        m.addConstr(exprs[key] >= kis[key])

    # we need at most one point in the ball
    # This is slow

    # build a constraint for every point
    for v, p in tqdm(zip(variables, features), desc="Building ball constraints", unit=" elems", total=N):
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
    # TODO: ensure else case is ONLY if model is feasible (error otherwise)
    if m.status == GRB.INFEASIBLE or m.status == GRB.INF_OR_UNBD:
        print(f'Model for {gamma} is infeasible')
        return False
    elif m.status == GRB.OPTIMAL or m.status == GRB.SOLUTION_LIMIT:
        print(f'Model for {gamma} is feasible')
        return True
    else:
        print(f'\n\n\n***ERROR: Model returned status code {m.status}***')
        print(f'Exiting')
        exit(-1)




def bin_lpsolve(k):
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

    # binary search params

    # the minimum quantization of our search space
    epsilon = np.float64("0.01")

    # coreset params
    # Set the size of the coreset
    coreset_size = 10000

    # other things for gurobi
    method = 2  # model method of solving

    # start the timer
    timer = utils.Stopwatch("Parse Data")

    colors, features = utils.read_CSV("./datasets/ads/adult.data", allFields, color_field, '_', feature_fields)
    assert (len(colors) == len(features))

    # all colors made by combining values in color_fields
    color_names = np.unique(colors)

    timer.split("Coreset")

    # "normalize" features
    # Should happen before coreset construction
    means = features.mean(axis=0)
    devs  = features.std(axis=0)
    features = (features - means) / devs

    timer.split("Normalization")


    print("[LPSOLVE] Number of points (original): ", len(features))
    d = len(feature_fields)
    m = len(color_names)
    features, colors = CORESET.Coreset_FMM(features, colors, k, m, d, coreset_size).compute()
    print("[LPSOLVE] Number of points (coreset): ", len(features))

    N = len(features)

    timer.split("Building Kis")

    kis = utils.buildKisMap(colors, k, 0.2)

    # keys are appended using underscores
    # TODO: group formulas => max(1, (1-a) * k n_c / n)
    # NOTE: DO THIS AFTER CORESET
    # a = 0.2
    # n = number of input points in color c
    # (k * n_c / n) <- is all we need!

    timer.split("Tree Creation")

    # create data structure
    data_struct = algo.create(features)

    timer.split("High Bound Search")

    # first we need to find the high value
    print('Solving for high bound')
    high = 50  # I'm assuming it's a BIT larger than 0.01
    gamma = high * epsilon
    m = gp.Model(f"feasibility model")  # workaround for OOM error?
    m.Params.method = method
    m.Params.SolutionLimit = 1

    variables = m.addMVar(N, name="x", vtype=GRB.CONTINUOUS)

    feasible = solve_lp(data_struct, kis, m, np.float_(gamma), variables, colors, features)
    while feasible:
        high *= 2
        gamma = high * epsilon

        feasible = solve_lp(data_struct, kis, m, np.float_(gamma), variables, colors, features)

    print(f'High bound is {high}; binary search')

    # binary search over multiples of epsilon
    timer.split("Binary Search")
    low = 1
    assert (low < high)

    multiple = math.ceil((low + high) / 2.0)
    while low < high:
        # solve model once for current gamma
        print(f'Current multiple is {multiple}')
        sys.stdout.flush()

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

    print(f'Final test for multiple {multiple} (gamma = {gamma}')

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

if __name__ == '__main__':
    # run some tests!
    for i in range(10, 101):
