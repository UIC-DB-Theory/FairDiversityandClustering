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

import BallTree as algo
import utils

def solve_lp(dataStruct, m: gp.Model, gamma: np.float64, variables: npt.NDArray[gp.MVar], colors: npt.NDArray[t.AnyStr],
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
    # m.tune()

    # reset model, removing constraints if they exist
    m.reset()
    m.remove(m.getConstrs())

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
        indices = algo.get_ind(dataStruct, np.float_(gamma / 2.0), p)

        # add constraint that all the variables need to add up
        other_vars = variables[indices]
        count = other_vars.shape[0]

        # a bit of workarounds to get from MVar to var here
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
    elif m.status == GRB.OPTIMAL:
        print(f'Model for {gamma} is feasible')
        return True
    else:
        print(f'\n\n\n***ERROR: Model returned status code {m.status}***')
        print(f'Exiting')
        exit(-1)

def construct_coreset(features, colors):
    """
    construct_coreset

    constructs the coreset for FMMD.

    :param features: np array of data points per model
    :param colors: np array of "colors" of each point
    """
    import coreset as CORESET
    l = len(feature_fields) # doubling dimension = d 
    e_coreset = 3 
    coreset_constructor = CORESET.Coreset_FMM(features, colors, k, e_coreset, l)
    features, colors = coreset_constructor.compute()
    return features, colors

if __name__ == '__main__':
    # variables for running LP bin-search
    color_field = 'sex'
    feature_fields = {'age', 'capital-gain', 'capital-loss'}
    # feature_fields = {'age'}
    kis = {"Male": 10, "Female": 10}
    k = 20
    # binary search params
    epsilon = np.float64("0.001")

    # other things for gurobi
    method = 2  # model method of solving

    # import data from file
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
    colors, features = utils.read_CSV("./datasets/ads/adult.data", allFields, color_field, feature_fields)
    assert (len(colors) == len(features))

    # "normalize" features
    # Should happen before coreset construction
    features = features / features.max(axis=0)


    print(f'Size of data = {len(features)}')
    features, colors = construct_coreset(features, colors)
    print(f'Coreset size = {len(features)}')

    N = len(features)

    # create data structure
    data_struct = algo.create(features)

    # first we need to find the high value
    print('Solving for high bound')
    high = 100  # I'm assuming it's a BIT larger than 0.0001
    gamma = high * epsilon
    m = gp.Model(f"feasibility model")  # workaround for OOM error?
    m.Params.method = method
    m.Params.SolutionLimit = 1

    variables = m.addMVar(N, name="x")

    feasible = solve_lp(data_struct, m, np.float_(gamma), variables, colors, features)
    while feasible:
        high *= 2
        gamma = high * epsilon

        feasible = solve_lp(data_struct, m, np.float_(gamma), variables, colors, features)

    print(f'High bound is {high}; binary search')

    # binary search over multiples of epsilon
    low = 1
    assert (low < high)

    multiple = math.ceil((low + high) / 2.0)
    while low < high:
        # solve model once for current gamma
        print(f'Current multiple is {multiple}')
        sys.stdout.flush()

        gamma = multiple * epsilon

        feasible = solve_lp(data_struct, m, np.float_(gamma), variables, colors, features)

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

    while not solve_lp(data_struct, m, np.float_(gamma), variables, colors, features):
        multiple -= 1
        gamma = multiple * epsilon

    print()
    if m.status == GRB.INFEASIBLE or m.status == GRB.INF_OR_UNBD:
        print(f'Model for {gamma} is infeasible')
        exit(-1)
    elif m.status == GRB.OPTIMAL:
        print(f'Model for {gamma} is feasible')
    else:
        print(f'\n\n\n***ERROR: Model returned status code {m.status}***')
        print(f'Exiting')
        exit(-1)

    # get results of the LP
    vars = m.getVars()
    X = m.getAttr("X", vars)
    print(X)