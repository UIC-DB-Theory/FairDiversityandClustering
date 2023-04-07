# lpsolve - simple lp solving based approach to the problem
import math
from collections import defaultdict
import gurobipy as gp
import numpy
from gurobipy import GRB

import sys

import numpy as np

from tqdm import tqdm
import typing as t
import numpy.typing as npt


def read_CSV(filename: t.AnyStr, field_names: t.Sequence, color_field: t.AnyStr, feature_fields: t.Set[t.AnyStr]) -> (npt.NDArray[t.AnyStr], npt.NDArray[numpy.float64]):
    """read_CSV.

    Reads in a CSV datafile as a list of dictionaries.
    The datafile should have no header row

    Returns a tuple of the colors of elements and their features

    :param filename: the file to read in
    :type filename: t.AnyStr
    :param field_names: the headers of the CSV file, in order
    :type field_names: t.Sequence
    :param color_field: the field containing the object color
    :type color_field: t.AnyStr
    :param feature_fields: the fields which are numerical data values for the point
    :type feature_fields: t.Set[t.AnyStr]
    """
    from csv import DictReader

    # read csv as a dict with given keys
    reader = DictReader(open(filename), fieldnames=field_names)

    # return the requested features and colors
    colors = []
    features = []

    for row in reader:
        colors.append(row[color_field].strip())
        features.append([float(row[field]) for field in feature_fields])

    # we want these as np arrays
    colors = np.array(colors)
    features = np.array(features, dtype=numpy.float64)

    return colors, features


def naive_ball(points : npt.NDArray[np.float64], r : np.float64, point) -> npt.NDArray[int]:
    """naive_ball.

    Returns the indices of points inside the ball

    :param dF: the distance function point -> point -> num
    :param points: a list of all points in the dataset
    :type points: t.List[t.Any]
    :param r: the radius of the sphere to bound
    :param point: the circle's center
    :rtype: t.List[t.Any]
    """
    from scipy.spatial.distance import cdist


    # we need to reshape the point vector into a row vector
    dim = point.shape[0] # this is a tuple (reasons!)
    point = np.reshape(point, (1, dim))

    # comes out as a Nx1 matrix
    dists = cdist(points, point)

    return (dists.flatten() <= r).nonzero()[0] # Also a tuple!


def solve_lp(m : gp.Model, gamma : np.float64, variables : npt.NDArray[gp.MVar], colors : npt.NDArray[t.AnyStr], features : npt.NDArray[np.float64]) -> bool:
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
    for p in tqdm(features, desc="Building ball constraints", unit=" elems", total=N):

        indices = naive_ball(features, gamma / 2.0, p)

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


if __name__ == '__main__':
    # variables for running LP bin-search
    color_field = 'sex'
    feature_fields = {'age', 'capital-gain', 'capital-loss'}
    # feature_fields = {'age'}
    kis = {"Male": 10, "Female": 10}

    # binary search params
    epsilon = np.float64("0.0001")

    # other things for gurobi
    method = 2 # model method of solving

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
    colors, features = read_CSV("./datasets/ads/adult.data", allFields, color_field, feature_fields)
    assert(len(colors) == len(features))

    # truncate for testing
    limit = 10000
    colors = colors[0:limit]
    features = features[0:limit]

    N = len(features)

    # "normalize" features
    features = features / features.max(axis=0)

    # first we need to find the high value
    print('Solving for high bound')
    high = 1500 # I'm assuming it's a BIT larger than 0.0001
    gamma = high * epsilon
    m = gp.Model(f"feasability model") # workaround for OOM error?
    m.Params.method = method
    m.Params.SolutionLimit = 1

    variables = m.addMVar(N, name="x")

    feasible = solve_lp(m, gamma, variables, colors, features)
    while feasible:
        high *= 2
        gamma = high * epsilon

        # reset model, removing constraints
        m.reset()
        m.remove(m.getConstrs())

        feasible = solve_lp(m, gamma, variables, colors, features)


    print(f'High bound is {high}; binary search')

    # binary search over multiples of epsilon
    low = 1
    assert(low < high)

    multiple = math.ceil((low + high) / 2.0)
    while low < high:
        # solve model once for current gamma
        print(f'Current multiple is {multiple}')
        sys.stdout.flush()

        gamma = multiple * epsilon

        # reset model, removing constraints
        m.reset()
        m.remove(m.getConstrs())

        feasible = solve_lp(m, gamma, variables, colors, features)

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

    feasible = solve_lp(m, gamma, variables, colors, features)

    print()
    if m.status == GRB.INFEASIBLE or m.status == GRB.INF_OR_UNBD:
        print(f'Model for {gamma} is infeasible')
    elif m.status == GRB.OPTIMAL:
        print(f'Model for {gamma} is feasible')
    else:
        print(f'\n\n\n***ERROR: Model returned status code {m.status}***')
        print(f'Exiting')
        exit(-1)