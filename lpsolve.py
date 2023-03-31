# lpsolve - simple lp solving based approach to the problem

from collections import defaultdict
import gurobipy as gp
import numpy
from gurobipy import GRB

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
        colors.append([row[color_field].strip()])  # both are nested to ensure we get matrices instead of vectors
        features.append([float(row[field]) for field in feature_fields])

    # we want these as np arrays
    colors = np.array(colors)
    features = np.array(features, dtype=numpy.float64)

    return colors, features


def naiveBall(dF, points : npt.NDArray[np.float64], r : np.float64, point) -> npt.NDArray[int]:
    """naiveBall.

    Returns the indices of points inside the ball

    :param dF: the distance function point -> point -> num
    :param points: a list of all points in the dataset
    :type points: t.List[t.Any]
    :param r: the radius of the sphere to bound
    :param point: the circle's center
    :rtype: t.List[t.Any]
    """
    from scipy.spatial.distance import cdist

    dists = cdist(points, np.reshape(point, (1,1)))
    dists = dists.flatten()

    indices = (dists <= r).nonzero()[0]  # this is a tuple (reasons!)

    return indices


if __name__ == '__main__':
    # variables for running LP bin-search
    color_field = 'sex'
    feature_fields = {'age'}
    kis = {"Male": 10, "Female": 10}

    # binary search params
    epsilon = np.float64("0.0001")
    multiple = 100000

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
        "captial-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "yearly-income",
    ]
    colors, features = read_CSV("./datasets/ads/adult.data", allFields, color_field, feature_fields)
    assert(len(colors) == len(features))
    N = len(features)

    # no objective model
    m = gp.Model(f"feasability (gamma = ")

    # for every row, we need a variable
    # this returns a gp "tuple list"
    variables = m.addVars(N, name="x")

    # we can build up the color constraints once and repeatedly add them
    # this builds up the lhs of the constraints in exprs
    exprs = defaultdict()
    exprs.default_factory = lambda: gp.LinExpr()  # this shouldn't be necessary but it is!
    for i, color in tqdm(enumerate(colors), desc="Building color constraints", unit=" elements"):
        exprs[color.item()].addTerms(1.0, variables[i])

    # we need at least ki of each color so build the final constraints here
    for key in kis.keys():
        m.addConstr(exprs[key] >= kis[key])

    m.update()

    # get our gamma value
    gamma = epsilon * multiple

    # we need at most one point in the ball
    # This is slow

    # we need a distance function (1-d) for now
    def dist(a, b):
        _, ea = a
        _, eb = b
        return abs((ea["age"]) - (eb["age"]))

    # build a constraint for every point
    for v, p in tqdm(zip(variables, features)):

        indices = naiveBall(dist, features, gamma / 2.0, p)

        print(indices)
        exit(0)

        # add constraint that all the variables need to add up
        other_vars = [v for (v, _) in others]

        in_rad = gp.LinExpr([1.0] * len(other_vars), other_vars)

        m.addConstr(in_rad <= 1)

    m.update()

    m.optimize()

    if m.status == GRB.INFEASIBLE:
        # let's find out what's wrong
        iis = m.computeIIS()
        m.write('fail.ilp')

        m.feasRelaxS(1, False, False, True)
        m.optimize()