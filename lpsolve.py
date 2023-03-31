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


def naiveBall(dF, r, point, points: t.List[t.Any]) -> t.List[t.Any]:
    """naiveBall.

    Returns a list of all points centered at point p with radius r.

    :param dF: the distance function point -> point -> num
    :param r: the radius of the sphere to bound
    :param point: the circle's center
    :param points: a list of all points in the dataset
    :type points: t.List[t.Any]
    :rtype: t.List[t.Any]
    """
    inside = []

    for p in points:
        if dF(point, p) <= r:
            inside.append(p)

    return inside


if __name__ == '__main__':
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

    # variables for running LP bin-search
    color_field = 'sex'
    feature_fields = {'age'}
    kis = {"Male": 10, "Female": 10}

    epsilon = np.float64("0.0001")

    # import data as needed
    colors, features = read_CSV("./datasets/ads/adult.data", allFields, color_field, feature_fields)
    assert(len(colors) == len(features))
    N = len(features)


    m = gp.Model("feasability")

    # for every row, we need a variable
    variables = m.addVars(N, name="x")


    # build up the LinExpr for our constraint

    # objective function is moot
    # m.setObjective(gp.LinExpr(0), GRB.MAXIMIZE)

    # we need at least ki of each color
    exprs = defaultdict()
    exprs.default_factory = lambda: gp.LinExpr()
    for v, e in varData:
        for field, ki in kis.items():
            e_color = e[color_field]
            exprs[e_color].addTerms(1.0, v)

    print(exprs.keys())

    for key in kis.keys():
        m.addConstr(exprs[key] >= kis[key])

    m.update()

    # we need at most one point in the ball
    # This is slow

    # exprs = []

    # we need a distance function (1-d) for now
    def dist(a, b):
        _, ea = a
        _, eb = b
        return abs((ea["age"]) - (eb["age"]))

    for row in tqdm(varData):
        v, e = row
        others = naiveBall(dist, gamma / 2.0, row, varData)

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