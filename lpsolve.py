# lpsolve - simple lp solving based approach to the problem

from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

import typing as t

def readCSV(filename : t.AnyStr, fieldNames: t.Sequence, wantedFields: t.Set) -> t.List[t.Dict[t.AnyStr, t.Any]]:
    """readCSV.

    Reads in a CSV datafile as a list of dictionaries.
    The datafile should have no header row

    :param filename: the file to read in
    :type filename: t.AnyStr
    :param fieldNames: the headers of the CSV file, in order
    :type fieldNames: t.Sequence
    :param wantedFields: a subset of fieldnames which will be returned (the rest are filtered)
    :type wantedFields: t.Set
    """
    from csv import DictReader

    # read csv as a dict with given keys
    reader = DictReader(open(filename), fieldnames=fieldNames)

    out = []
    for row in reader:
        # only return the keys we care about from the csv
        out.append({k: v for k, v in row.items() if k in wantedFields})

    return out

def naiveBall(dF, r, point, points : t.List[t.Any]) -> t.List[t.Any]:
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
            "fnlwgt", # what on earth is this one?
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
    wantedFields = {"age", "sex"}

    # variables for running LP
    k = 20
    color_field = 'sex'
    kis = {"Male": 10, "Female": 10}
    gamma = 10

    data = readCSV("./datasets/ads/adult.data", allFields, wantedFields)

    # read to int
    for elem in data:
        elem['age'] = int(elem['age'])

    m = gp.Model("feasability")

    # for every row, we need a variable
    varData = [(m.addVar(name=f"elem #{c}"), elem) for c, elem in enumerate(data)]

    # varData = varData[0:250]

    # updates the model to reflect new vars
    # probably not needed but can't hurt?
    # m.update()

    # we need a distance function (1-d) for now
    def dist(a, b):
        return abs((a["age"]) - (b["age"]))

    # build up the LinExpr for our constraint
    # as a side note, MaxMin kind of sucks to implement

    # every pair of points we pick is greater than the minimum distance
    mind = m.addVar(name="Min Dist")
    for i in range(len(varData)):
        for j in range(i):
            vi, ei = varData[i]
            vj, ej = varData[j]
            m.addConstr(vi * vj * dist(ei, ej) >= mind)

    # then we maximize the minimum distance
    m.setObjective(mind, GRB.MAXIMIZE)

    # we need at least ki of each color
    exprs = defaultdict()
    exprs.default_factory=lambda: gp.LinExpr()
    for v, e in varData:
        for field, ki in kis.items():
            e_color = e[color_field]
            exprs[e_color].addTerms(1.0, v)    
    
    for key in kis.keys():
        m.addConstr(exprs[key] >= kis[key])


    # we need at most one point in the ball
    # for v, e in varData:
        # others = naiveBall(dist, gamma / 2.0, e, data)
        # print(others)
