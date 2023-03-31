# lpsolve - simple lp solving based approach to the problem

from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

from tqdm import tqdm

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
        out.append({k.strip(): v.strip() for k, v in row.items() if k in wantedFields})

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
    color_field = 'sex'
    kis = {"Male": 10, "Female": 10}
    gamma = 10

    data = readCSV("./datasets/ads/adult.data", allFields, wantedFields)

    # read to int
    for elem in data:
        elem['age'] = int(elem['age'])

    m = gp.Model("feasability")

    # for every row, we need a variable
    varData = [(m.addVar(name=f"elem #{c}", lb=0.0, ub=1.0), elem) for c, elem in enumerate(data)]

    # varData = varData[0:250]

    # updates the model to reflect new vars
    # probably not needed but can't hurt?
    m.update()

    # we need a distance function (1-d) for now
    def dist(a, b):
        _, ea = a
        _, eb = b
        return abs((ea["age"]) - (eb["age"]))

    # build up the LinExpr for our constraint

    # objective function is moot
    # m.setObjective(gp.LinExpr(0), GRB.MAXIMIZE)

    # we need at least ki of each color
    exprs = defaultdict()
    exprs.default_factory=lambda: gp.LinExpr()
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