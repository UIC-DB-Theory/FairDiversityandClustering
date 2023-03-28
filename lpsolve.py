# lpsolve - simple lp solving based approach to the problem

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

    data = readCSV("./datasets/ads/adult.data", allFields, wantedFields)

    # quick test
    print(data[0:5])
