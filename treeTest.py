import typing

from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
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
import BallTree as btree
import KDTree as kdtree
import NaiveBall as nball

def read_csv(filename: t.AnyStr, field_names: t.Sequence, color_field: t.AnyStr, feature_fields: t.Set[t.AnyStr]) -> (
        npt.NDArray[t.AnyStr], npt.NDArray[np.float64]):
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
    # elements = list()
    # count = 0
    for row in reader:
        # colors = []
        # features = []
        colors.append(row[color_field].strip())
        features.append([float(row[field]) for field in feature_fields])
        # elements.append(Elem(count, np.array(colors), np.array(features, dtype=np.float64)))
        # count += 1

    # we want these as np arrays
    colors = np.array(colors)
    features = np.array(features, dtype=np.float64)

    return colors, features

if __name__ == '__main__':
    # variables for running LP bin-search
    color_field = 'sex'
    feature_fields = {'age', 'capital-gain', 'capital-loss'}
    # feature_fields = {'age'}
    kis = {"Male": 10, "Female": 10}

    # binary search params
    epsilon = np.float64("0.0001")

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
    colors, features = read_csv("./datasets/ads/adult.data", allFields, color_field, feature_fields)
    assert (len(colors) == len(features))

    # print(elements)
    N = len(features)

    # print(colors)
    # print(features)


    # "normalize" features
    features = features / features.max(axis=0)

    print(len(features))

    # tree = KDTree(features)
    # btree = BallTree(features)
    print(f"Test point: {features[0]}")
    point = features[0]
    # dim = point.shape[0]  # this is a tuple (reasons!)
    # point = np.reshape(point, (1, dim))
    # treedist, treeind = tree.query(np.reshape(features[0],(1,-1)), k=1)
    # btreedist, btreeind = btree.query(np.reshape(features[0],(1,-1)), k=1)
    # treedist, treeind = tree.query(point, k=10)
    # btreedist, btreeind = btree.query(point, k=10)
    kd_struct = kdtree.create(features)
    kdind = kdtree.get_ind(kd_struct, np.float_(.15/2), point)
    print(f"kdtree: {kdind}")

    ball_struct = btree.create(features)
    ballt_ind = btree.get_ind(ball_struct, np.float_(.15 / 2), point)
    print(f"balltree: {ballt_ind}")

    nb_struct = nball.create(features)
    nb_ind = nball.get_ind(nb_struct, np.float_(.15 / 2), point)
    print(f"naive ball ind: {nb_ind}")
    kdTest = False
    ballTest = False

    for idx in nb_ind:
        kdTest = True
        if idx not in kdind:
            kdTest = False
        if not kdTest:
            break
    print(kdTest)

    for idx in nb_ind:
        ballTest = True
        if idx not in ballt_ind:
            ballTest = False
        if not ballTest:
            break
    print(ballTest)

    # treestats = tree.get_tree_stats()
    # print(f"KDTree stats:\ntrims: {treestats[0]}; leaves: {treestats[1]}; splits: {treestats[2]}")
    # print(f"KDTreeDist: {treedist.flatten()}")
    # print(f"KDTreeIndex: {treeind.flatten()}")
    # print()
    # btreestats = btree.get_tree_stats()
    # print(f"BallTree stats:\ntrims: {btreestats[0]}; leaves: {btreestats[1]}; splits: {btreestats[2]}")
    # print(f"BallTreeDist: {btreedist.flatten().nonzero()}")
    # print(f"BallTreeIndex:{btreeind.flatten()}")
    # raddist, radind = tree.query_radius(features, .1)
    # print(raddist)
    # print(radind)
    #
    # dist,ind = tree.query(features, k=2)
    # print(ind)
    # print(dist)
