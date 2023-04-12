import typing

from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
import math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

import sys

import numpy as np

from tqdm import tqdm
import typing as t
import numpy.typing as npt
import BallTree as btree
import KDTree as kdtree
import NaiveBall as nball
import KDTree2 as kdtree2
import utils

def preamble():
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
    colors, features = utils.read_CSV("./datasets/ads/adult.data", allFields, color_field, feature_fields)
    assert (len(colors) == len(features))
    return colors, features

if __name__ == '__main__':
    colors, features = preamble()
    N = len(features)

    # "normalize" features
    features = features / features.max(axis=0)

    tree = kdtree.create(features)
    for p in features:
        kdind = kdtree.get_ind(tree, np.float_(.15 / 2), p)

    tree = kdtree2.create(features)
    for p in features:
        kdind = kdtree2.get_ind(tree, np.float_(.15 / 2), p)

    tree = btree.create(features)
    for p in features:
        kdind = btree.get_ind(tree, np.float_(.15 / 2), p)

    tree = nball.create(features)
    for p in features:
        kdind = nball.get_ind(tree, np.float_(.15 / 2), p)
