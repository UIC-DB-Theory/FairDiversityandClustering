
from sklearn.neighbors import BallTree

import time
import numpy as np
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

    # "normalize" features
    features = features / features.max(axis=0)

    weights = [int(10000 * r ** 2) for r in np.random.rand(len(feature_fields))]

    return colors, weights, features

def balltreeCount(tree, point):
    dim = point.shape[0]  # this is a tuple (reasons!)
    point_reshaped = np.reshape(point, (1, dim))
    res = tree.query_radius(point_reshaped, np.float_(.15 / 2), count_only=True)
    return res,


def balltreeSum(tree, point):
    dim = point.shape[0]  # this is a tuple (reasons!)
    point_reshaped = np.reshape(point, (1, dim))
    idxs = tree.query_radius(point_reshaped, np.float_(.15 / 2)).flatten()[0]
    return np.sum(idxs)


if __name__ == '__main__':
    colors, weights, features = preamble()
    N = len(features)
    tree = BallTree(features, metric='minkowski', sample_weight=weights)

    countTest = np.array([balltreeCount(tree, point) for point in features])
    sumTest = np.array([balltreeSum(tree,point)for point in features])

    print(f"count: {countTest}")
    print(f"sum: {sumTest}")

