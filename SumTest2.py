
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


if __name__ == '__main__':
    colors, weights, features = preamble()
    N = len(features)
    tree = BallTree(features, metric='minkowski', sample_weight=weights)
    tree2 = BallTree(features, metric='minkowski', sample_weight=weights)
    countTimes = list()
    sumTimes = list()
    # print(tree.get_tree_stats())
    # print(tree2.get_tree_stats())
    for p in features:
        dim = p.shape[0]  # this is a tuple (reasons!)
        point_reshaped = np.reshape(p, (1, dim))
        start_time = time.time_ns()
        res = tree.query_radius(point_reshaped, np.float_(.15 / 2), count_only=True).flatten()[0]
        timeD = time.time_ns()-start_time
        if timeD > 0:
            countTimes.append(timeD)
    # print(tree.get_tree_stats())
    # print(tree2.get_tree_stats())

    for p in features:
        dim = p.shape[0]  # this is a tuple (reasons!)
        point_reshaped = np.reshape(p, (1, dim))
        start_time = time.time_ns()
        res = np.sum(tree2.query_radius(point_reshaped, np.float_(.15 / 2)).flatten()[0])
        timeD = time.time_ns() - start_time
        if timeD > 0:
            sumTimes.append(timeD)


    endStateCount = tree.__getstate__()
    print(f"leaf size: {endStateCount[4]}")
    print(f"nodes: {endStateCount[6]}")
    print(f"leaves: {endStateCount[8]}")
    print(f"splits: {endStateCount[9]}")

    # print(tree.__getstate__())
    # print(tree2.__getstate__())

    countTest = np.average(countTimes)
    sumTest = np.average(sumTimes)

    print(f"count: {countTest/1000000} ms")
    print(f"sum: {sumTest/1000000} ms")

