import time

import numpy as np

import BallTree as btree
import KDTree as kdtree
import KDTree2 as kdtree2
import NaiveBall as nball
import utils


def preamble():
    # variables for running LP bin-search
    color_field = 'sex'
    feature_fields = {'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'}
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
    colors, features = utils.read_CSV("../datasets/ads/adult.data", allFields, color_field, feature_fields)
    assert (len(colors) == len(features))

    # "normalize" features
    features = features / features.max(axis=0)

    weights = np.array([r for r in np.random.rand(len(features))])

    return colors, weights, features


if __name__ == '__main__':
    colors, weights, features = preamble()
    N = len(features)

    start_time = time.time_ns()
    tree = kdtree.create(features)
    sklearnkdCreationTime = time.time_ns() - start_time

    start_time = time.time_ns()
    for p in features:
        kdind = kdtree.get_ind(tree, np.float_(.15 / 2), p)
    sklearnkdQTime = time.time_ns() - start_time

    start_time = time.time_ns()
    tree = kdtree2.create(features)
    scipykdCreationTime = time.time_ns() - start_time

    start_time = time.time_ns()
    for p in features:
        kdind = kdtree2.get_ind(tree, np.float_(.15 / 2), p)
    scipykdQTime = time.time_ns() - start_time

    start_time = time.time_ns()
    tree = btree.create(features)
    ballCreationTime = time.time_ns() - start_time

    start_time = time.time_ns()
    for p in features:
        kdind = btree.get_ind(tree, np.float_(.15 / 2), p)
    ballQTime = time.time_ns() - start_time

    start_time = time.time_ns()
    tree = nball.create(features)
    naiveCreationTime = time.time_ns() - start_time

    start_time = time.time_ns()
    for p in features:
        kdind = nball.get_ind(tree, np.float_(.15 / 2), p)
    naiveQTime = time.time_ns() - start_time

    print("Creation Times using Adult data with all continuous features:")
    print(f"Sklearn KD Tree: {sklearnkdCreationTime / 1000000} ms")
    print(f"Sklearn Ball Tree: {ballCreationTime / 1000000} ms")
    print(f"Scipy KD Tree: {scipykdCreationTime / 1000000} ms")
    print(f"Naive Ball: {naiveCreationTime / 1000000} ms")
    print()
    print("Total Query Times running distance query over each point:")
    print(f"Sklearn KD Tree: {sklearnkdQTime / 1000000} ms")
    print(f"Sklearn Ball Tree: {ballQTime / 1000000} ms")
    print(f"Scipy KD Tree: {scipykdQTime / 1000000} ms")
    print(f"Naive Ball: {naiveQTime / 1000000} ms")
