import time

import numpy as np

# import BallTree as btree
# import KDTree as kdtree
# import KDTree2 as kdtree2
# import NaiveBall as nball
from scipy.spatial import KDTree
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

    # start_time = time.time_ns()
    # tree = kdtree.create(features)
    # sklearnkdCreationTime = time.time_ns() - start_time
    #
    # start_time = time.time_ns()
    # for p in features:
    #     kdind = kdtree.get_ind(tree, np.float_(.15 / 2), p)
    # sklearnkdQTime = time.time_ns() - start_time

    start_time = time.time_ns()
    tree = KDTree(features)
    scipykdCreationTime = time.time_ns() - start_time

    start_time = time.time_ns()
    dim = features[0].shape[0]
    tree1 = KDTree(np.reshape(features[0],(1,dim)))
    tree1kdCreationTime = time.time_ns() - start_time

    start_time = time.time_ns()
    for p in features:
        dim = p.shape[0]  # this is a tuple (reasons!)
        point_reshaped = KDTree(np.reshape(p, (1, dim)))
        test = tree.count_neighbors(point_reshaped, np.float_(.15/2), weights=(weights,None))

    scipykdQTime = time.time_ns() - start_time

    start_time = time.time_ns()
    for p in features:
        # dim = p.shape[0]  # this is a tuple (reasons!)
        # point_reshaped = np.reshape(p, (1, dim))
        ind = tree.query_ball_point(p, np.float_(.15/2))
        np.sum(weights[ind])


    scipykdINDime = time.time_ns() - start_time
    start_time = time.time_ns()
    ind = tree.query_ball_tree(tree, np.float_(.15/2))
    new_ind = list()
    for i in ind:
        x = np.sum(weights[i])
    scipykdalltime = time.time_ns() - start_time

    print("Creation Times using Adult data with all continuous features:")
    print(f"Scipy KD Tree: {scipykdCreationTime / 1000000} ms")
    print()
    print("Total Query Times running distance query over each point:")
    print(f"Scipy KD Tree: {scipykdQTime / 1000000} ms")
    print(f"Creation Times for creating one node tree")
    print(f"Scipy one node: {tree1kdCreationTime / 1000000} ms")
    print()
    print("Total Query Times running distance query over each point:")
    print(f"Scipy KD Tree: {scipykdINDime / 1000000} ms")
    print()
    print("Total Query Times running distance query over each point:")
    print(f"Scipy KD Tree: {scipykdalltime / 1000000} ms")