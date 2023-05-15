import math
from collections import defaultdict

import sys
import numpy as np

from tqdm import tqdm
import typing as t
import numpy.typing as npt

import KDTree2 as algo
import coreset as CORESET
import utils
from rounding import rand_round

def mult_weight_upd(gamma, features, colors, kis, epsilon):
    """
    uses the multiplicative weight update method to
    generate an integer solution for the LP
    :param gamma: the minimum distance to optimize for
    :param features: dataset's features
    :param colors: matching colors
    :param kis: the color->count mapping
    :param epsilon: allowed error value
    :return: a nx1 vector X of the solution or None if infeasible
    """



if __name__ == '__main__':
    # File fields
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

    # fields we care about for parsing
    color_field = ['race', 'sex']
    feature_fields = {'age', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt', 'education-num'}

    # variables for running LP bin-search
    # keys are appended using underscores
    kis = {
        'White_Male': 15,
        'White_Female': 35,
        'Asian-Pac-Islander_Male': 55,
        'Asian-Pac-Islander_Female': 35,
        'Amer-Indian-Eskimo_Male': 15,
        'Amer-Indian-Eskimo_Female': 35,
        'Other_Male': 15,
        'Other_Female': 35,
        'Black_Male': 15,
        'Black_Female': 35,
    }
    k = sum(kis.values())
    # binary search params
    epsilon = np.float64("0.001")

    # coreset params
    # Set the size of the coreset
    coreset_size = 10000

    # start the timer
    timer = utils.Stopwatch("Parse Data")

    colors, features = utils.read_CSV("./datasets/ads/adult.data", allFields, color_field, '_', feature_fields)
    assert (len(colors) == len(features))

    # "normalize" features
    # Should happen before coreset construction
    features = features / features.max(axis=0)

    timer.split("Coreset")

    print("Number of points (original): ", len(features))
    d = len(feature_fields)
    m = len(kis.keys())
    features, colors = CORESET.Coreset_FMM(features, colors, k, m, d, coreset_size).compute()
    print("Number of points (coreset): ", len(features))

    N = len(features)

