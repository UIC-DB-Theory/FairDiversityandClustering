import csv
import random
import numpy as np
import algorithms_offline as algo
import alg
import utils
import utils2
import itertools
import algorithms_streaming as algs

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
    feature_fields = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt', 'education-num']

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
    c = len(kis)
    k = sum(kis.values())
    # binary search params
    epsilon = np.float64("0.001")

    dataset = "../datasets/ads/adult.data"
    colors, features = utils2.read_CSV(dataset, allFields, color_field, '_', feature_fields)
    assert (len(colors) == len(features))

    N = len(features)

    print(f'***********Parameters***********')
    print(f'Dataset: {dataset}')
    print(f'Result set size(k): {k}')
    print(f'EPS(e): {epsilon}')
    print(f'Colors/Grouped By: {color_field}')
    print(f'Num colors(c): {c}')
    print(f'\t\tconstraint\t\tcolor')
    j = 1
    for i in kis:
        print(f'\t{j}\t{kis[i]}\t\t{i}')
        j = j + 1
    print(f'********************************')
    

    # read the Adult dataset grouped by sex+race (c=10)
    elements = []
    with open("./data/adult_small.csv", "r") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in csvreader:
            features = []
            for i in range(4, len(row)):
                features.append(float(row[i]))
            elem = utils.Elem(int(row[0]), int(row[3]), features)
            elements.append(elem)


    # Adjust problem definition to work for scalable fmmd ILP
    # For the adult dataset, grouped by age and sex

    # As done in the paper we will pre-process the data
    # by normalizing to have zero mean and unit standard deviation.
    features = np.array(features)
    features_normalized = features.copy()
    mean = np.mean(features_normalized, axis=0)
    std = np.std(features_normalized, axis=0)
    features_normalized = features_normalized - mean
    features_normalized = features_normalized/std
    features = features.tolist()
    features_normalized = features_normalized.tolist()

    # Adjust the elements to work with fmmd ILP
    color_number_map = [
        'White_Male',
        'White_Female',
        'Asian-Pac-Islander_Male',
        'Asian-Pac-Islander_Female',
        'Amer-Indian-Eskimo_Male',
        'Amer-Indian-Eskimo_Female',
        'Other_Male',
        'Other_Female',
        'Black_Male',
        'Black_Female'
    ]
    elements_normalized = []
    elements = []
    for i in range(0, len(features_normalized)):
        elem_normalized = utils.Elem(i, color_number_map.index(colors[i]), features_normalized[i])
        elem = utils.Elem(i, color_number_map.index(colors[i]), features[i])
        elements.append(elem)
        elements_normalized.append(elem_normalized)

    # Adjust the constraints per color to as a range
    # +- offset for the values of kis
    c_offset = 5
    constr = []
    for color in color_number_map:
        range_constr = [kis[color] - c_offset, kis[color] + c_offset]
        constr.append(range_constr)
    


    
    # alg.scalable_fmmd_ILP(V=features, k=k, EPS=epsilon, C=c,constr=constr[k] ,dist=utils.euclidean_dist)