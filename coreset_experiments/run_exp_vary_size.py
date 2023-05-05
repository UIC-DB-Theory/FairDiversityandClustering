#Coreset size experiment
# lpsolve - simple lp solving based approach to the problem
import sys
import os
import argparse

# Adding parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import coreset as CORESET
import utils

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="data file")
    args = parser.parse_args()
    return args

def main(args):
    print("****CORESET EXPERIMENTS****")
    print("file: ", args.file)
    
    if args.file and not os.path.isfile(args.file):
        print('File not found.')
        print('for help run: python3 run_exp_vary_size.py -h')
        exit(-1)
    
    # variables for running LP bin-search
    color_field = 'sex'
    feature_fields = ['age', 'capital-gain', 'capital-loss']
    # feature_fields = {'age'}
    kis = {"Male": 10, "Female": 10}
    k = 20

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
    # Should happen before coreset construction
    features = features / features.max(axis=0)
    
    d = len(feature_fields)
    m = len(kis.keys())
    # Set the size of the coreset
    coreset_size = 5000
    coreset_constructor = CORESET.Coreset_FMM(features, colors, k, m, d, coreset_size)
    features, colors = coreset_constructor.compute()
    
if __name__ == "__main__":
    main(parseArgs())