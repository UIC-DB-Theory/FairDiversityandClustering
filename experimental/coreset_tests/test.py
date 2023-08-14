#Coreset size experiment
# lpsolve - simple lp solving based approach to the problem
import sys
import os
import argparse
import time
import csv
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Any, Callable, List

# Adding parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import coreset as CORESET
import utils


def getNumColors(colors):
    color_set = set(colors)
    return len(color_set)

def getDimension(features):
    return len(features[0])

def main():
    print("****CORESET EXPERIMENTS: CENSUS FULL****")

    print("\tgrouped by age (m=7)")
    # read the Census dataset grouped by age (m=7)
    features = []
    colors = []
    with open(f'./datasets/census.csv', "r") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in csvreader:
            attributes = []
            for i in range(4, len(row)):
                attributes.append(float(row[i]))
            colors.append(int(row[2]))
            features.append(attributes)
    
    assert(len(colors) == len(features))

    d =  getDimension(features)
    m =  getNumColors(colors)
    k = 60

    print("d = ", d)
    print("m = ", m)
    print("k = ", k)
    print("Data Size = ", len(features))

    coreset_constructor = CORESET.Coreset_FMM(features, colors, k, m, d, 1000)
    features_coreset, colors_coreset = coreset_constructor.compute()

    print("Coreset Size = ", len(features_coreset))

    gamma_high = coreset_constructor.compute_gamma_upper_bound()

    print("Gamma Upper Bound = ", gamma_high)

if __name__ == "__main__":
    main()
