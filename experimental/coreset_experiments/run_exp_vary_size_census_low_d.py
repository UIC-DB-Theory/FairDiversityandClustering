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
            for i in range(4, len(row)-20):
                attributes.append(float(row[i]))
            colors.append(int(row[2]))
            features.append(attributes)
    
    assert(len(colors) == len(features))

    d =  getDimension(features)
    m =  getNumColors(colors)

    k_vals = [20, 60, 100, 1000]
    coreset_sizes = [1000, 5000, 10000, 20000, 30000, 40000, 50000]

    print("d = ", d)
    print("m = ", m)


    resultfile = open("census_coreset_vary_k_size_low_d.csv", "w")
    resultfile.write(f'k,m,d,coreset_size,time,e\n')
    resultfile.flush()
    for k in k_vals:
        for coreset_size in coreset_sizes:
            print(f'Running: {k},{m},{d},{coreset_size}')

            coreset_constructor = CORESET.Coreset_FMM(features, colors, k, m, d, coreset_size)

            start_time = time.time()
            coreset_constructor.compute()
            end_time = time.time()

            t = (end_time - start_time)
            e = coreset_constructor.e
            resultfile.write(f'{k},{m},{d},{coreset_size},{t},{e}\n')
            resultfile.flush()

if __name__ == "__main__":
    main()
