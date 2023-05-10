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


def plot_graph(graphname, x_label, x_values, y_label, y_values, outputdir):

    plt.clf()
    plt.plot(x_values, y_values)
    plt.title(graphname)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(outputdir + "/" + graphname, dpi=300)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def getNumColors(colors):
    color_set = set(colors)
    return len(color_set)

def getDimension(features):
    return len(features[0])

def main():
    print("****CORESET EXPERIMENTS: CENSUS FULL****")

    print("\tgrouped by sex (c=2)")
    # read the Census dataset grouped by sex (c=2)
    features = []
    colors = []
    with open(f'./datasets/census.csv', "r") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in csvreader:
            attributes = []
            for i in range(4, len(row)):
                attributes.append(float(row[i]))
            colors.append(int(row[1]))
            features.append(attributes)
    
    assert(len(colors) == len(features))

    d =  getDimension(features)
    m =  getNumColors(colors)
    k = 60

    print("d = ", d)
    print("m = ", m)
    print("k = ", k)

    t = []
    ratios = []
    errors = []
    coreset_constructor = CORESET.Coreset_FMM(features, colors, k, m, d, k)

    resultfile = open("census_coreset.csv", "a")
    # Vary the coreset size
    # Intially start with coreset size equal to number of colors
    # Increment by number of colors per iteration
    for coreset_size in range(1000, int(len(features)), 10000):

        print("coreset_size = ", coreset_size)

        # Update the size of the coreset
        coreset_constructor.update_coreset_size(coreset_size)

        # Take multiple readings for time
        exec_time = 0.0
        num_readings = 1
        for i in range(0, num_readings):
            start_time = time.time()
            coreset_constructor.compute()
            end_time = time.time()
            exec_time += (end_time - start_time)
        
        # Take the average
        exec_time = exec_time/num_readings

        # Record the measures
        t.append(exec_time)
        ratios.append(coreset_size/len(features))
        errors.append(coreset_constructor.e)
        resultfile.write(f'{coreset_size},{exec_time}, {coreset_constructor.e}\n')
        resultfile.flush()
    
    outdir = './graphs'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    graphname = f'census_t_vs_coresetsize_k{k}_d{d}_m{m}'
    plot_graph(graphname, "coreset_size/N", ratios, "time (s)", t, outdir)

    graphname = f'census_e_vs_coresetsize_k{k}_d{d}_m{m}'
    plot_graph(graphname, "coreset_size/N", ratios, "e", errors, outdir)


    
if __name__ == "__main__":
    main()
