#Coreset size experiment
# lpsolve - simple lp solving based approach to the problem
import sys
import os
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import trange

# Adding parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import coreset as CORESET
import utils

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="data file")
    parser.add_argument("-d", "--dimension", help="data dimension")
    args = parser.parse_args()
    return args

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

def main(args):
    print("****CORESET EXPERIMENTS****")
    print("file: ", args.file)
    
    if args.file and not os.path.isfile(args.file):
        print('File not found.')
        print('for help run: python3 run_exp_vary_size.py -h')
        exit(-1)
    
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
    # variables for running LP bin-search
    color_field = 'sex'
    feature_fields = ['age', 'capital-gain', 'capital-loss']
    kis = {"Male": 10, "Female": 10}
    k = 20
    colors, features = utils.read_CSV("../datasets/ads/adult.data", allFields, color_field, feature_fields)
    assert (len(colors) == len(features))

    # "normalize" features
    # Should happen before coreset construction
    features = features / features.max(axis=0)
    
    d = len(feature_fields)
    m = len(kis.keys())
    t = []
    ratios = []
    errors = []
    coreset_constructor = CORESET.Coreset_FMM(features, colors, k, m, d, k)

    # Vary the coreset size
    # Intially start with coreset size equal to number of colors
    # Increment by number of colors per iteration
    for coreset_size in trange(k, int(len(features)), m):

        # Update the size of the coreset
        coreset_constructor.update_coreset_size(coreset_size)

        # Take multiple readings for time
        exec_time = 0.0
        for i in range(0, 10):
            start_time = time.time()
            coreset_constructor.compute()
            end_time = time.time()
            exec_time += (end_time - start_time)
        
        # Take the average
        exec_time = exec_time/10

        # Record the measures
        t.append(exec_time)
        ratios.append(coreset_size/len(features))
        errors.append(coreset_constructor.e)
    
    outdir = './graphs'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    graphname = f't_vs_coresetsize_k{k}_d{m}_m{d}'
    plot_graph(graphname, "coreset_size/N", ratios, "time (s)", t, outdir)

    graphname = f'e_vs_coresetsize_k{k}_d{m}_m{d}'
    plot_graph(graphname, "coreset_size/N", ratios, "e", errors, outdir)
    
if __name__ == "__main__":
    main(parseArgs())