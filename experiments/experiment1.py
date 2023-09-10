"""
Description:
    Measure runtime and diversity for varying k. Where k is the size of the output.
"""
import sys
import json
import numpy as np
sys.path.append("..")


setup = {
    "datasets" : {
        "Adult" : {
            "data_dir" : "../datasets/adult",
            "color_fields" : [
                "race", 
                "sex"
            ],
            "feature_fields" : [
                "age", 
                "capital-gain", 
                "capital-loss", 
                "hours-per-week", 
                "fnlwgt", 
                "education-num"],
            "normalize" : 1
        }
    },
    "parametes" : {
        "k" : [25, 351, 50]
    }
}


# Read all the datasets
from datasets.utils import read_dataset
datasets = {}
for dataset in setup["datasets"]:
    datasets[dataset] = read_dataset(
        setup["datasets"][dataset]["data_dir"],
        setup["datasets"][dataset]["feature_fields"],
        setup["datasets"][dataset]["color_fields"],
        normalize=True if setup["datasets"][dataset]["normalize"] == 1 else False
    )
    setup["datasets"][dataset]["points_per_color"] = datasets[dataset]["points_per_color"]
    datasets[dataset]["d"] = len(setup["datasets"][dataset]["feature_fields"])
    datasets[dataset]["m"] = len(datasets[dataset]["points_per_color"])

# Define experiment for SFDM-2
def experiment_sfdm2(dataset, k, include_coreset_time = False, include_gamma_high_time = False, include_gamma_low_time = False):
    print("Running experiment for SFDM-2")
    print(f'\t\tk = {k}')
    from experiments.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)
    d = dataset["d"]
    m = dataset["m"]
    coreset_size = 10*k
    features = dataset["features"]
    colors = dataset["colors"]
    print("\tCompute coreset")
    from coreset import Coreset_FMM
    coreset = Coreset_FMM(features, colors, k, m, d, coreset_size)
    core_features, core_colors = coreset.compute()
    print(f'\t\tcoreset_size = {len(core_features)}')

    print("\tCompute gamma_high")
    dmax = coreset.compute_gamma_upper_bound()
    print(f'\t\t{dmax}')
    print("\tCompute gamma_low")
    dmin = coreset.compute_closest_pair()
    print(f'\t\t{dmin}')

    print("\tRunning algorithm instance")
    from algorithms.sfdm2 import StreamFairDivMax2
    sol, div, t = StreamFairDivMax2(
        features = core_features, 
        colors = core_colors, 
        kis = kis, 
        epsilon= 0.15, 
        gammahigh=dmax, 
        gammalow=dmin, 
        normalize=False
    )

    if include_coreset_time:
        t = t + coreset.coreset_compute_time
    
    if include_gamma_high_time:
        t = t + coreset.gamma_upper_bound_compute_time
    
    if include_gamma_low_time:
        t = t + coreset.closest_pair_compute_time

    return t, div

k_values = []
runtimes = []
diversity_values = []
for k in range(25, 300, 25):
    k_values.append(k)
    t, div = experiment_sfdm2(datasets["Adult"], k)
    runtimes.append(t)
    diversity_values.append(div)
    print(f"k = {k}, t = {t}, div = {div}")

import matplotlib.pyplot as plt
# # Plot the graph t vs k
plt.clf()
plt.plot(k_values, runtimes)

plt.yscale("log")
plt.legend(title = "time vs k - Adult Full", bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.xlabel("k")
plt.ylabel("runtime (s)")
plt.savefig("t_vs_k_adult_full", dpi=300, bbox_inches='tight')

# Plot the graph reported d vs k 
plt.clf()
plt.plot(k_values, diversity_values)

# plt.yscale("log")
plt.legend(title = "d vs k - Adult Full", bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.xlabel("k")
plt.ylabel("d")
plt.savefig("d_reported_vs_k_adult_full", dpi=300, bbox_inches='tight')

plt.close()