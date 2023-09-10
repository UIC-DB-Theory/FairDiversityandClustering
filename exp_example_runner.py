"""
Description:
    Measure runtime and diversity for varying k. Where k is the size of the output.
    For the Adult dataset
"""
import sys
import json
import numpy as np
import signal
from contextlib import contextmanager
from algorithms.coreset import Coreset_FMM

setup = {}
setup_file = sys.argv[1]
result_file = setup_file.split('.')[0] + ".result"
with open(setup_file, 'r') as json_file:
    setup = json.load(json_file)

results = {}


def write_results(setup, result):
    print("Writting summary...")
    summary = {
        "setup" : setup,
        "results" : results
    }
    # Save the results from the experiment
    json_object = json.dumps(summary, indent=4)
    print(json_object)
    with open(result_file, "w") as outfile:
        outfile.write(json_object)
        outfile.flush()


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
    from algorithms.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)
    d = dataset["d"]
    m = dataset["m"]
    coreset_size = 10*k
    features = dataset["features"]
    colors = dataset["colors"]
    print("\tCompute coreset")
    coreset = Coreset_FMM(features, colors, k, m, d, coreset_size)
    core_features, core_colors = coreset.compute()
    print(f'\t\tcoreset_size = {len(core_features)}')

    print("\tCompute gamma_high")
    dmax = coreset.compute_gamma_upper_bound()
    print(f'\t\t{dmax}')
    print("\tCompute gamma_low")
    dmin = coreset.compute_closest_pair()
    while dmin == 0:
        dmin = 0.1
    print(f'\t\t{dmin}')

    print("\tRunning algorithm instance...")
    from algorithms.sfdm2 import StreamFairDivMax2
    _, div, t = StreamFairDivMax2(
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
    print(f'\t\tt = {t}, div = {div}')
    return t, div

# Define experiment for FMMD-S
def experiment_fmmds(dataset, k):
    print("Running experiment for FMMD-S")
    print(f'\t\tk = {k}')
    from algorithms.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)
    d = dataset["d"]
    m = dataset["m"]
    coreset_size = 10*k
    features = dataset["features"]
    colors = dataset["colors"]
    print("\tRunning algorithm instance...")
    from algorithms.fmmds import FMMDS
    _, div, t = FMMDS(
        features = features,
        colors = colors,
        kis = kis,
        epsilon= 0.15,
        normalize=False
    )
    print(f'\t\tt = {t}, div = {div}')
    return t, div

# Define experiment for FairFlow
def experiment_fairflow(dataset, k, include_coreset_time = False):
    print("Running experiment for FairFlow")
    print(f'\t\tk = {k}')
    from algorithms.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)
    d = dataset["d"]
    m = dataset["m"]
    coreset_size = 10*k
    features = dataset["features"]
    colors = dataset["colors"]
    print("\tCompute coreset")
    coreset = Coreset_FMM(features, colors, k, m, d, coreset_size)
    core_features, core_colors = coreset.compute()
    print(f'\t\tcoreset_size = {len(core_features)}')

    print("\tRunning algorithm instance...")
    from algorithms.fairflow import FairFlow
    _, div, t = FairFlow(
        features = core_features, 
        colors = core_colors, 
        kis = kis, 
        normalize=False
    )

    if include_coreset_time:
        t = t + coreset.coreset_compute_time
    print(f'\t\tt = {t}, div = {div}')
    return t, div

# Define experiment for FairGreedyFlow
def experiment_fairgreedyflow(dataset, k, include_coreset_time = False, include_gamma_high_time = False, include_gamma_low_time = False):
    print("Running experiment for FairGreedyFlow")
    print(f'\t\tk = {k}')
    from algorithms.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)
    d = dataset["d"]
    m = dataset["m"]
    coreset_size = 10*k
    features = dataset["features"]
    colors = dataset["colors"]
    print("\tCompute coreset")
    coreset = Coreset_FMM(features, colors, k, m, d, coreset_size)
    core_features, core_colors = coreset.compute()
    print(f'\t\tcoreset_size = {len(core_features)}')

    print("\tCompute gamma_high")
    dmax = coreset.compute_gamma_upper_bound()
    print(f'\t\t{dmax}')
    print("\tCompute gamma_low")
    dmin = coreset.compute_closest_pair()
    while dmin == 0:
        dmin = 0.1
    print(f'\t\t{dmin}')

    print("\tRunning algorithm instance...")
    from algorithms.fairgreedyflow import FairGreedyFlow
    _, div, t = FairGreedyFlow(
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
    print(f'\t\tt = {t}, div = {div}')
    return t, div

# Define experiment for FMMD-MWU
def experiment_fmmdmwu(dataset, k, include_coreset_time = False, include_gamma_high_time = False):
    print("Running experiment for FMMD-MWU")
    print(f'\t\tk = {k}')
    from algorithms.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)
    d = dataset["d"]
    m = dataset["m"]
    coreset_size = 10*k
    features = dataset["features"]
    colors = dataset["colors"]
    print("\tCompute coreset")
    coreset = Coreset_FMM(features, colors, k, m, d, coreset_size)
    core_features, core_colors = coreset.compute()
    print(f'\t\tcoreset_size = {len(core_features)}')

    print("\tCompute gamma_high")
    dmax = coreset.compute_gamma_upper_bound()
    print(f'\t\t{dmax}')

    print("\tRunning algorithm instance...")
    from fmmdmwu_nyoom import epsilon_falloff as FMMDMWU
    _, div, t = FMMDMWU(
        features = core_features, 
        colors = core_colors, 
        kis = kis,
        gamma_upper=dmax,
        mwu_epsilon=0.75,
        falloff_epsilon=0.15,
        return_unadjusted=False
    )

    if include_coreset_time:
        t = t + coreset.coreset_compute_time
    
    if include_gamma_high_time:
        t = t + coreset.gamma_upper_bound_compute_time
    print(f'\t\tt = {t}, div = {div}')
    return t, div

# Define experiment for FMMD-LP
def experiment_fmmdlp(dataset, k, include_coreset_time = False, include_gamma_high_time = False):
    print("Running experiment for FMMD-LP")
    print(f'\t\tk = {k}')
    from algorithms.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)
    d = dataset["d"]
    m = dataset["m"]
    coreset_size = 10*k
    features = dataset["features"]
    colors = dataset["colors"]
    print("\tCompute coreset")
    coreset = Coreset_FMM(features, colors, k, m, d, coreset_size)
    core_features, core_colors = coreset.compute()
    print(f'\t\tcoreset_size = {len(core_features)}')

    print("\tCompute gamma_high")
    dmax = coreset.compute_gamma_upper_bound()
    print(f'\t\t{dmax}')

    print("\tRunning algorithm instance...")
    from fmmd_lp import epsilon_falloff as FMMDLP
    _, div, t = FMMDLP(
        features = core_features, 
        colors = core_colors,
        upper_gamma = dmax,
        kis = kis, 
        epsilon = 0.15, 
    )

    if include_coreset_time:
        t = t + coreset.coreset_compute_time

    if include_gamma_high_time:
        t = t + coreset.gamma_upper_bound_compute_time

    
    print(f'\t\tt = {t}, div = {div}')
    return t, div

# Lambdas for running experiments
alg_experiments = {
    'SFDM-2' : lambda dataset, k : experiment_sfdm2(dataset, k, include_coreset_time=True, include_gamma_high_time=True, include_gamma_low_time=True),
    'FMMD-S' : lambda dataset, k : experiment_fmmds(dataset, k),
    'FairFlow' : lambda dataset, k : experiment_fairflow(dataset, k, include_coreset_time=True),
    'FairGreedyFlow' : lambda dataset, k : experiment_fairgreedyflow(dataset, k, include_coreset_time=True, include_gamma_high_time=True, include_gamma_low_time=True),
    'FMMD-MWU' : lambda dataset, k : experiment_fmmdmwu(dataset, k, include_coreset_time=True, include_gamma_high_time=True),
    'FMMD-LP' : lambda dataset,k : experiment_fmmdlp(dataset, k, include_coreset_time=True, include_gamma_high_time=True)
}


class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def main():
    # Run the experiments
    write_results(setup, results)
    # For each dataset
    for dataset_name, dataset in datasets.items():
        # Run each algorithm
        for alg, runner in alg_experiments.items():
            k_values = []
            runtimes = []
            diversity_values = []
            # While varying k
            for k in range(setup["parameters"]["k"][0] ,setup["parameters"]["k"][1], setup["parameters"]["k"][2]):

                
                t = 0
                div = 0
                timedout = False

                if "timeout" in setup["parameters"]:
                    # With timeout
                    for rn in range(0, setup["parameters"]["observations"]):
                        print(rn)
                        try:
                            with time_limit(setup["parameters"]["timeout"]):
                                t_val, div_val = runner(dataset, k)
                        except TimeoutException as e:
                            timedout = True
                            print("Timed out!")

                        t = t + t_val
                        div = div+div_val
                    t = t/setup["parameters"]["observations"]
                    div = div/setup["parameters"]["observations"]

                else:
                    # No timeout
                    for rn in range(0, setup["parameters"]["observations"]):
                        t_val, div_val = runner(dataset, k)
                        t = t + t_val
                        div = div+div_val
                    t = t/setup["parameters"]["observations"]
                    div = div/setup["parameters"]["observations"]
                if not timedout:
                    k_values.append(k)
                    runtimes.append(t)
                    diversity_values.append(div)
                else:
                    t = -1
                    div = -1
                print(f"k = {k}, t = {t}, div = {div}")

            if dataset_name not in results:
                results[dataset_name] = {
                    alg : {
                        "xs" : {"k_values" : k_values},
                        "ys" : {"runtimes" : runtimes, "diversity_values" : diversity_values}
                    } 
                }
            else:
                results[dataset_name][alg] = {
                    "xs" : {"k_values" : k_values},
                    "ys" : {"runtimes" : runtimes, "diversity_values" : diversity_values}
                }
            write_results(setup, results)

    write_results(setup, results)

if __name__ == "__main__":
    main()