"""
Description:
    Measure runtime and diversity for varying k. Where k is the size of the output.
    For the Adult dataset
    
    Usage: python3 exp_t_div_vs_k_runner.py /path/to/setup/file.setup
"""
import sys
import json
import numpy as np
import signal
import os
from contextlib import contextmanager
from algorithms.coreset import Coreset_FMM
from datetime import datetime
timestamp = datetime.now().strftime('%H_%M_%d_%m_%Y')

setup = {}
results = {}

setup_file = sys.argv[1]
result_file = setup_file.split('/')[-1].split('.')[0] + "_" + timestamp + ".json"
result_dir = './experiments/' + result_file.split('.')[0]
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


result_file = result_dir + "/" + result_file
print(f'Writing results to: {result_file}')

with open(setup_file, 'r') as json_file:
    setup = json.load(json_file)


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
    
    datasets[dataset]["coresets"] = {}

    print("Precomputing coresets...")
    # precompute the coreset for all values of k. Same for each algorithm but different for each observation.
    for k in range(setup["parameters"]["k"][0] ,setup["parameters"]["k"][1], setup["parameters"]["k"][2]):

        datasets[dataset]["coresets"][k] = []
        coreset_size = datasets[dataset]["m"]*k

        for i in range(0, setup["parameters"]["observations"]):
            coreset = Coreset_FMM(datasets[dataset]["features"], datasets[dataset]["colors"], k, datasets[dataset]["m"], datasets[dataset]["d"], coreset_size)
            print(f'\tComputing for k = {k}, coreset_size = {coreset_size}')
            out, _= coreset.compute()
            print(f'\t\tcomputed coreset_size = {len(out)}')
            gup = coreset.compute_gamma_upper_bound()
            print(f'\t\tgamma upper bound = {gup}')
            clop = coreset.compute_closest_pair()
            print(f'\t\tclosest pair distance = {clop}')
            datasets[dataset]["coresets"][k].append(coreset)


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


# Define experiment for SFDM-2
def experiment_sfdm2(i, dataset, k, include_coreset_time = False, include_gamma_high_time = False, include_gamma_low_time = False, use_coreset = False):
    print("Running experiment for SFDM-2")
    print(f'\t\tk = {k}')
    from algorithms.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)

    features = dataset["features"]
    colors = dataset["colors"]

    coreset = dataset["coresets"][k][i]
    if use_coreset:
        print('\t\tUsing coreset')
        features = coreset.out_features
        colors = coreset.out_colors
        print(f'\t\tcoreset_size = {len(features)}')

    print("\tCompute gamma_high")
    dmax = coreset.gamma_upper_bound
    print(f'\t\t{dmax}')
    print("\tCompute gamma_low")
    dmin = coreset.closest_pair
    print(f'\t\t{dmin}')

    print("\tRunning algorithm instance...")
    from algorithms.sfdm2 import StreamFairDivMax2
    _, div, t = StreamFairDivMax2(
        features = features, 
        colors = colors, 
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
def experiment_fmmds(i, dataset, k, use_coreset = False):
    print("Running experiment for FMMD-S")
    print(f'\t\tk = {k}')
    from algorithms.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)

    coreset = dataset["coresets"][k][i]
    if use_coreset:
        print('\t\tUsing coreset')
        features = coreset.out_features
        colors = coreset.out_colors
        print(f'\t\tcoreset_size = {len(features)}')
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
def experiment_fairflow(i, dataset, k, include_coreset_time = False, use_coreset = False):
    print("Running experiment for FairFlow")
    print(f'\t\tk = {k}')
    from algorithms.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)

    features = dataset["features"]
    colors = dataset["colors"]
    coreset = dataset["coresets"][k][i]
    if use_coreset:
        print('\t\tUsing coreset')
        features = coreset.out_features
        colors = coreset.out_colors
        print(f'\t\tcoreset_size = {len(features)}')

    print("\tRunning algorithm instance...")
    from algorithms.fairflow import FairFlow
    _, div, t = FairFlow(
        features = features, 
        colors = colors, 
        kis = kis, 
        normalize=False
    )

    if include_coreset_time:
        t = t + coreset.coreset_compute_time
    print(f'\t\tt = {t}, div = {div}')
    return t, div

# Define experiment for FairGreedyFlow
def experiment_fairgreedyflow(i, dataset, k, include_coreset_time = False, include_gamma_high_time = False, include_gamma_low_time = False, use_coreset = False):
    print("Running experiment for FairGreedyFlow")
    print(f'\t\tk = {k}')
    from algorithms.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)

    features = dataset["features"]
    colors = dataset["colors"]
    coreset = dataset["coresets"][k][i]
    if use_coreset:
        print('\t\tUsing coreset')
        features = coreset.out_features
        colors = coreset.out_colors
        print(f'\t\tcoreset_size = {len(features)}')

        print("\tCompute gamma_high")
    dmax = coreset.gamma_upper_bound
    print(f'\t\t{dmax}')
    print("\tCompute gamma_low")
    dmin = coreset.closest_pair
    print(f'\t\t{dmin}')

    print("\tRunning algorithm instance...")
    from algorithms.fairgreedyflow import FairGreedyFlow
    _, div, t = FairGreedyFlow(
        features = features, 
        colors = colors, 
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
def experiment_fmmdmwu(i, dataset, k, include_coreset_time = False, include_gamma_high_time = False, use_coreset = False):
    print("Running experiment for FMMD-MWU")
    print(f'\t\tk = {k}')
    from algorithms.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)
    
    features = dataset["features"]
    colors = dataset["colors"]
    coreset = dataset["coresets"][k][i]
    if use_coreset:
        print('\t\tUsing coreset')
        features = coreset.out_features
        colors = coreset.out_colors
        print(f'\t\tcoreset_size = {len(features)}')

    print("\tCompute gamma_high")
    dmax = coreset.gamma_upper_bound
    print(f'\t\t{dmax}')

    print("\tRunning algorithm instance...")
    from fmmdmwu_nyoom import epsilon_falloff as FMMDMWU
    _, div, t = FMMDMWU(
        features = features, 
        colors = colors, 
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

# Define experiment for FMMD-MWUS - Sampled version
def experiment_fmmdmwus(i, dataset, k, include_coreset_time = False, include_gamma_high_time = False, use_coreset = False):
    print("Running experiment for FMMD-MWU")
    print(f'\t\tk = {k}')
    from algorithms.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)
    
    features = dataset["features"]
    colors = dataset["colors"]
    coreset = dataset["coresets"][k][i]
    if use_coreset:
        print('\t\tUsing coreset')
        features = coreset.out_features
        colors = coreset.out_colors
        print(f'\t\tcoreset_size = {len(features)}')

    print("\tCompute gamma_high")
    dmax = coreset.gamma_upper_bound
    print(f'\t\t{dmax}')

    print("\tRunning algorithm instance...")
    from fmmdmwu_sampled import epsilon_falloff as FMMDMWUS
    _, div, t = FMMDMWUS(
        features = features, 
        colors = colors, 
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
def experiment_fmmdlp(i, dataset, k, include_coreset_time = False, include_gamma_high_time = False, use_coreset = False):
    print("Running experiment for FMMD-LP")
    print(f'\t\tk = {k}')
    from algorithms.utils import buildKisMap
    kis = buildKisMap(dataset["colors"], k, 0.1)
    
    features = dataset["features"]
    colors = dataset["colors"]
    coreset = dataset["coresets"][k][i]
    if use_coreset:
        print('\t\tUsing coreset')
        features = coreset.out_features
        colors = coreset.out_colors
        print(f'\t\tcoreset_size = {len(features)}')

    print("\tCompute gamma_high")
    dmax = coreset.gamma_upper_bound
    print(f'\t\t{dmax}')


    print("\tRunning algorithm instance...")
    from fmmd_lp import epsilon_falloff as FMMDLP
    _, div, t = FMMDLP(
        features = features, 
        colors = colors,
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
    'SFDM-2' : lambda i, dataset, k : 
        experiment_sfdm2(i, dataset, k, include_coreset_time=True, include_gamma_high_time=True, include_gamma_low_time=True, use_coreset=setup["algorithms"]['SFDM-2']["coreset"]),
    'FMMD-S' : lambda i, dataset, k : experiment_fmmds(i, dataset, k, use_coreset=setup["algorithms"]['FMMD-S']["coreset"]),
    'FairFlow' : lambda i, dataset, k : experiment_fairflow(i, dataset, k, include_coreset_time=True, use_coreset=setup["algorithms"]['FairFlow']["coreset"]),
    'FairGreedyFlow' : lambda i, dataset, k : experiment_fairgreedyflow(i, dataset, k, include_coreset_time=True, include_gamma_high_time=True, include_gamma_low_time=True, use_coreset=setup["algorithms"]['FairGreedyFlow']["coreset"]),
    'FMMD-MWU' : lambda i, dataset, k : experiment_fmmdmwu(i, dataset, k, include_coreset_time=True, include_gamma_high_time=True, use_coreset=setup["algorithms"]['FMMD-MWU']["coreset"]),
    'FMMD-LP' : lambda i, dataset,k : experiment_fmmdlp(i, dataset, k, include_coreset_time=True, include_gamma_high_time=True, use_coreset=setup["algorithms"]['FMMD-LP']["coreset"]),
    'FMMD-MWUS' : lambda i, dataset, k : experiment_fmmdmwu(i, dataset, k, include_coreset_time=True, include_gamma_high_time=True, use_coreset=setup["algorithms"]['FMMD-MWUS']["coreset"]),
}


# Timeout implementation
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
        for alg in setup["algorithms"]:
        #for alg, runner in alg_experiments.items():
            runner = alg_experiments[alg]
            k_values = []
            runtimes = []
            diversity_values = []
            # While varying k
            for k in range(setup["parameters"]["k"][0] ,setup["parameters"]["k"][1], setup["parameters"]["k"][2]):

                t = 0
                div = 0
                all_timedout = False
                successful_observations = 0
                # Run the algorithms with or without timeout
                if "timeout" in setup["parameters"]:
                    # With timeout
                    for rn in range(0, setup["parameters"]["observations"]):
                        print(rn)
                        try:
                            with time_limit(setup["parameters"]["timeout"]):
                                t_val, div_val = runner(rn, dataset, k)
                                successful_observations += 1
                        except TimeoutException as e:
                            print("Timed out!")

                        t = t + t_val
                        div = div+div_val
                    if successful_observations == 0:
                            all_timedout = True
                            break
                    t = t/successful_observations
                    div = div/successful_observations

                else:
                    # No timeout
                    for rn in range(0, setup["parameters"]["observations"]):
                        t_val, div_val = runner(rn, dataset, k)
                        t = t + t_val
                        div = div+div_val
                    t = t/setup["parameters"]["observations"]
                    div = div/setup["parameters"]["observations"]
                
                # Check if it timed out, only add values to result if it didnt
                if not all_timedout:
                    k_values.append(k)
                    runtimes.append(t)
                    diversity_values.append(div)
                else:
                    t = -1
                    div = -1
                    break
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