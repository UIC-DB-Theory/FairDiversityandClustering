"""
Description:
    Measure runtime and diversity for varying k. Where k is the size of the output.
    For the Adult dataset
    
    Usage: python3 exp_t_div_vs_k_runner.py /path/to/setup/file.json
"""

import sys
import re
import os
import json
import numpy as np
import copy

from datetime import datetime

# Parse setup file path
setup_file_path = sys.argv[1]
result = re.search(r'^(.+)\/([^\/]+)$', setup_file_path)
setup_file_dir = result.group(1)
setup_file_name = result.group(2)
print(result.group(1))
print(f'Setup file: {setup_file_path}')

# Create result location -- in the same directory as the setup file
result_file_dir = setup_file_dir + '/result_' + setup_file_name.split('.')[0]
if not os.path.exists(result_file_dir):
   os.mkdir(result_file_dir)

# Create result file 
timestamp = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
result_file_name = 'result_' + timestamp + '.json'
result_file_path = result_file_dir + '/' + result_file_name
print(f'Result file: {result_file_path}')

# Read the setup file
setup = {}
with open(setup_file_path, 'r') as json_file:
    setup = json.load(json_file)


from algorithms.sfdm2 import StreamFairDivMax2
from algorithms.fmmds import FMMDS
from algorithms.fairflow import FairFlow
from fmmdmwu_nyoom import epsilon_falloff as FMMDMWU
from algorithms.fairgreedyflow import FairGreedyFlow
from fmmd_lp import epsilon_falloff as FMMDLP
from fmmdmwu_sampled import epsilon_falloff as FMMDMWUS
from algorithms.utils import buildKisMap
# Lambdas for running experiments
algorithms = {
    'SFDM-2' : lambda k, kwargs: StreamFairDivMax2(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']), 
        epsilon = setup['algorithms']['SFDM-2']['epsilon'], 
        gammahigh = kwargs['dmax'], 
        gammalow = kwargs['dmin'], 
        normalize = False
    ),
    'FMMD-S' : lambda k, kwargs: FMMDS(
        features = kwargs['features'],
        colors = kwargs['colors'],
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']),
        epsilon = setup['algorithms']['FMMD-S']['epsilon'],
        normalize = False
    ),
    'FairFlow' : lambda k, kwargs : FairFlow(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']), 
        normalize = False
    ),
    'FairGreedyFlow' : lambda k, kwargs : FairGreedyFlow(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']), 
        epsilon= setup['algorithms']['FairGreedyFlow']['epsilon'], 
        gammahigh=kwargs['dmax'], 
        gammalow = kwargs['dmin'], 
        normalize=False
    ),
    'FMMD-MWU' : lambda k, kwargs : FMMDMWU(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']),
        gamma_upper = kwargs['dmax'],
        mwu_epsilon = setup['algorithms']['FMMD-MWU']['mwu_epsilon'],
        falloff_epsilon = setup['algorithms']['FMMD-MWU']['falloff_epsilon'],
        percent_theoretical_limit = setup['algorithms']['FMMD-MWU']['percent_theoretical_limit'],
        return_unadjusted = False
    ),
    'FMMD-LP' : lambda k, kwargs : FMMDLP(
        features = kwargs['features'], 
        colors = kwargs['colors'],
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']), 
        upper_gamma = kwargs['dmax'],
        epsilon = setup['algorithms']['FMMD-LP']['epsilon'], 
    ),
    'FMMD-MWUS' : lambda k, kwargs : FMMDMWUS(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']),
        gamma_upper=kwargs['dmax'],
        mwu_epsilon=setup['algorithms']['FMMD-MWUS']['mwu_epsilon'],
        falloff_epsilon=setup['algorithms']['FMMD-MWUS']['falloff_epsilon'],
        return_unadjusted=False,
        sample_percentage=setup['algorithms']['FMMD-MWUS']['sample_percentage'],
    ),
}


def write_results(setup, results):
    print("Writting summary...")
    summary = {
        "setup" : setup,
        "results" : results
    }
    # Save the results from the experiment
    json_object = json.dumps(summary, indent=4)
    with open(result_file_path, "w") as outfile:
        outfile.write(json_object)
        outfile.flush()

from contextlib import contextmanager
import signal
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


# Main experiment loop
results = {}
from datasets.utils import read_dataset
for dataset_name in setup["datasets"]:
    print(f'****************************MAIN LOOP******************************', file=sys.stderr)
    print(f'Dataset: {dataset_name}')

    results_per_k_per_alg = {}
    for k in range(setup["parameters"]["k"][0] ,setup["parameters"]["k"][1], setup["parameters"]["k"][2]):
        print(f'\tRunning for k = {k}...')
        print()
        # each observation in the list would consist of the t & div for each algorithm
        observations = []

        for obs in range(0, setup['parameters']['observations']):

            print(f'Observation number = {obs + 1}')

            # Read the dataset everytime -- to prevent overwriting the features and colors
            dataset = read_dataset(
                setup['datasets'][dataset_name]['data_dir'],
                setup['datasets'][dataset_name]['feature_fields'],
                setup['datasets'][dataset_name]['color_fields'],
                normalize=setup["datasets"][dataset_name]['normalize'],
                unique=setup["datasets"][dataset_name]['filter_unique']
            )
            setup["datasets"][dataset_name]['points_per_color'] = dataset['points_per_color']
            write_results(setup, results)
            features = dataset['features']
            colors = dataset['colors']

            # Calculate the coreset, dmax, dmin (same for each alg in each observation)
            from algorithms.coreset import Coreset_FMM
            dimensions = len(setup["datasets"][dataset_name]["feature_fields"])
            num_colors = len(setup["datasets"][dataset_name]['points_per_color'])
            coreset_size = num_colors * k
            coreset = Coreset_FMM(
                features, 
                colors, 
                k, 
                num_colors, 
                dimensions, 
                coreset_size)
            core_features, core_colors = coreset.compute()
            dmax = coreset.compute_gamma_upper_bound()
            dmin = coreset.compute_closest_pair()

            result_per_alg = {}
            for alg in setup['algorithms']:
                print()
                print(f'\t\t\tRunning {alg}...')
                t = 0
                div = 0
                data_size = 0
                
                alg_args = copy.deepcopy(setup['algorithms'][alg])
                alg_args['features'] = features
                alg_args['colors'] = colors

                if (setup['algorithms'][alg]['use_coreset']):
                    print(f'\t\tcomputed coreset size  = {len(core_features)}')
                    t = t + coreset.coreset_compute_time
                    alg_args['features'] = core_features
                    alg_args['colors'] = core_colors

                if (setup['algorithms'][alg]['use_dmax']):
                    print(f'\t\tcomputed dmax = {dmax}')
                    t = t + coreset.gamma_upper_bound_compute_time
                    alg_args['dmax'] = dmax

                if (setup['algorithms'][alg]['use_dmin']):
                    print(f'\t\tcomputed dmin = {dmin}')
                    t = t + coreset.closest_pair_compute_time
                    alg_args['dmin'] = dmin

                runner = algorithms[alg]
                sol, div, t_alg = runner(k, alg_args)
                t = t + t_alg
                print(f'\t\t***solution size = {len(sol)}***')
                print(f'\t\tdiv = {div}')
                print(f'\t\tt = {t}')
                result_per_alg[alg] = [len(alg_args['features']), dmax, dmin, len(sol), div, t]
                # End of algorithms loop

            observations.append(result_per_alg)
        # End of observations loop

        avgs = {}
        # Average out the observations
        for alg in setup['algorithms']:
            for i in range(0, setup['parameters']['observations']):
                observation = observations[i][alg]
                if alg not in avgs:
                    avgs[alg] = [observation]
                else:
                    avgs[alg].append(observation)
        for alg in avgs:
            avgs[alg] = np.mean(np.array(avgs[alg]), axis=0).tolist()
            if k not in results_per_k_per_alg:
                results_per_k_per_alg[k] = {alg: avgs[alg]}
            else:
                results_per_k_per_alg[k][alg] = avgs[alg]
    # End of k loop
    print(results_per_k_per_alg)
    results[dataset_name] = {}
    for k in results_per_k_per_alg:
        for alg in results_per_k_per_alg[k]:
            if alg not in results[dataset_name]:
                results[dataset_name][alg] = {
                    'xs' : {
                        'k' : [k]
                    },
                    'ys' : {
                        'data_size' : [results_per_k_per_alg[k][alg][0]],
                        'dmax' : [results_per_k_per_alg[k][alg][1]],
                        'dmin' : [results_per_k_per_alg[k][alg][2]],
                        'solution_size' : [results_per_k_per_alg[k][alg][3]],
                        'diversity' : [results_per_k_per_alg[k][alg][4]],
                        'runtime' : [results_per_k_per_alg[k][alg][5]],
                        'div-runtime' : [results_per_k_per_alg[k][alg][4]/results_per_k_per_alg[k][alg][5]]
                    }
                }
            else:
                results[dataset_name][alg]['xs']['k'].append(k)
                results[dataset_name][alg]['ys']['data_size'].append(results_per_k_per_alg[k][alg][0])
                results[dataset_name][alg]['ys']['dmax'].append(results_per_k_per_alg[k][alg][1])
                results[dataset_name][alg]['ys']['dmin'].append(results_per_k_per_alg[k][alg][2])
                results[dataset_name][alg]['ys']['solution_size'].append(results_per_k_per_alg[k][alg][3])
                results[dataset_name][alg]['ys']['diversity'].append(results_per_k_per_alg[k][alg][4])
                results[dataset_name][alg]['ys']['runtime'].append(results_per_k_per_alg[k][alg][5])
                results[dataset_name][alg]['ys']['div-runtime'].append(results_per_k_per_alg[k][alg][4]/results_per_k_per_alg[k][alg][5])

# End of dataset loop

write_results(setup, results)
