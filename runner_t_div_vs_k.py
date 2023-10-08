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

import random
# just in case, but we don't use these
random.seed(0)
np.random.seed(0)

# global generator used for all randomness
gen = np.random.default_rng(seed=0)

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
    'SFDM-2' : lambda gen, name, kis, kwargs: StreamFairDivMax2(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = kis, 
        epsilon = setup['algorithms'][name]['epsilon'], 
        gammahigh = kwargs['dmax'], 
        gammalow = kwargs['dmin'], 
        normalize = False
    ),
    'FMMD-S' : lambda gen, name, kis, kwargs: FMMDS(
        features = kwargs['features'],
        colors = kwargs['colors'],
        kis = kis,
        epsilon = setup['algorithms'][name]['epsilon'],
        normalize = False
    ),
    'FairFlow' : lambda gen, name, kis, kwargs : FairFlow(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = kis, 
        normalize = False
    ),
    'FairGreedyFlow' : lambda gen, name, kis, kwargs : FairGreedyFlow(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = kis, 
        epsilon= setup['algorithms'][name]['epsilon'], 
        gammahigh=kwargs['dmax'], 
        gammalow = kwargs['dmin'], 
        normalize=False
    ),
    'FMMD-MWU' : lambda gen, name, kis, kwargs : FMMDMWU(
        gen=gen,
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = kis,
        gamma_upper = kwargs['dmax'],
        mwu_epsilon = setup['algorithms'][name]['mwu_epsilon'],
        falloff_epsilon = setup['algorithms'][name]['falloff_epsilon'],
        percent_theoretical_limit = setup['algorithms'][name]['percent_theoretical_limit'],
        return_unadjusted = False
    ),
    'FMMD-LP' : lambda gen, name, kis, kwargs : FMMDLP(
        gen=gen,
        features = kwargs['features'], 
        colors = kwargs['colors'],
        kis = kis, 
        upper_gamma = kwargs['dmax'],
        epsilon = setup['algorithms'][name]['epsilon'], 
    ),
    'FMMD-MWUS' : lambda gen, name, kis, kwargs : FMMDMWUS(
        gen=gen,
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = kis,
        gamma_upper=kwargs['dmax'],
        mwu_epsilon=setup['algorithms'][name]['mwu_epsilon'],
        falloff_epsilon=setup['algorithms'][name]['falloff_epsilon'],
        return_unadjusted=False,
        sample_percentage=setup['algorithms'][name]['sample_percentage'],
        percent_theoretical_limit=setup['algorithms'][name]['percent_theoretical_limit'],
    ),
}

def check_flag(struct, flag):
    
    if flag in struct:
        return struct[flag]
    else:
        return False

def write_results(setup, results, color_results):

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
    
    print("Writting summary...")
    summary = {
        "setup" : setup,
        "results" : results,
        "color_results" : color_results
    }
    # Save the results from the experiment
    json_object = json.dumps(summary, indent=4, cls=NpEncoder)
    with open(result_file_path, "w") as outfile:
        outfile.write(json_object)
        outfile.flush()



timeout_dict = {}
for alg in setup['algorithms']:
    timeout_dict[alg] = False



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
color_results = []
alg_status = []
from datasets.utils import read_dataset
for dataset_name in setup["datasets"]:
    print(f'****************************MAIN LOOP******************************', file=sys.stderr)
    print(f'Dataset: {dataset_name}')

    results_per_k_per_alg = {}
    for k in range(setup["parameters"]["k"][0] ,setup["parameters"]["k"][1], setup["parameters"]["k"][2]):

        # each observation in the list would consist of the t & div for each algorithm
        observations = []

        # Read the dataset everytime -- to prevent overwriting the features and colors
        dataset = read_dataset(
            setup['datasets'][dataset_name]['data_dir'],
            setup['datasets'][dataset_name]['feature_fields'],
            setup['datasets'][dataset_name]['color_fields'],
            normalize=setup["datasets"][dataset_name]['normalize'],
            unique=setup["datasets"][dataset_name]['filter_unique']
        )
        setup["datasets"][dataset_name]['points_per_color'] = dataset['points_per_color']
        setup["datasets"][dataset_name]['size'] = len(dataset['features'])
        write_results(setup, results, color_results)
        features = dataset['features']
        colors = dataset['colors']

        # one kis' map to ask for
        kimap = buildKisMap(dataset['colors'], k, setup['parameters']['buildkis_alpha'], equal_k_js=check_flag(setup['parameters'],'buildkis_equal_k_js'))
        adj_k = sum(kimap.values()) # the actual number of points we asked for

        print(f'***************************************')
        print(f'\t***Running for k = {adj_k}, {k}...')
        print(json.dumps(kimap, indent=4))
        print(f'***************************************')

        for obs in range(0, setup['parameters']['observations']):

            print(f'Observation number = {obs + 1}')

            # Calculate the coreset, dmax, dmin (same for each alg in each observation)
            from algorithms.coreset import Coreset_FMM
            dimensions = len(setup["datasets"][dataset_name]["feature_fields"])
            num_colors = len(setup["datasets"][dataset_name]['points_per_color'])
            coreset_size = num_colors * adj_k
            coreset = Coreset_FMM(
                gen,
                features, 
                colors, 
                adj_k, 
                num_colors, 
                dimensions, 
                coreset_size)
            core_features, core_colors = coreset.compute()
            dmax = coreset.compute_gamma_upper_bound()
            dmin = coreset.compute_closest_pair()

            result_per_alg = {}
            for name in setup['algorithms']:
                print()
                print(f'\t\t\tRunning {name}...')
                t = 0
                div = 0
                data_size = 0
                
                alg_args = copy.deepcopy(setup['algorithms'][name])
                alg_args['features'] = copy.deepcopy(features)
                alg_args['colors'] = copy.deepcopy(colors)

                if (check_flag(setup['algorithms'][name],'use_coreset')):
                    print(f'\t\tcomputed coreset size  = {len(core_features)}')
                    t = t + coreset.coreset_compute_time
                    alg_args['features'] = copy.deepcopy(core_features)
                    alg_args['colors'] = copy.deepcopy(core_colors)

                if (check_flag(setup['algorithms'][name],'use_dmax')):
                    print(f'\t\tcomputed dmax = {dmax}')
                    t = t + coreset.gamma_upper_bound_compute_time
                    alg_args['dmax'] = dmax

                if (check_flag(setup['algorithms'][name],'use_dmin')):
                    print(f'\t\tcomputed dmin = {dmin}')
                    t = t + coreset.closest_pair_compute_time
                    alg_args['dmin'] = dmin
                
                # Check if the alg is to be run with a timeout
                if 'timeout' in setup['algorithms'][name]:
                    timeout = setup['algorithms'][name]['timeout']
                    print(f'\t\tUsing timeout of {timeout}')
                    # Check if the alg has timedout before
                    if timeout_dict[name]:
                        print('Timed out/Exception occured in previous iteration!')
                        continue
                    import gurobipy
                    try:
                        with time_limit(timeout):
                            runner = algorithms[setup['algorithms'][name]['alg']]
                            sol, div, t_alg = runner(gen, name, kimap, alg_args)
                            t = t + t_alg
                            print(f'\t\t***solution size = {len(sol)}***')
                            print(f'\t\tdiv = {div}')
                            print(f'\t\tt = {t}')
                            result_per_alg[name] = [len(alg_args['features']), dmax, dmin, len(sol), div, t]
                    except TimeoutException as e:
                        print("Timed out!")
                        timeout_dict[name] = True
                        alg_status.append(f'{name} timed out at k = {adj_k}')
                        continue
                    except gurobipy.GurobiError as gbe:
                        print(f'Gurobi Error - {gbe.message}')
                        timeout_dict[name] = True
                        alg_status.append(f'{name} gurobi errored at k = {adj_k}')
                        continue
                    except Exception as e:
                        print(f'Some exception occured = {e.message}')
                        timeout_dict[name] = True
                        alg_status.append(f'{name} exception occured at k = {adj_k}')
                        continue
                    result_per_alg[name] = [len(alg_args['features']), dmax, dmin, len(sol), div, t]

                # Else run without timeout
                else:
                    if timeout_dict[name]:
                        print('Timed out/Exception occured in previous iteration!')
                        continue
                    import gurobipy
                    runner = algorithms[setup['algorithms'][name]['alg']]
                    try:
                            sol, div, t_alg = runner(gen, name, kimap, alg_args)
                    except gurobipy.GurobiError as gbe:
                        print(f'Gurobi Error - {gbe.message}')
                        alg_status.append(f'{name} gurobi errored at k = {adj_k}')
                        timeout_dict[name] = True
                        continue
                    except Exception as e:
                        print(f'Some exception occured = {e.message}')
                        alg_status.append(f'{name} exception occured at k = {adj_k}')
                        timeout_dict[name] = True
                        continue
                    t = t + t_alg
                    print(f'\t\t***solution size = {len(sol)}***')
                    print(f'\t\tdiv = {div}')
                    print(f'\t\tt = {t}')
                    result_per_alg[name] = [len(alg_args['features']), dmax, dmin, len(sol), div, t]
                
                if not timeout_dict[name]:
                    from algorithms.utils import check_returned_kis
                    kis_delta = check_returned_kis(alg_args['colors'], kimap, sol)
                    color_results.append([dataset_name, name, adj_k, kis_delta])


                # End of algorithms loop

            observations.append(result_per_alg)
        # End of observations loop

        avgs = {}
        # Average out the observations
        for alg in setup['algorithms']:
            if timeout_dict[alg]:
                continue
            for i in range(0, setup['parameters']['observations']):
                observation = observations[i][alg]
                if alg not in avgs:
                    avgs[alg] = [observation]
                else:
                    avgs[alg].append(observation)
        for alg in avgs:
            if timeout_dict[alg]:
                continue
            avgs[alg] = np.mean(np.array(avgs[alg]), axis=0).tolist()
            if adj_k not in results_per_k_per_alg:
                results_per_k_per_alg[adj_k] = {alg: avgs[alg]}
            else:
                results_per_k_per_alg[adj_k][alg] = avgs[alg]

    # End of k loop
    print(results_per_k_per_alg)
    results[dataset_name] = {}
    # we can go back to normal "k" here
    # since we're iterating over the adj_k keys we added to the map
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
write_results(setup, results, color_results)
print(alg_status)