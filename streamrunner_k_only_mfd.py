"""
Description:
    Measure runtime and diversity for varying k. Where k is the size of the output.
    For the Adult dataset
    
    Usage: python3 exp_t_div_vs_k_runner.py /path/to/setup/file.json
"""
cached_algs = ['SFDM-2 (e=.75)', 'SFDM-2 (e=.15)']

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
from fmmdmwu_stream import fmmdmwu_stream as SMWUFD
from algorithms.utils import buildKisMap
from algorithms.utils import calculate_dmin_dmax

dmin_dmax = {}

# Lambdas for running experiments
algorithms = {
    'SFDM-2' : lambda gen, name, kis, kwargs: StreamFairDivMax2(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = kis, 
        epsilon = setup['algorithms'][name]['epsilon'], 
        gammahigh = kwargs['dmax'], 
        gammalow = kwargs['dmin'], 
        normalize = False,
        streamtimes=True
    ),
    'SMWUFD' : lambda gen, name, kis, kwargs: SMWUFD(
        gen=gen,
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = kis,
        gamma_upper = 0, # We calcualte this on the streamed coreset
        mwu_epsilon = setup['algorithms'][name]['mwu_epsilon'],
        falloff_epsilon = setup['algorithms'][name]['falloff_epsilon'],
        percent_theoretical_limit = setup['algorithms'][name]['percent_theoretical_limit'],
        return_unadjusted = False,
        streamtimes=True
    )
    
}

def check_flag(struct, flag):
    
    if flag in struct:
        return struct[flag]
    else:
        return False

def write_results(setup, results, color_results, alg_status):

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
        "alg_status" : alg_status,
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

    # reset the timeout dict for each dataset
    for alg in setup['algorithms']:
        timeout_dict[alg] = False

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
        write_results(setup, results, color_results, alg_status)
        features = dataset['features']
        colors = dataset['colors']

        if dataset_name not in dmin_dmax:
            print('Calculating dataset dmin & dmax for', dataset_name)
            tmp_features = copy.deepcopy(features)
            dmin_full, dmax_full = calculate_dmin_dmax(tmp_features)
            print('\t\tdmin', dmin_full)
            print('\t\tdmax', dmax_full)
            dmin_dmax[dataset_name] = [dmin_full, dmax_full]

        # one kis' map to ask for
        kimap = buildKisMap(dataset['colors'], k, setup['parameters']['buildkis_alpha'], equal_k_js=check_flag(setup['parameters'],'buildkis_equal_k_js'))
        adj_k = sum(kimap.values()) # the actual number of points we asked for

        print(f'***************************************')
        print(f'\t***Running for k = {adj_k}, {k}...')
        print(json.dumps(kimap, indent=4))
        print(f'***************************************')

        for obs in range(0, setup['parameters']['observations']):

            print(f'\n\nObservation number = {obs + 1}')

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

            print('********Offline Param stats**********')
            print(f'\t dmin = {dmin}')
            print(f'\t dmax = {dmax}')
            core_stats = {}
            for i in range(0, len(core_features)):
                if core_colors[i] in core_stats:
                    core_stats[core_colors[i]] += 1
                else:
                    core_stats[core_colors[i]] = 1
            print(f'\t core stats:')
            for iter in core_stats:
                print(f'\t\t {iter} : {core_stats[iter]}')
            print('********Offline Param stats**********')

            # Use dmin and dmax of full dataset instead
            dmin = dmin_dmax[dataset_name][0]
            dmax = dmin_dmax[dataset_name][1]

            result_per_alg = {}
            for name in setup['algorithms']:

                if name in cached_algs :
                    print(f'\t\t\tSkipping {name} will use cached results...')
                    continue

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
                    print(f'\t\tcompute time  = {coreset.coreset_compute_time}')
                    t = t + coreset.coreset_compute_time
                    alg_args['features'] = copy.deepcopy(core_features)
                    alg_args['colors'] = copy.deepcopy(core_colors)

                if (check_flag(setup['algorithms'][name],'use_dmax')):
                    print(f'\t\tcomputed dmax = {dmax}')
                    print(f'\t\tcompute time  = {coreset.gamma_upper_bound_compute_time}')
                    t = t + coreset.gamma_upper_bound_compute_time
                    alg_args['dmax'] = dmax

                if (check_flag(setup['algorithms'][name],'use_dmin')):
                    print(f'\t\tcomputed dmin = {dmin}')
                    print(f'\t\tcompute time  = {coreset.closest_pair_compute_time}')
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

                            # NOTE: For streaming setting additional time states are returned
                            # sol, div, t_alg = runner(gen, name, kimap, alg_args)
                            sol, div, times = runner(gen, name, kimap, alg_args)

                            stream_time = times[0]
                            post_time = times[1]
                            total_time = times[2]

                            # We ignore dmin & dmax calculation times
                            # t = t + t_alg

                            print(f'\t\t***solution size = {len(sol)}***')
                            print(f'\t\tdiv = {div}')
                            print(f'\t\tstream time = {stream_time}')
                            print(f'\t\tpost time = {post_time}')
                            print(f'\t\ttotal time = {total_time}')
                            # print(f'\t\tt = {t}')
                            result_per_alg[name] = [len(alg_args['features']), dmax, dmin, len(sol), div, stream_time, post_time, total_time]
                    except TimeoutException as e:
                        print("Timed out!")
                        timeout_dict[name] = True
                        alg_status.append([f'{name} timed out at k = {adj_k}'])
                        continue
                    except gurobipy.GurobiError as gbe:
                        print(f'Gurobi Error - {gbe.message}')
                        timeout_dict[name] = True
                        alg_status.append([f'{name} gurobi errored at k = {adj_k}', e.message])
                        continue
                    except Exception as e:
                        print(f'Some exception occured = {e.message}')
                        timeout_dict[name] = True
                        alg_status.append([f'{name} exception occured at k = {adj_k}', e.message])
                        continue
                    result_per_alg[name] = [len(alg_args['features']), dmax, dmin, len(sol), div, stream_time, post_time, total_time]

                # Else run without timeout
                else:
                    if timeout_dict[name]:
                        print('Timed out/Exception occured in previous iteration!')
                        continue
                    import gurobipy
                    runner = algorithms[setup['algorithms'][name]['alg']]
                    try:
                            #sol, div, t_alg = runner(gen, name, kimap, alg_args)
                            sol, div, times = runner(gen, name, kimap, alg_args)
                            stream_time = times[0]
                            post_time = times[1]
                            total_time = times[2]

                    except gurobipy.GurobiError as gbe:
                        print(f'Gurobi Error - {gbe.message}')
                        alg_status.append([f'{name} gurobi errored at k = {adj_k}', e.message])
                        timeout_dict[name] = True
                        continue
                    except Exception as e:
                        print(f'Some exception occured = {e.message}')
                        alg_status.append([f'{name} exception occured at k = {adj_k}', e.message])
                        timeout_dict[name] = True
                        continue
                    # t = t + t_alg
                    print(f'\t\t***solution size = {len(sol)}***')
                    print(f'\t\tdiv = {div}')
                    print(f'\t\tstream time = {stream_time}')
                    print(f'\t\tpost time = {post_time}')
                    print(f'\t\ttotal time = {total_time}')
                    result_per_alg[name] = [len(alg_args['features']), dmax, dmin, len(sol), div, stream_time, post_time, total_time]
                
                # Streaming setting does not support this as coresets are 
                # constructed individually by each algorithm, indices are
                # returned by some are relative to coreset.
                # if not timeout_dict[name]:
                #     from algorithms.utils import check_returned_kis
                #     kis_delta = check_returned_kis(alg_args['colors'], kimap, sol)
                #     color_results.append([dataset_name, name, adj_k, kis_delta, kimap])

                # End of algorithms loop

            observations.append(result_per_alg)
        # End of observations loop

        avgs = {}
        # Average out the observations
        for alg in setup['algorithms']:
            if alg in cached_algs :
                continue
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
                        'streamtime' : [results_per_k_per_alg[k][alg][5]],
                        'posttime' : [results_per_k_per_alg[k][alg][6]],
                        'totaltime' : [results_per_k_per_alg[k][alg][7]]
                    }
                }
            else:
                results[dataset_name][alg]['xs']['k'].append(k)
                results[dataset_name][alg]['ys']['data_size'].append(results_per_k_per_alg[k][alg][0])
                results[dataset_name][alg]['ys']['dmax'].append(results_per_k_per_alg[k][alg][1])
                results[dataset_name][alg]['ys']['dmin'].append(results_per_k_per_alg[k][alg][2])
                results[dataset_name][alg]['ys']['solution_size'].append(results_per_k_per_alg[k][alg][3])
                results[dataset_name][alg]['ys']['diversity'].append(results_per_k_per_alg[k][alg][4])
                results[dataset_name][alg]['ys']['streamtime'].append(results_per_k_per_alg[k][alg][5])
                results[dataset_name][alg]['ys']['posttime'].append(results_per_k_per_alg[k][alg][6])
                results[dataset_name][alg]['ys']['totaltime'].append(results_per_k_per_alg[k][alg][7])


# Add cached results for non MFD algs
for alg in setup["algorithms"]:
    if alg in cached_algs :
        print('Adding cached results for', alg)
        cached_result_file_path = re.sub(r'setup', 'results', setup_file_path)
        print('Using file', cached_result_file_path)
        cached_result = {}
        with open(cached_result_file_path, 'r') as json_file:
            cached_result = json.load(json_file)
        for dataset in results:
            results[dataset][alg] = cached_result['results'][dataset][alg]

# End of dataset loop
write_results(setup, results, color_results, alg_status)
print(alg_status)