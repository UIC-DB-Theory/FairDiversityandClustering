"""
Description:
    Measure runtime and diversity for varying k. Where k is the size of the output.
    For the Adult dataset
    
    Usage: python3 exp_t_div_vs_k_runner.py /path/to/setup/file.json
"""

import sys
import re
import os
import datetime
import json


# Parse setup file path
setup_file_path = sys.argv[1]
result = re.search(r'^(.+)\/([^\/]+)$', setup_file_path)
setup_file_dir = result.group(1)
setup_file_name = result.group(2)
print(result.group(1))

# Create result location -- in the same directory as the setup file
result_file_dir = setup_file_dir + '/result_' + setup_file_name
if not os.path.exists(result_file_dir):
    os.mkdir(result_file_dir)

# Create result file 
timestamp = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
result_file_name = 'result_' + timestamp + '.json'
result_file_path = result_file_dir + '/' + result_file_name

# Read the setup file
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
alg_experiments = {
    'SFDM-2' : lambda k, **kwargs: StreamFairDivMax2(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']), 
        epsilon = setup['algorithms']['SFDM-2']['epsilon'], 
        gammahigh = kwargs['dmax'], 
        gammalow = kwargs['dmin'], 
        normalize = False
    ),
    'FMMD-S' : lambda k, **kwargs: FMMDS(
        features = kwargs['features'],
        colors = kwargs['colors'],
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']),
        epsilon = setup['algorithms']['FMMD-S']['epsilon'],
        normalize = False
    ),
    'FairFlow' : lambda k, **kwargs : FairFlow(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']), 
        normalize = False
    ),
    'FairGreedyFlow' : lambda k, **kwargs : FairGreedyFlow(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']), 
        epsilon= setup['algorithms']['FairGreedyFlow']['epsilon'], 
        gammahigh=kwargs['dmax'], 
        gammalow = kwargs['dmin'], 
        normalize=False
    ),
    'FMMD-MWU' : lambda k, **kwargs : FMMDMWU(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']),
        gamma_upper = kwargs['dmax'],
        mwu_epsilon = setup['algorithms']['FMMD-MWU']['mwu_epsilon'],
        falloff_epsilon = setup['algorithms']['FMMD-MWU']['falloff_epsilon'],
        return_unadjusted = False
    ),
    'FMMD-LP' : lambda k, **kwargs : FMMDLP(
        features = kwargs['features'], 
        colors = kwargs['colors'],
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']), 
        upper_gamma = kwargs['dmax'],
        epsilon = setup['algorithms']['FMMD-LP']['epsilon'], 
    ),
    'FMMD-MWUS' : lambda k, **kwargs : FMMDMWUS(
        features = kwargs['features'], 
        colors = kwargs['colors'], 
        kis = buildKisMap(kwargs['colors'], k, setup['parameters']['buildkis_alpha']),
        gamma_upper=kwargs['dmax'],
        mwu_epsilon=setup['algorithms']['FMMD-MWU']['mwu_epsilon'],
        falloff_epsilon=setup['algorithms']['FMMD-MWU']['falloff_epsilon'],
        return_unadjusted=False
    ),
}