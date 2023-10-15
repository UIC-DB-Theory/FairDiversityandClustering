"""
Usage: python3 edit.py /path/to/result/file

Makes edits to algorithm labels, deletions, merging results etc

Examples:

python3 edit.py rename <alg_name> <new_name>
python3 edit.py delete <alg_name>
python3 edit.py oddk (only keeps the ks run at odd indices)

"""
import json
import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
######################################################################################
# Parse result file path
######################################################################################
result_file_path = sys.argv[1]
result = re.search(r'^(.+)\/([^\/]+)$', result_file_path)
result_file_dir = result.group(1)
result_file_name = result.group(2).split('.')[0]
print(result.group(1))
print(f'Editing file: {result_file_path}')
######################################################################################

######################################################################################
# Read file 
######################################################################################
with open(result_file_path, 'r') as json_file:
    data = json.load(json_file)
    results = data["results"]
    setup = data["setup"]
    color_results = data["color_results"]
    alg_status = data["alg_status"]
######################################################################################

######################################################################################
# Make changes
######################################################################################
def change_ks_20():
    alg = 'CHANGE'
    global results
    global color_results
    # Change in the vs k result section
    for dataset_name in results:
        print(dataset_name)
        for k, l in results[dataset_name][alg]["xs"].items():
            c = 0
            temp = []
            for i in l:
                if (c+1)%2 == 0:
                    temp.append(i)
                c = c + 1
            print(temp)
            results[dataset_name][alg]["xs"][k] = temp

        for k, l in results[dataset_name][alg]["ys"].items():
            c = 0
            temp = []
            for i in l:
                if (c+1)%2 == 0:
                    temp.append(i)
                c = c + 1
            print(temp)
            results[dataset_name][alg]["ys"][k] = temp
    temp = []
    for dataset_name in results:
        ks = results[dataset_name][alg]["xs"]['k']
        print(ks)
        # Change in the color results
        for color_result in color_results:
            if color_result[0] == dataset_name and color_result[2] in ks:
                temp.append(color_result)
    color_results = temp




def delete_marked():
    alg = 'DELETE'
    # Change in the setup section
    setup["algorithms"].pop(alg)

    # Change in the vs k result section
    for dataset_name in results:
        results[dataset_name].pop(alg)
    
    # Change in the color results
    # for i in color_results:
    #     if color_result[1] == alg:
    #         color_result[1] = alg_new

def alg_name_change(alg, alg_new):

    # Change in the setup section
    setup["algorithms"][alg_new] = setup["algorithms"].pop(alg)

    # Change in the vs k result section
    for dataset_name in results:
        if alg in results[dataset_name]:
            results[dataset_name][alg_new] = results[dataset_name].pop(alg)
    
    # Change in the color results
    for color_result in color_results:
        if color_result[1] == alg:
            color_result[1] = alg_new

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

######################################################################################

######################################################################################
# Display all algs
######################################################################################
alg_names = [ alg_name for alg_name in setup['algorithms']]
print('************************Algorithm Labels************************')
for alg_name in setup['algorithms']:
    print(alg_name)
print('****************************************************************')
alg_change = input("Enter name to change:")

alg_new = input("Enter new name:")
alg_name_change(alg_change, alg_new)

del_marked = input("Delete marker (y/n):")
if del_marked == 'y':
    delete_marked()

change_to_20 = input("Change to 20 marker (y/n):")
if change_to_20 == 'y':
    change_ks_20()



write_results(setup, results, color_results, alg_status)