"""
Usage: python3 plot_k.py /path/to/result/file
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
print(f'Plotting from file: {result_file_path}')
######################################################################################

######################################################################################
# Create result location -- in the same directory as the result file
######################################################################################
plot_dir = result_file_dir + '/' + result_file_name
if not os.path.exists(plot_dir):
   os.mkdir(plot_dir)
######################################################################################

######################################################################################
# Read file 
######################################################################################
with open(result_file_path, 'r') as json_file:
    data = json.load(json_file)
    results = data["results"]
    setup = data["setup"]
    color_results = data["color_results"]
######################################################################################


######################################################################################
# Plots for y vs k
######################################################################################

def plot(y_key, x_key, ylogscale = False):

    plt.clf()

    # Each dataset gets 1 subplot
    fig = plt.figure(figsize=(20, 5))
    grid_specs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1])
    legend_handles = []


    for gs, dataset_name in zip(grid_specs, results):
        ax = plt.subplot(gs)
        for alg,result in results[dataset_name].items():
            x = result["xs"][x_key]
            y = result["ys"][y_key]
            color = setup["algorithms"][alg]["color"]
            marker = setup["algorithms"][alg]["marker"]
            legend_handles.append(mlines.Line2D([], [], color=color, marker=marker, linestyle='-',
                            markersize=10, label=alg))
            ax.plot(x,y, color=color, marker=marker)
            if ylogscale:
                ax.set_yscale('log')
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_title(f'{dataset_name}')

    
    ax_legend = plt.subplot(grid_specs[0])
    ax_legend.legend(
        title = f't vs k',  
        handles=legend_handles[:len(setup["algorithms"])],
        ncol=len(setup["algorithms"]),
        loc='lower left', 
        bbox_to_anchor=(0.9, 1.1),borderaxespad=0
    )
    plt.savefig(f'{plot_dir}/{y_key}_vs_{x_key}', dpi=300)

plot( "runtime", "k", ylogscale = True)
plot( "diversity", "k", ylogscale = False)
plot( "div-runtime", "k", ylogscale = True)
plt.close()
######################################################################################

######################################################################################
# Plots of color results
######################################################################################
plot_dir = result_file_dir + '/' + result_file_name + '/color_results'
if not os.path.exists(plot_dir):
   os.mkdir(plot_dir)
use_ratio = True
use_deltas = False
color_mappings = {}
color_mappings2 = {}
runner_ks = [k for k in range(setup["parameters"]["k"][0] ,setup["parameters"]["k"][1], setup["parameters"]["k"][2])]
data = {}
for color_result in color_results:
    dataset = color_result[0]
    algorithm = color_result[1]
    k = color_result[2]
    kis_delta = color_result[3]
    kis = color_result[4]
    kis_returned_ratio = {}
    kis_ratio = {}

    if dataset not in color_mappings:
        color_mappings[dataset] = {}
        color_mappings2[dataset] = {}

    # Sanity check & color map
    k_calculated = 0
    for color, ki in kis.items():
        color_mappings[dataset][color] = (np.random.random(), np.random.random(), np.random.random())
        color_mappings2[dataset][color] = (np.random.random(), np.random.random(), np.random.random())
        k_calculated = k_calculated + int(ki)
    assert k == k_calculated

    # Calculate kis in ratio
    k_returned = 0
    for color, ki in kis.items():
        if use_ratio:
            kis_ratio[color] = ki/k
        elif use_deltas:
            kis_ratio[color] = ki
        else:
            kis_ratio[color] = ki
        if color in kis_delta:
            kis_returned_ratio[color] = (ki + kis_delta[color])
            k_returned += kis_returned_ratio[color]
            if use_deltas:
                kis_returned_ratio[color] = -kis_delta[color]

        else:
            kis_returned_ratio[color] = ki
            if use_deltas:
                kis_returned_ratio[color] = -kis_delta[color]
    if use_ratio:
        sum1 = 0
        sum2 = 0
        for color in kis_returned_ratio:
            kis_returned_ratio[color] = kis_returned_ratio[color]/k_returned
            sum1 += kis_returned_ratio[color]
            sum2 += kis_ratio[color]
    
    if use_deltas:
        kis_ratio = kis_returned_ratio

    # Intitalizaton
    if dataset not in data:
        data[dataset] = {
            algorithm : {
                'ks' : [],
                'returned_counts' : {c : [] for c in kis},
                'required_counts' : {c : [] for c in kis}
            }
        }
    if algorithm not in data[dataset]:
        data[dataset][algorithm] = {
            'ks' : [],
            'returned_counts' : {c : [] for c in kis},
            'required_counts' : {c : [] for c in kis}
        }
    
    # Add the first observations by appending only when a k is seen for the first time
    if k not in data[dataset][algorithm]['ks']:
        data[dataset][algorithm]['ks'].append(k)
        for color in kis:
            data[dataset][algorithm]['returned_counts'][color].append(kis_returned_ratio[color])
            data[dataset][algorithm]['required_counts'][color].append(kis_ratio[color])



def plot_color_results(algorithm):
    plt.clf()
    width = 0.4
    only_odds = True
    for dataset in  data:
        fig, ax = plt.subplots()
        ks = data[dataset][algorithm]['ks']
        returned_counts = data[dataset][algorithm]['returned_counts']
        required_counts = data[dataset][algorithm]['required_counts']
        if only_odds:
                temp1 = []
                max_ind= len(ks)
                for i in range(0, max_ind):
                    if (i+1)%2 == 0:
                        temp1.append(ks[i])
                ks = temp1
                for color in required_counts:
                    prev1 = required_counts[color]
                    prev2 = returned_counts[color]
                    new1 = []
                    new2 = []
                    for i in range(0, max_ind):
                        if (i+1)%2 == 0:
                            new1.append(prev1[i])
                            new2.append(prev2[i])
                    required_counts[color] = new1
                    returned_counts[color] = new2

        bottom = np.zeros(len(ks))
        ind = np.arange(len(ks))
        for color in required_counts:
            ax.bar(ind, required_counts[color], width, label=color, bottom=bottom, color = color_mappings[dataset][color])
            bottom += required_counts[color]
        
        bottom = np.zeros(len(ks))
        ind = np.arange(len(ks))
        for color in returned_counts:
            ax.bar(ind + width + 0.05, returned_counts[color], width, bottom=bottom, color = color_mappings[dataset][color])
            bottom += returned_counts[color]
        ax.set_xticks(ind + width/2 + 0.025, ks)
        ax.set_title(f'{dataset}')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[::-1], labels[::-1],
            title = f'Colors in {dataset}',
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )
        plt.savefig(f'{plot_dir}/{dataset}_{algorithm}.png', dpi=300, bbox_inches='tight')
        plt.close()

for alg in setup['algorithms']:
    plot_color_results(alg)
