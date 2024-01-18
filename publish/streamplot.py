"""
Usage: python3 streamplot_k.py /path/to/result/file
"""
import json
import sys
import os
import re
import csv
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
no_legend = False
only_legend = False
if len(sys.argv) == 3:
    if sys.argv[2] == '-nolegend':
        no_legend = True
    elif sys.argv[2] == '-onlylegend':
        only_legend = True

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
print('Plotting y vs k...')
def plot(y_key, x_key, ylogscale = False):

    set_y_label = y_key
    if y_key == 'posttime':
        set_y_label = 'Post Time (sec)'
    
    if y_key == 'streamtime':
        set_y_label = 'Stream Time/Elem (sec)'

    if y_key == 'totaltime':
        set_y_label = 'Total Time (sec)'
    

    plt.clf()

    # Each dataset gets 1 subplot
    fig = plt.figure(figsize=(4, 4))
    grid_specs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1])
    legend_handles = []

    alp = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    i = 0 
    for gs, dataset_name in zip(grid_specs, results):
        ax = plt.subplot(gs)
        for alg,result in results[dataset_name].items():
            print('Plotting alg: ', alg)
            x = result["xs"][x_key]
            y = result["ys"][y_key]
            color = setup["algorithms"][alg]["color"]
            marker = setup["algorithms"][alg]["marker"]
            legend_handles.append(mlines.Line2D([], [], color=color, marker=marker, linestyle='-',
                            markersize=10, label=alg))
            ax.plot(x,y, color=color, marker=marker)
            if ylogscale:
                ax.set_yscale('log')
            ax.set_xlabel(x_key, fontsize="16")
            # ax.set_title(f'({alp[i]}) {dataset_name}', y = -0.4, fontsize="20")
            ax.set_xticks([20, 40, 60, 80, 100])
            ax.tick_params(axis='both', which='major', labelsize=18)
        i += 1

    
    ax_legend = plt.subplot(grid_specs[0])
    ax_legend.set_ylabel(set_y_label, fontsize="16")

    if not no_legend:
        ax_legend.legend(
            handles=legend_handles[:len(setup["algorithms"])],
            ncol=len(setup["algorithms"]),
            loc='lower left', 
            bbox_to_anchor=(-0.2, 1.1),
            borderaxespad=0,
            fontsize="20"
    )
    # ax_legend.legend(
    #     handles=legend_handles[:len(setup["algorithms"])],
    #     ncol=len(setup["algorithms"]),
    #     loc='lower left', 
    #     bbox_to_anchor=(1.4, 1.1),
    #     borderaxespad=0,
    #     fontsize="20"
    # )
    plt.tight_layout(pad=2.0)
    plt.savefig(f'{plot_dir}/{y_key}_vs_{x_key}', dpi=300, bbox_inches='tight')

plot( "streamtime", "k", ylogscale = False)
plot( "posttime", "k", ylogscale = True)
plot( "totaltime", "k", ylogscale = True)
plot( "diversity", "k", ylogscale = False)
plt.close()