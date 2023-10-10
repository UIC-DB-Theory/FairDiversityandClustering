"""
Usage: python3 exp_t_div_vs_k_plotter.py /path/to/result/file.result
"""
import json
import sys
import os
import re

# Parse setup file path
result_file_path = sys.argv[1]
result = re.search(r'^(.+)\/([^\/]+)$', result_file_path)
result_file_dir = result.group(1)
result_file_name = result.group(2).split('.')[0]
print(result.group(1))
print(f'Plotting from file: {result_file_path}')

# Create result location -- in the same directory as the setup file
plot_dir = result_file_dir + '/' + result_file_name
if not os.path.exists(plot_dir):
   os.mkdir(plot_dir)

with open(result_file_path, 'r') as json_file:
    data = json.load(json_file)
    results = data["results"]
    setup = data["setup"]
    color_results = data["color_results"]

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.lines as mlines

def plot(y_key, x_key, ylogscale = False):

    plt.clf()

    # Each dataset gets 1 subplot
    fig = plt.figure(figsize=(15, 10))
    grid_specs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
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

    ax_legend = plt.subplot(grid_specs[5])
    ax_legend.axis('off')  # Hide the empty subplot
    ax_legend.legend(title = f'log(t) vs k',  handles=legend_handles[:len(setup["algorithms"])], bbox_to_anchor=[0.5, 0.5], loc='center',)
    plt.savefig(f'{plot_dir}/{y_key}_vs_{x_key}', dpi=300)

plot( "runtime", "k", ylogscale = True)
plot( "diversity", "k", ylogscale = False)
plot( "div-runtime", "k", ylogscale = True)


#TODO: Plot Color deltas below
color_results