"""
Usage: python3 plot_k.py /path/to/result/file
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

    plt.clf()

    # Each dataset gets 1 subplot
    fig = plt.figure(figsize=(25, 3))
    grid_specs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1])
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
            ax.set_title(f'({alp[i]}) {dataset_name}', y = -0.4, fontsize="20")
            ax.set_xticks([20, 40, 60, 80, 100])
            ax.tick_params(axis='both', which='major', labelsize=18)
        i += 1

    
    ax_legend = plt.subplot(grid_specs[0])
    ax_legend.set_ylabel(y_key, fontsize="16")
    ax_legend.legend(
        handles=legend_handles[:len(setup["algorithms"])],
        ncol=len(setup["algorithms"]),
        loc='lower left', 
        bbox_to_anchor=(0.3, 1.1),
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
    plt.savefig(f'{plot_dir}/{y_key}_vs_{x_key}', dpi=100, bbox_inches='tight')

plot( "runtime", "k", ylogscale = True)
plot( "diversity", "k", ylogscale = False)
plot( "div-runtime", "k", ylogscale = True)
plt.close()
######################################################################################



######################################################################################
# Plots of diversity vs time
######################################################################################
print('Plotting diversity vs time...')
def plot_diversity_time(index, ykey, xkey ,ylogscale = False, xlogscale = False):

    plt.clf()

    # Each dataset gets 1 subplot
    fig = plt.figure(figsize=(25, 3))
    grid_specs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1])
    legend_handles = []

    alp = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    i = 0
    
    for gs, dataset_name in zip(grid_specs, results):
        points = [] 
        ax = plt.subplot(gs)
        for alg,result in results[dataset_name].items():
            print('Plotting alg: ', alg)
            try: 
                x = [result["ys"][xkey][index]]
                y = [result["ys"][ykey][index]]
                kval =result["xs"]["k"][index]
                points.append((result["ys"][xkey][index],result["ys"][ykey][index]))
            except:
                continue
            color = setup["algorithms"][alg]["color"]
            marker = setup["algorithms"][alg]["marker"]
            legend_handles.append(mlines.Line2D([], [], color=color, marker=marker, linestyle='-',
                            markersize=10, label=alg))
            ax.plot(x,y, color=color, marker=marker)
            if ylogscale:
                ax.set_yscale('log')
            if xlogscale:
                ax.set_xscale('log')
            ax.set_xlabel(xkey, fontsize="16")
            ax.set_title(f'({alp[i]}) {dataset_name}', y = -0.4, fontsize="20")
            # ax.set_xticks([20, 40, 60, 80, 100])
            # ax.tick_params(axis='both', which='major', labelsize=18)
        # plot the skyline dotted line
        points.sort(key=lambda point: point[0])
        max_y = 0
        skyline_points = [(0,0)]
        suboptimal_points = []
        for point in points:
            x = point[0]
            y = point[1]
            if y > max_y:
                max_y = y
                skyline_points.append(point)
            else:
                suboptimal_points.append(points)
        

        # Initialize variables to keep track of the current maximum y-coordinate
        for i in range(len(skyline_points) - 1):
            x_start, y_start = skyline_points[i]
            x_end, y_end = skyline_points[i + 1]

            # Plot a line going right from the first point
            ax.plot([x_start, x_end], [y_start, y_start], linestyle='dotted', color='blue')

            # Plot a dotted line going up along the y-axis 
            ax.plot([x_end, x_end], [y_start, y_end], linestyle='dotted', color='blue')
        last_point = skyline_points[-1]
        _, x_max = ax.get_xlim()
        ax.plot([last_point[0], x_max], [last_point[1], last_point[1]], color='blue', linestyle='dotted')

        for point in skyline_points:
            ax.plot(point[0], point[1], marker='o', markersize=15, fillstyle='none', markeredgecolor='black', linestyle='None')
        i += 1

    
    ax_legend = plt.subplot(grid_specs[0])
    ax_legend.set_ylabel(ykey, fontsize="16")
    ax_legend.legend(
        handles=legend_handles[:len(setup["algorithms"])],
        ncol=len(setup["algorithms"]),
        loc='lower left', 
        bbox_to_anchor=(0.3, 1.1),
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
    plt.savefig(f'{plot_dir}/{ykey}_vs_{xkey}_{kval}', dpi=100, bbox_inches='tight')

plot_diversity_time(0, "diversity", "runtime", xlogscale = True)
plot_diversity_time(1, "diversity", "runtime", xlogscale = True)
plot_diversity_time(2, "diversity", "runtime", xlogscale = True)
plot_diversity_time(3, "diversity", "runtime", xlogscale = True)
plot_diversity_time(4, "diversity", "runtime", xlogscale = True)
plt.close()
######################################################################################

######################################################################################
# Plots of color results
######################################################################################
print('Plotting color ratios...')
plot_dir = result_file_dir + '/' + result_file_name + '/color_results'
if not os.path.exists(plot_dir):
   os.mkdir(plot_dir)
use_ratio = False
use_deltas = False
color_mappings = {}
color_mappings2 = {}
runner_ks = [k for k in range(setup["parameters"]["k"][0] ,setup["parameters"]["k"][1], setup["parameters"]["k"][2])]
data = {}
for color_result in color_results:
    dataset = color_result[0]
    algorithm = color_result[1]
    # if algorithm == 'MWU 0.3':
    #     print('added mwu')
    k = color_result[2]
    kis_delta = color_result[3]
    kis = color_result[4]
    kis_returned_ratio = {}
    kis_ratio = {}

    # Sanity check & color map
    k_calculated = 0
    for color, ki in kis.items():
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
        print('init: ', algorithm)
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
    only_odds = False
    for dataset in  data:
        print(data[dataset].keys())
        fig, ax = plt.subplots()
        if algorithm not in data[dataset]:
            print('not found:', algorithm)
            return
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
            ax.bar(ind, required_counts[color], width, label=color, bottom=bottom, color = color_mappings[dataset][color], edgecolor='black', linewidth = 2)
            bottom += required_counts[color]
        
        bottom = np.zeros(len(ks))
        ind = np.arange(len(ks))
        for color in returned_counts:
            ax.bar(ind + width + 0.05, returned_counts[color], width, bottom=bottom, color = color_mappings[dataset][color])
            bottom += returned_counts[color]
        ax.set_xticks(ind + width/2 + 0.025, ks)
        ax.set_title(f'{dataset}')
        ax.set_xlabel('k', fontsize="14")
        ax.set_ylabel('color ratios', fontsize="14")
        ax.tick_params(axis='both', which='major', labelsize=18)
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(
        #     handles[::-1], labels[::-1],
        #     title = f'Colors in {dataset}',
        #     loc='center left',
        #     bbox_to_anchor=(1, 0.5)
        # )
        plt.savefig(f'{plot_dir}/{dataset}_{algorithm}.png', dpi=100, bbox_inches='tight')
        plt.close()

def save_color_stats(algorithm):
    '''
    saves csv of format:
    k,color,required_points,returned points,miss%

    '''
    for dataset in data:
        filepath = f'{plot_dir}/{dataset}_{algorithm}.csv'
        header = ['k']
        csv_rows = []
        print(data[dataset].keys())
        if algorithm not in data[dataset]:
            print('not found:', algorithm)
            return
        ks = data[dataset][algorithm]['ks']
        returned_counts = data[dataset][algorithm]['returned_counts']
        required_counts = data[dataset][algorithm]['required_counts']
        for i in range(0, len(ks)):
            k = ks[i]
            csv_row = [k]
            header = ['k']
            for color in required_counts:
                header.append(color)
                required_count = required_counts[color][i]
                returned_count = returned_counts[color][i]
                delta = required_count - returned_count
                csv_row.append(f'{delta}/{required_count}')
            csv_rows.append(csv_row)
    
        # clear file first
        open(filepath, 'w').close()

        with open(filepath, 'w+') as csvfile:  

            # creating a csv writer object  
            csvwriter = csv.writer(csvfile)  

            csvwriter.writerow(header)  
                
            # writing the fields  
            csvwriter.writerows(csv_rows)  


def generate_colors(n):
    if n > 14:
        return []
    return [
        '#2f4f4f',
        '#228b22',
        '#7f0000',
        '#4b0082',
        '#ff8c00',
        '#ffff00',
        '#deb887',
        '#00ff00',
        '#00bfff',
        '#0000ff',
        '#ff00ff',
        '#dda0dd',
        '#ff1493',
        '#7fffd4'
    ]



color_mappings = {
    dataset_name : {
        color_d : color_m for color_d, color_m in zip(setup['datasets'][dataset_name]['points_per_color'], generate_colors(len(setup['datasets'][dataset_name]['points_per_color'])))
    } for dataset_name in setup['datasets']

}

for alg in setup['algorithms']:
    plt.close()
    # plot_color_results(alg)
    save_color_stats(alg)
