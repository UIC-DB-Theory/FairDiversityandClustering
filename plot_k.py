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
            ax.plot([x_start, x_end], [y_start, y_start], linestyle='dotted', color='black')

            # Plot a dotted line going up along the y-axis 
            ax.plot([x_end, x_end], [y_start, y_end], linestyle='dotted', color='black')
        last_point = skyline_points[-1]
        _, x_max = ax.get_xlim()
        ax.plot([last_point[0], x_max], [last_point[1], last_point[1]], color='black', linestyle='dotted')

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
# Genreate missing points csv
######################################################################################
print('Missing points csv')
plot_dir = result_file_dir + '/' + result_file_name + '/color_results'
if not os.path.exists(plot_dir):
   os.mkdir(plot_dir)



'''
This keeps track of the avg missing points\
{
    'dataset-1': {
    `               'alg-1' : {
                                'k': [],
                                'color1' : [[missing], [required]],
                                'color2' : [...],
                                .
                                .
                            },
                    'alg-2' : {
                                .
                                .
                            },
    
                },
    'dataset-2': {
                    .
                    .
                    .
                },
    .
    .
    .

}
'''
avg_color_results = {}

num_runs = setup['parameters']['observations']
for color_result in color_results:
    dataset = color_result[0]
    algorithm = color_result[1]
    k_value = color_result[2]
    missing_counts = color_result[3]
    required_counts = color_result[4]

    
    if dataset not in avg_color_results:
        # Add the counts or new dataset
        avg_color_results[dataset] = {}
        avg_color_results[dataset][algorithm] = {}
        avg_color_results[dataset][algorithm][k_value] = {
            c : [(missing_counts[c])/num_runs, required_counts[c]] for c in required_counts
        }
    
    elif algorithm not in avg_color_results[dataset]:
        # Add the counts for new algorithm
        avg_color_results[dataset][algorithm] = {}
        avg_color_results[dataset][algorithm][k_value] = {
            c : [(missing_counts[c])/num_runs, required_counts[c]] for c in required_counts
        }

    elif k_value not in avg_color_results[dataset][algorithm]:
        # Add the counts for new k valu[e
        avg_color_results[dataset][algorithm][k_value] = {
            c : [(missing_counts[c])/num_runs, required_counts[c]] for c in required_counts
        }
    
    else:
        # Average the counts
        for c in avg_color_results[dataset][algorithm][k_value]:
            counts = avg_color_results[dataset][algorithm][k_value][c]
            counts[0] = counts[0] + (-1*missing_counts[c])/num_runs

for dataset in avg_color_results:
    for algorithm in avg_color_results[dataset]:
        filepath = f'{plot_dir}/{dataset}_{algorithm}.csv'
        header = ['k', 'required points per color']
        csv_rows = []
        for k in avg_color_results[dataset][algorithm]:
            row = [k]
            required_points_per_color = 0
            temp = []
            for color in avg_color_results[dataset][algorithm][k]:
                if len(header) != 2 + len(avg_color_results[dataset][algorithm][k]): 
                    header.append(color)
                required_points_per_color = avg_color_results[dataset][algorithm][k][color][1]
                miss = avg_color_results[dataset][algorithm][k][color][0]
                temp.append(f'{miss}')
            row.append(required_points_per_color)
            row = row + temp
            csv_rows.append(row)

        open(filepath, 'w').close()

        with open(filepath, 'w+') as csvfile:  

            # creating a csv writer object  
            csvwriter = csv.writer(csvfile)  

            csvwriter.writerow(header)  

            # writing the fields  
            csvwriter.writerows(csv_rows)