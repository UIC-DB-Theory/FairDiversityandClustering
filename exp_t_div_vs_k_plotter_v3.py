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

for dataset_name, dataset_results in results.items():
    plt.clf()

    # plot t vs k
    for alg,result in dataset_results.items():
        x = result["xs"]["k"]
        y =result["ys"]["runtime"]
        plt.plot(x,y, setup["algorithms"][alg]["color"], label=alg)

    plt.legend(title = f'runtime vs k - {dataset_name}', bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel("k")
    plt.ylabel("runtime (s)")
    plt.savefig(f'{plot_dir}/t_vs_k', dpi=300, bbox_inches='tight')
    plt.yscale("log")
    plt.ylabel("log(runtime)")
    plt.legend(title = f'log(runtime) vs k - {dataset_name}', bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(f'{plot_dir}/log_t_vs_k', dpi=300, bbox_inches='tight')
    plt.clf()

    # plot div vs k
    for alg,result in dataset_results.items():
        x = result["xs"]["k"]
        y =result["ys"]["diversity"]
        plt.plot(x,y, setup["algorithms"][alg]["color"], label=alg)

    plt.legend(title = f'diversity vs k - {dataset_name}', bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel("k")
    plt.ylabel("diversity")
    plt.savefig(f'{plot_dir}/diversity_vs_k', dpi=300, bbox_inches='tight')
    plt.clf()

    # plot div/t vs k
    for alg,result in dataset_results.items():
        x = result["xs"]["k"]
        y = result["ys"]["div-runtime"]
        plt.plot(x,y, setup["algorithms"][alg]["color"], label=alg)

    plt.legend(title = f'd/t vs k - {dataset_name}', bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel("k")
    plt.ylabel("div/t")
    plt.savefig(f'{plot_dir}/div_t_vs_k', dpi=300, bbox_inches='tight')
    plt.yscale("log")
    plt.ylabel("log(div/t)")
    plt.legend(title = f'log(d/t) vs k - {dataset_name}', bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(f'{plot_dir}/log_div_t_vs_k', dpi=300, bbox_inches='tight')
    plt.clf()
    
    # plot solution_size vs k
    for alg,result in dataset_results.items():
        x = result["xs"]["k"]
        y = result["ys"]["solution_size"]
        plt.plot(x,y, setup["algorithms"][alg]["color"], label=alg)

    plt.legend(title = f'solution_size vs k - {dataset_name}', bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel("k")
    plt.ylabel("solution size")
    plt.savefig(f'{plot_dir}/sol_size_vs_k', dpi=300, bbox_inches='tight')
    plt.clf()
    
    # plot data_size vs k
    for alg,result in dataset_results.items():
        x = result["xs"]["k"]
        y = result["ys"]["data_size"]
        plt.plot(x,y, setup["algorithms"][alg]["color"], label=alg)

    plt.legend(title = f'data_size vs k - {dataset_name}', bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel("k")
    plt.ylabel("data size")
    plt.savefig(f'{plot_dir}/data_size_vs_k', dpi=300, bbox_inches='tight')
    plt.clf()
    
# plot the delta of points returned per color
temp = {}
for result in color_results:
    dataset_name = result[0]
    alg_name = result[1]
    k_value = result[2]
    kis_delta = result[3]
    if dataset_name not in temp:
        temp[dataset_name] = {

            alg_name : {
                        "xs" : {
                            "k" : [k_value]
                            },
                        "ys" : {
                            c: [count] for c, count in kis_delta.items()
                            }
                        }
        }
    else:
        if alg_name not in temp[dataset_name]:
            temp[dataset_name][alg_name] = {
            "xs" : {
                "k" : [k_value]
                },
            "ys" : {
                c: [count] for c, count in kis_delta.items()
                }
            }
        else:
            temp[dataset_name][alg_name]["xs"]["k"].append(k_value)
            for c, count in kis_delta.items():
                temp[dataset_name][alg_name]["ys"][c].append(count)
    
# json_object = json.dumps(temp, indent=4)
# print(json_object)
def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter(x, y, s=20, color='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    return plt.scatter(rand_jitter(x), rand_jitter(y), s=s, color=color, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)

plot_dir = plot_dir + '/color_deltas'
if not os.path.exists(plot_dir):
   os.mkdir(plot_dir)
for dataset_name in temp:
    for alg_name in temp[dataset_name]:
        plt.clf()
        result_file_path = f'{plot_dir}/{dataset_name}_{alg_name}.png'
        x = temp[dataset_name][alg_name]["xs"]["k"]
        print(f'{alg_name} : {x}')
        for color_name in temp[dataset_name][alg_name]["ys"]:
            y = temp[dataset_name][alg_name]["ys"][color_name]
            print(f'\t\t{color_name} : {y}')
            import numpy as np
            jitter(x,y,color = (np.random.random(), np.random.random(), np.random.random()), label=color_name)
        plt.legend(title = f'color delta vs k - {dataset_name} - {alg_name}', bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.xlabel("k")
        plt.ylabel("color delta")
        plt.savefig(result_file_path, dpi=300, bbox_inches='tight')
        plt.clf()
        