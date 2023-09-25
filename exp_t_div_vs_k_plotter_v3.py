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
plot_dir = result_file_dir + '/result_' + result_file_name
if not os.path.exists(plot_dir):
   os.mkdir(plot_dir)

with open(result_file_path, 'r') as json_file:
    data = json.load(json_file)
    results = data["results"]
    setup = data["setup"]

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