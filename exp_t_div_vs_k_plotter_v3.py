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

with open(result_file_path, 'r') as json_file:
    data = json.load(json_file)
    results = data["results"]
    setup = data["setup"]

import matplotlib.pyplot as plt

for dataset_name, dataset_results in results.items():
    plt.clf()
    for alg,alg_results in dataset_results.items():
        
        for x_label in alg_results['xs']:

            x_vals = alg_results['xs'][x_label]

            for y_label in alg_results['ys']:

                y_vals = alg_results['xs'][y_label]
                
                plt.plot(x_vals, y_vals, setup['algorithms'][alg]['color'], label = alg)
                plt.legend(title = f'{y_label} vs {x_label} - {dataset_name}', bbox_to_anchor=(1.05, 1.0), loc='upper left')
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.savefig(f'{result_file_dir}/{x_label}_vs_{y_label}_{dataset_name}', dpi=300, bbox_inches='tight')
                if(y_label == 'runtime'):
                    plt.yscale("log")
                    plt.savefig(f'{result_file_dir}/log_{x_label}_vs_{y_label}_{dataset_name}', dpi=300, bbox_inches='tight')
                plt.clf()
