import json

results = {}
with open('experiment1.json', 'r') as json_file:
    data = json.load(json_file)
    results = data["results"]

# Plot the experiments
import matplotlib.pyplot as plt
alg_colors = {
    'SFDM-2' : 'tab:blue',
    'FMMD-S' : 'k',
    'FairFlow' : 'y-',
    'FairGreedyFlow' : 'tab:brown',
    'FMMD-MWU' : 'tab:green',
    'FMMD-LP' : 'tab:red',
}

# For each dataset
for dataset_name, dataset_results in results.items():
    
    plt.clf()
    # plot t vs k
    for alg,result in dataset_results.items():
        x = result["xs"]["k_values"]
        y =result["ys"]["runtimes"]
        plt.plot(x,y, alg_colors[alg], label=alg)

    plt.legend(title = f'runtime vs k - {dataset_name}', bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel("k")
    plt.ylabel("runtime (s)")
    plt.savefig(f't_vs_k_{dataset_name}', dpi=300, bbox_inches='tight')
    plt.yscale("log")
    plt.savefig(f'log_t_vs_k_{dataset_name}', dpi=300, bbox_inches='tight')

    plt.clf()
    # plot d vs k
    for alg,result in dataset_results.items():
        x = result["xs"]["k_values"]
        y = result["ys"]["diversity_values"]
        plt.plot(x,y, alg_colors[alg], label=alg)

    plt.legend(title = f'd vs k - {dataset_name}', bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel("k")
    plt.ylabel("d")
    plt.savefig(f'd_vs_k_{dataset_name}', dpi=300, bbox_inches='tight')

    plt.close()