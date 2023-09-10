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

plt.clf()
# t vs k
for alg,result in results.items():
    x = result["xs"]["k_values"]
    y =result["ys"]["runtimes"]
    plt.plot(x,y, alg_colors[alg], label=alg)

plt.legend(title = "runtime vs k - Adult", bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.xlabel("k")
plt.ylabel("runtime (s)")
plt.savefig("t_vs_k_adult", dpi=300, bbox_inches='tight')
plt.yscale("log")
plt.savefig("log_t_vs_k_adult", dpi=300, bbox_inches='tight')

plt.clf()
# d vs k
for alg,result in results.items():
    x = result["xs"]["k_values"]
    y = result["ys"]["diversity_values"]
    plt.plot(x,y, alg_colors[alg], label=alg)

plt.legend(title = "d vs k - Adult", bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.xlabel("k")
plt.ylabel("d")
plt.savefig("d_vs_k_adult", dpi=300, bbox_inches='tight')

plt.close()