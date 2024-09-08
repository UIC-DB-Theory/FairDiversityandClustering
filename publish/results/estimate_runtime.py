# Original: 72 hours
# Use cached results for FMMD-S because of no Gurobi license: 49 hours
# Use cached results for FMMD-S (No gurobi), SFDM-2 (e=.15) (Slow algorithm): 18 hours

files = [
    'macro_exp_equal.json',
]

total_runtime = 0
observations = 5

for file in files:
    import json
    with open(file, 'r') as json_file:
        print("", file)
        data = json.load(json_file)
        results = data["results"]
        for dataset in results:
            print("\t\t", dataset)
            for alg in results[dataset]:
                if 'FMMD-S' in alg:
                    continue
                if 'FairFlow' in alg:
                    continue
                if 'FairGreedyFlow' in alg:
                    continue
                if 'SFDM-2 (e=.15)' in alg:
                    continue
                if 'SFDM-2 (e=.75)' in alg:
                    continue
                print("\t", alg)
                if "runtime" in results[dataset][alg]["ys"]:
                    runtime = sum(results[dataset][alg]["ys"]["runtime"])
                    total_runtime += runtime

print('Est runtime: ', total_runtime*observations)
