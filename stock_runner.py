datainfo = {
            "data_dir": "./datasets/stocks",
            "color_fields": [
                "category"
            ],
            "feature_fields": [
                "r7d",
                "r14d",
                "r1m",
                "r3m",
                "r6m",
                "r8m"
            ],
            "normalize": True,
            "filter_unique": False
        }

from datasets.utils import read_dataset
dataset = read_dataset(
            datainfo['data_dir'],
            datainfo['feature_fields'],
            datainfo['color_fields'],
            datainfo['normalize'],
            datainfo['filter_unique']
        )

ofeatures = dataset['features']
ocolors = dataset['colors']

from algorithms.utils import buildKisMap
kimap = buildKisMap(dataset['colors'], 50, 0.0, equal_k_js=True)
adj_k = sum(kimap.values())

import numpy as np
gen = np.random.default_rng(seed=0)

from algorithms.coreset import Coreset_FMM
dimensions = len(datainfo["feature_fields"])
num_colors = len(dataset['points_per_color'])
coreset_size = num_colors * adj_k
coreset = Coreset_FMM(
                gen,
                ofeatures, 
                ocolors, 
                adj_k, 
                num_colors, 
                dimensions, 
                coreset_size)
core_features, core_colors = coreset.compute()
dmax = coreset.compute_gamma_upper_bound()
dmin = coreset.compute_closest_pair()

from fmmdmwu_nyoom import epsilon_falloff as FMMDMWU

sol, div, t_alg = FMMDMWU(
        gen=gen,
        features = ofeatures, 
        colors = ocolors, 
        kis = kimap,
        gamma_upper = dmax,
        mwu_epsilon = 0.75,
        falloff_epsilon = 0.15,
        percent_theoretical_limit = 0.7,
        return_unadjusted = False)

print(f'Diversity: {div}')
print(f'Solution: {sol}')

sol = list(sol)
all_stocks = []
import csv
csv_file_path = './datasets/stocks/stocks.data'
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        all_stocks.append(row[1])

for v in sol:
    print(all_stocks[v])