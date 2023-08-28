from collections import defaultdict
import matplotlib.pyplot as plt

import fdmalgs
import utils
import lpsolve
import multweights_nyoom as multweights
from coreset import Coreset_FMM
import time

import numpy as np
def color(alg_name):
    '''
    Returns the color used for plotting the graph of the given algorithm
    
    alg_name - name of the algorithm
    '''
    if alg_name == "FMMD-LP":
        # Blue
        return 'tab:blue'
    elif alg_name == "FMMD-MWU": 
        # Yellow
        return 'y-'
    elif alg_name == "FMMD-S":
        # Red
        return 'tab:red'
    elif alg_name == "FairFlow":
        # Black
        return 'k'
    elif alg_name == "FairGreedyFlow":
        # Purple
        return 'tab:purple'
    elif alg_name == "SFDM-2":
        # Green
        return 'tab:green'
    # elif alg_name == "fairgreedyflow":
    #     # Cyan
    #     return 'tab:cyan'
    # elif alg_name == "scalable_fmmd_modified_greedy":
    #     # Brown
    #     return 'tab:brown'

           
if __name__ == '__main__':
    # setup
    # File fields
    allFields = [
        "age",
        "workclass",
        "fnlwgt",  # what on earth is this one?
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "yearly-income",
    ]

    # fields we care about for parsing
    color_field = ['race', 'sex']
    feature_fields = {'age', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt', 'education-num'}

    colors, features = utils.read_CSV("./datasets/ads/adult.data", allFields, color_field, '_', feature_fields)
    assert (len(colors) == len(features))

    # ALGORITHMS TO TEST
    # we wrap each one so it has the same signature
    # various "tuning" parameters are set here
    # TODO SFDM2 FFMD-S
    algs = {
        'FMMD-LP':
            lambda fs, cs, kis, gamma_upper: lpsolve.epsilon_falloff(
                features=fs,
                colors=cs,
                upper_gamma=gamma_upper,
                kis=kis,
                epsilon=0.15,
            ),
        'FMMD-MWU':
            lambda fs, cs, kis, gamma_upper: multweights.epsilon_falloff(
                features=fs,
                colors=cs,
                kis=kis,
                gamma_upper=gamma_upper,
                mwu_epsilon=0.75,
                falloff_epsilon=0.15,
                return_unadjusted=False,
            ),
        'FairFlow':
            lambda fs, cs, kis, _: fdmalgs.FairFlowWrapped(
                features=fs,
                colors=cs,
                kis=kis,
                normalize=False,
            ),
        'FairGreedyFlow':
            lambda fs, cs, kis, _: fdmalgs.FairGreedyFlowWrapped(
                features=fs,
                colors=cs,
                kis=kis,
                epsilon=0.15,
                # experiments used fixed, pre-supplied values?
                gammahigh=3.43,
                gammalow=1.37,
                normalize=False,
            ),
        'FMMD-S':
            lambda fs, cs, kis, gamma_upper: fdmalgs.FMMDSWrapped(
                features=fs,
                colors=cs,
                kis=kis,
                epsilon=0.15,
                normalize=False,
            ),
        'SFDM-2':
            lambda fs, cs, kis, gamma_upper: fdmalgs.FairGreedyFlowWrapped(
                features=fs,
                colors=cs,
                kis=kis,
                epsilon=0.15,
                # experiments used fixed, pre-supplied values?
                gammahigh=3.43,
                gammalow=1.37,
                normalize=False,
            ),
        }

    # run some tests!
    results = defaultdict(list)

    # first for the proper 100
    for k in range(25, 351, 50):
        # compute coreset of size
        coreset_size = 10 * k
        # all colors made by combining values in color_fields
        color_names = np.unique(colors)

        # "normalize" features
        # Should happen before coreset construction
        means = features.mean(axis=0)
        devs = features.std(axis=0)
        features = (features - means) / devs

        d = len(feature_fields)
        m = len(color_names)
        coreset_time = 0

        coreset = Coreset_FMM(features, colors, k, m, d, coreset_size)
        # can't overwrite our original dataset
        core_features, core_colors = coreset.compute()

        # starting estimate (used by some algorithms)
        upper_gamma = coreset.compute_gamma_upper_bound()

        # ki generation
        kis = utils.buildKisMap(colors, k, 0.1)

        # real K value is the sum of the ki values
        k_rounded = sum(kis.values())

        # run every algorithm on this set
        for alg, runner in algs.items():
            print(f'Running {alg} for {k_rounded}...')
            rep_div_avg = 0
            calc_div_avg = 0
            time_avg = 0
            num_runs = 1
            for i in range(0, num_runs):
                if alg == 'FMMD-MWU' or alg == 'FMMD-LP':
                    sol, div, algtime = runner(core_features, core_colors, kis, upper_gamma)
                    algtime = algtime + coreset.coreset_compute_time
                    print(f'coreset compute time: {coreset.coreset_compute_time}')
                    rep_div_avg += div
                    # calc_div_avg += utils.compute_maxmin_diversity(core_features[sol])
                    time_avg += algtime
                else:
                    sol, div, algtime = runner(features, colors, kis, upper_gamma)
                    rep_div_avg += div
                    # calc_div_avg += utils.compute_maxmin_diversity(core_features[sol])
                    time_avg += algtime
            rep_div_avg = rep_div_avg/num_runs
            # calc_div_avg = calc_div_avg/num_runs
            time_avg = time_avg/num_runs
            print(f'{alg} finished in {time_avg}! (diversity of {rep_div_avg}, calculated diversity of {calc_div_avg})')
            
            results[alg].append((k_rounded, rep_div_avg, calc_div_avg, time_avg))


    data_per_alg_t_vs_k = {}
    data_per_alg_rd_vs_k = {}
    data_per_alg_cd_vs_k = {}
    print('\n\nFINAL RESULTS:')

    for alg, resList in results.items():
        print(f'********{alg}********')
        data_per_alg_t_vs_k[alg] = {"y":[], "x": []}
        data_per_alg_rd_vs_k[alg] = {"y":[], "x": []}
        data_per_alg_cd_vs_k[alg] = {"y":[], "x": []}
        for k, rd, cd, t in resList:
            print(f'{k}\t{rd}\t{cd}\t{t}')
            data_per_alg_t_vs_k[alg]["y"].append(t)
            data_per_alg_t_vs_k[alg]["x"].append(k)
            data_per_alg_rd_vs_k[alg]["y"].append(rd)
            data_per_alg_rd_vs_k[alg]["x"].append(k)
            data_per_alg_cd_vs_k[alg]["y"].append(cd)
            data_per_alg_cd_vs_k[alg]["x"].append(k)

        print('\n\n', end='')
    


    # Plot the graph t vs k
    plt.clf()
    for alg in data_per_alg_t_vs_k:
        plt.plot(data_per_alg_t_vs_k[alg]["x"], data_per_alg_t_vs_k[alg]["y"], color(alg), label=alg)
    
    plt.yscale("log")
    plt.legend(title = "time vs k - Adult Full", bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel("k")
    plt.ylabel("Time (s)")
    plt.savefig("t_vs_k_adult_full", dpi=300, bbox_inches='tight')

    # Plot the graph reported d vs k 
    plt.clf()
    for alg in data_per_alg_rd_vs_k:
        plt.plot(data_per_alg_rd_vs_k[alg]["x"], data_per_alg_rd_vs_k[alg]["y"], color(alg), label=alg)
    
    # plt.yscale("log")
    plt.legend(title = "d(reported) vs k - Adult Full", bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel("k")
    plt.ylabel("d")
    plt.savefig("d_reported_vs_k_adult_full", dpi=300, bbox_inches='tight')

    plt.close()

    # Plot the graph calculated d vs k 
    # plt.clf()
    # for alg in data_per_alg_rd_vs_k:
    #     plt.plot(data_per_alg_rd_vs_k[alg]["x"], data_per_alg_rd_vs_k[alg]["y"], color(alg), label=alg)
    
    # # plt.yscale("log")
    # plt.legend(title = "d(calcuated) vs k - Adult Full", bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # plt.xlabel("k")
    # plt.ylabel("d")
    # plt.savefig("d_calcualted_vs_k_adult_full", dpi=300, bbox_inches='tight')

    # plt.close()


