from collections import defaultdict

import fdmalgs
import utils
import lpsolve
from coreset import Coreset_FMM

import numpy as np

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
    algs = {
        'lp':
            lambda fs, cs, kis, gamma_upper: lpsolve.epsilon_falloff(
                features=fs,
                colors=cs,
                upper_gamma=gamma_upper,
                kis=kis,
                epsilon=0.1,
            ),
        'fairflow':
            lambda fs, cs, kis, gamma_upper: fdmalgs.FairFlowWrapped(
                features=fs,
                colors=cs,
                kis=kis,
                normalize=False,
            )
        }

    # run some tests!
    results = defaultdict(list)

    # first for the proper 100
    for k in range(10, 201, 5):
        # compute coreset of size
        coreset_size = 100 * k
        # all colors made by combining values in color_fields
        color_names = np.unique(colors)

        # "normalize" features
        # Should happen before coreset construction
        means = features.mean(axis=0)
        devs = features.std(axis=0)
        features = (features - means) / devs

        d = len(feature_fields)
        m = len(color_names)
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
            _, div, time = runner(core_features, core_colors, kis, upper_gamma)
            print(f'{alg} finished in {time}! (diversity of {div})')

            results[alg].append((k_rounded, div, time))

    # TODO: double check how many of each color compared to K map
    # TODO: run another algorithm to compare diversity

    print('\n\nFINAL RESULTS:')

    for alg, resList in results.items():
        print(f'********{alg}********')
        for k, d, t in resList:
            print(f'{k}\t{d}\t{t}')

        print('\n\n', end='')