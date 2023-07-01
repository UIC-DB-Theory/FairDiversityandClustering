import numpy as np
import KDTree2 as algo

def rand_round(min_dist, X, features, colors, kis, theory_accurate=True):
    """
    Randomly rounds a real solution to an integer one
    :param X: the variables of the program
    :param features: feature set
    :param colors: colors of features
    :param kis: color-count mapping
    :return: the indices of points in the integer solution
    """
    # we only want to pick from points we care about
    # (and we need to go backwards later)
    nonzero_indexes = np.nonzero(X != 0)[0] # always a tuple
    nonzeros = X[nonzero_indexes]

    # get a random permutation
    rands = np.random.random_sample(size=len(nonzeros))
    # of the original array!
    argsort = np.argsort(rands ** (1.0 / nonzeros))
    i_permutation = nonzero_indexes[argsort]

    if len(i_permutation) == 0:
        print('Empty result')
        exit(0)

    # declare variables
    # we always keep the first point
    # so we add it in first to ensure the tree is non-empty
    S = np.array([], dtype=int)
    # all our points to build a tree of points
    viewed_points = np.empty((1, features.shape[1]))
    # counts for colors we have found already
    b = {k: 0 for k in kis.keys()}

    for index in i_permutation:
        q = features[index]
        color = colors[index]

        # get distance to nearest point
        # (for now, we build the tree every time)
        # either we base this off of every point seen or every point included
        if theory_accurate:
            points = viewed_points
        else:
            points = features[S]
        dists, _ = algo.get_ind(algo.create(points), 1, q)
        dist = dists[0]

        viewed_points = np.append(viewed_points, [q], axis=0)

        if dist > min_dist and b[color] < kis[color]:
            b[color] += 1
            S = np.append(S, [index])
        else:
            # do nothing
            pass

    return S
