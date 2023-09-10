import numpy as np

def buildKisMap(colors, k, a):
    """
    builds a map to select at least (1-a) * k total points, spaced out in all colors
    The amount per color is calculated via the formula:
        max(1, (1-a) * k n_c / n)

    :param colors: an array of every color of the points in the dataset
    :param k: total # to select
    :param a: minimum percentage of k to allow
    :return: a map of unique colors to counts
    """
    N = len(colors)

    color_names, color_counts = np.unique(colors, return_counts=True)

    # we build up a dictionary of kis based on the names and counts
    # groups are built by the formula  max(1, (1-a) * k n_c / n)
    def calcKi(a, k, c):
        raw = max(1, ((1.0 - a) * k * c) / N)
        return math.ceil(raw)

    kis = {n: calcKi(a, k, c) for n, c in zip(color_names, color_counts)}

    return kis