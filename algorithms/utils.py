import typing
import typing as t
import sys
import numpy as np
import numpy.typing as npt

import time

import math


class Stopwatch:
    def __init__(self, name: t.AnyStr):
        """
        Creates a new stopwatch that has started timing

        :param name: the name of the first split
        """
        self.names = [name]
        self.times = [time.perf_counter()]

    def split(self, name: t.AnyStr) -> ():
        """
        Stop the previous split and start another one with the given name
        :param name: the name of the new split
        :return: None
        """
        self.names.append(name)
        self.times.append(time.perf_counter())

    def get_splits(self) -> [t.AnyStr]:
        """
        Provides all existing splits, including the currently running split
        :return: a list of strings, one per split
        """
        return self.names

    def _calc_deltas(self) -> [float]:
        return [b - a for a, b in zip(self.times, self.times[1:])]

    def stop(self) -> [(t.AnyStr, float)]:
        """
        Stops the clock
        :return: a list of (split-name, delta-time) pairs for every segment created via "split" and creation
                 as well as the total time
        """
        self.times.append(time.perf_counter())
        return zip(self.names, self._calc_deltas()), self.times[-1] - self.times[0]

def compute_diversity(points: npt.NDArray[np.float64]) -> float:
    from scipy.spatial import distance
    div = sys.float_info.max
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            div = min(div, distance.euclidean(points[i], points[j]))
    return div

def compute_maxmin_diversity(points : npt.NDArray[np.float64]) -> float:
    """
    Computes maxmin diversity of a set of points (the minimum distance between any two points)
    under the Euclidean metric
    :param points: the set of points to compute diversity for
    :return: the minimum distance between points in points
    """
    from scipy.spatial import KDTree

    # get nearest point to each element
    tree = KDTree(points)
    distances, _ = tree.query(points, k=2)
    nonzero_distances = distances[:, 1]

    return np.min(nonzero_distances)


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

def check_returned_kis(colors, kis, S):
    """
    Computes the differences between each color value and chosen colors
    :param colors: the colors in the dataset
    :param kis: requested count per color
    :param S: computed solution (indicies into features and colors arrays)
    :return: a map of color to delta between kis and S
    """
    sol_colors = colors[S]

    color_vals, color_counts = np.unique(sol_colors, return_counts=True)

    # if we didn't return a color, our delta is the inverse of the requested count
    deltas = {c: -1 * count for c, count in kis.items()}

    # otherwise, we can compute the difference
    for color, computed in zip(color_vals, color_counts):
        actual = kis[color]
        delta = computed - actual
        deltas[color] = delta

    # we better have every color here
    assert (len(deltas.keys()) == len(kis.keys()))
    return deltas
