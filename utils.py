import typing
import typing as t

import numpy as np
import numpy.typing as npt

import time


def read_CSV(filename: t.AnyStr, field_names: t.Sequence, color_fields: t.List[t.AnyStr], color_sep: t.AnyStr, feature_fields: t.Set[t.AnyStr]) -> (
        npt.NDArray[t.AnyStr], npt.NDArray[np.float64]):
    """read_CSV.

    Reads in a CSV datafile as a list of dictionaries.
    The datafile should have no header row

    Returns a tuple of the colors of elements and their features

    :param filename: the file to read in
    :type filename: t.AnyStr
    :param field_names: the headers of the CSV file, in order
    :type field_names: t.Sequence
    :param color_fields: the fields containing the object color; the end "color" will be a tuple of all the colors
    :type color_fields: t.Set[t.AnyStr]
    :param color_sep: Separator for joining together multiple color fields
    :type color_sep: t.AnyStr
    :param feature_fields: the fields which are numerical data values for the point
    :type feature_fields: t.Set[t.AnyStr]
    """
    from csv import DictReader

    # read csv as a dict with given keys
    reader = DictReader(open(filename), fieldnames=field_names)

    # return the requested features and colors
    colors = []
    features = []

    for row in reader:
        # this will become a row per color
        color_list = [row[color_field].strip() for color_field in color_fields]
        colors.append(color_sep.join(color_list))
        features.append([float(row[field]) for field in feature_fields])

    # we want these as np arrays
    colors = np.array(colors)
    features = np.array(features, dtype=np.float64)

    return colors, features

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
        """
        self.times.append(time.perf_counter())
        return zip(self.names, self._calc_deltas())