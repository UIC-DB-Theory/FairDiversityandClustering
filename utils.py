import typing as t

import numpy as np
import numpy.typing as npt


def read_CSV(filename: t.AnyStr, field_names: t.Sequence, color_field: t.AnyStr, feature_fields: t.Set[t.AnyStr]) -> (
        npt.NDArray[t.AnyStr], npt.NDArray[np.float64]):
    """read_CSV.

    Reads in a CSV datafile as a list of dictionaries.
    The datafile should have no header row

    Returns a tuple of the colors of elements and their features

    :param filename: the file to read in
    :type filename: t.AnyStr
    :param field_names: the headers of the CSV file, in order
    :type field_names: t.Sequence
    :param color_field: the field containing the object color
    :type color_field: t.AnyStr
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
        colors.append(row[color_field].strip())
        features.append([float(row[field]) for field in feature_fields])

    # we want these as np arrays
    colors = np.array(colors)
    features = np.array(features, dtype=np.float64)

    return colors, features
