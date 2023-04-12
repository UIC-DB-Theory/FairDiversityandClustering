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

def make_coreset(colors: npt.NDArray[t.AnyStr], features: npt.NDArray[np.float64], kis, epsilon, error) -> (npt.NDArray[t.AnyStr], npt.NDArray[np.float64]):
    """
    returns the corest of the dataset; a subset that has similar properties for our purposes

    :param possible_colors: all possible color values in colors
    :param colors: the colors of the data
    :param features: corresponding features
    :param kis: the map of colors to required amounts
    :param epsilon: the division of space used in the algorithm
    :param error: the epsilon error for the coreset computation
    :return: a pair of new colors and features
    """
    import coreset_kcenter as kc

    out_colors = []
    out_features = []

    # our "K" value for number of clusters
    dim = features.shape[1]
    K = sum(kis.values()) / (error ** dim)

    # trying to hardcode for now since this was way too large
    K = 100

    # run k-center on all possible colors
    all_colors, inverse = np.unique(colors, return_inverse=True)
    for color in range(len(all_colors)):
        color_features = features[inverse == color]
        # TODO: just replicate a color repeatedly instead of doing this
        color_colors = colors[inverse == color]

        centerer = kc.Coreset_kCenter(color_features, K, error)
        indices = centerer.compute_kCenter_Coreset()
        out_colors.append(color_colors[indices]) #TODO: see above
        out_features.append(color_features[indices])

    out_colors = np.concatenate(out_colors)
    out_features = np.concatenate(out_features)

    return out_colors, out_features