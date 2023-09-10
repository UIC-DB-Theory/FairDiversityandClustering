import json
import numpy as np
from csv import DictReader
import os


def read_dataset(datadir, feature_fields, color_fields, normalize = False):
    print(f'Reading dataset at: {datadir}')
    print(f'\tNormalize = {normalize}')
    metadatafilepath = ""
    for file in os.listdir(datadir):
        if file.endswith(".metadata"):
            metadatafilepath = os.path.join(datadir, file)

    with open(metadatafilepath, 'r') as f:
        metadata = json.load(f)
    datafilepath = os.path.join(datadir, metadata["filename"])
    fields = metadata["fields"]
    colors, features = read_CSV(datafilepath, fields, color_fields, "_", feature_fields)

    points_per_color = {}
    for color in colors:
        if color in points_per_color:
              points_per_color[color] += 1
        else:
             points_per_color[color] = 1

    if normalize:
        means = features.mean(axis=0)
        devs = features.std(axis=0)
        features = (features - means) / devs
    return {
         "features" : features,
         "colors" : colors,
         "points_per_color" : points_per_color
    }
    

def read_CSV(filename, field_names, color_fields, color_sep, feature_fields):
        """
        read_CSV.

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