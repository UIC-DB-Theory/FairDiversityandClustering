
import json
import utils
import typing
import typing as t
import sys
import numpy as np
import numpy.typing as npt
from collections import defaultdict
import matplotlib.pyplot as plt

import fdmalgs
import utils
import lpsolve
import multweights_nyoom as multweights
from coreset import Coreset_FMM

class Dataset:

    def __init__(self, datainfopath: str):
        """
        Constructs the dataset object that has the dataset loaded in it.
        :param datainfopath: path to the dataset files.
        """
        self.features = None
        self.colors = None

        
        # Read dataset info file
        with open(datainfopath) as f:
            datainfo = json.load(f)
        
        self.allFields = datainfo['allFields']
        self.colorFields = datainfo['colorFields']
        self.featureFields = datainfo['featureFields']
        self.dim = len(self.featureFields)
        self.datapath = datainfopath.split('/')[:-1].join('/') + '/' + datainfo['filename']

        # Read the dataset
        self.colors, self.features = self.read_CSV(self.datapath, self.allFields, self.colorFields, '_', self.featureFields)
        assert (len(self.colors) == len(self.features))
        
    
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

class Runner:
    def __init__(self, settingfilepath: str, dataset):
            """
            Compares algorithms
            :param datainfopath: path to the dataset files.
            """
            self.features = None
            self.colors = None

            
            # Read dataset info file
            with open(settingfilepath) as f:
                setting = json.load(f)

if __name__ == "__main__":

    d = Dataset("./datasets/ads/dataset.json")


