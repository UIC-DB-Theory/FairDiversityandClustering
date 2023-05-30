import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gen_pixel_dataset import make_image

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import coreset_mod as CORESET
import utils

def make_image_data (features, colors):
    data = []
    for i in range(len(features)):
        row = [ colors[i], features[i][0], features[i][1]]
        data.append(row)
    return data

def main():
    print('Coreset Construction')
    
    if len(sys.argv) != 4:
        print('Usage: python3 vis.py [<datafile>] [<width>] [<height>]')
        exit(-1)

    datafile = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])

    allFields = [
        "color",
        "x",
        "y"
    ]

    # fields we care about for parsing
    color_fields = ['color']
    feature_fields = ['x', 'y']

    kis = {
        'r': 3,
        'g': 3,
        'b': 3,
    }
    k = sum(kis.values())

    colors, features = utils.read_CSV(datafile, allFields, color_fields, '_', feature_fields)

    coreset_size = 25

    print("Number of points (original): ", len(features))
    d = len(feature_fields)
    m = len(kis.keys())
    features, colors = CORESET.Coreset_FMM(features, colors, k, m, d, coreset_size).compute()
    print("Number of points (coreset): ", len(features))

    make_image('coreset', width, height, make_image_data(features, colors))


if __name__ == "__main__":
    main()