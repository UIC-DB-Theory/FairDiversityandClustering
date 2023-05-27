import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Adding parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import utils

def line(l):
    
    tline = str(l[0])
    for i in range(1, len(l)):
     tline = tline + ", " + str(l[i])
    
    return tline + "\n"

def make_image(data, out):
    
    #print("Plotting: ", out)
    img = Image.new('RGB', (width, height), color='white')

    for p in data:
        color = p[0]
        x = p[1]
        y = p[2]
        pixel_color = (0,0,0)
        if color == "r":
            pixel_color = (255,0,0)
        elif color == "g":
            pixel_color = (0,255,0)
        elif color == "b":
            pixel_color = (0,0,255)
        #print(f'\tplotting: ({x}, {y}) with {color}')
        img.putpixel((int(x), int(y)), pixel_color)
    
    img.save(out)

width = 50
height = 50
f = open("pixels.csv", "w")

labels = ["color", "x", "y"]
#f.write(line(labels))

colors = ["r", "g", "b"]

data = []

for x in range (width):
    for y in range (height):
        color =random.choice(colors)
        data.append([color, x, y])
        f.write(line([color, x, y]))

f.close()


# variables for running LP bin-search
color_field = 'color'
feature_fields = ['x', 'y']
# feature_fields = {'age'}
kis = {"r": 10, "g": 10, "b": 10}
k = 30
# binary search params
epsilon = np.float64("0.001")

# other things for gurobi
method = 0  # model method of solving

# import data from file
allFields = [
    "color",
    "x",
    "y"
]
colors, features = utils.read_CSV("./pixels.csv", allFields, [color_field],'_',feature_fields)
assert (len(colors) == len(features))

# "normalize" features
# Should happen before coreset construction
# features = features / features.max(axis=0)


#print(f'Size of data = {len(features)}')
#print("Original\n", features)
#print("Original\n", colors)

import coreset as CORESET
#l = len(feature_fields) # doubling dimension = d 
coreset_constructor = CORESET.Coreset_FMM(features, colors, k, e_coreset, len(feature_fields))
features_c, colors_c = coreset_constructor.compute()

f = open(f'pixels_coreset_{eps}.csv', "w")
data_coreset = []
for i in range(0, len(features_c)):
    color = colors_c[i]
    x = int(features_c[i][0])
    y = int(features_c[i][1])
    data_coreset.append([color, x, y])
    f.write(line([color, x, y]))
make_image(data_coreset, f'coreset_{eps}.png')

# plot the images
make_image(data, "data.png")