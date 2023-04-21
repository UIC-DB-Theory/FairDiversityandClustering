import random
import numpy as np
import utils
import lpsolve as alg
import matplotlib.pyplot as plt
from PIL import Image

def line(l):
    
    tline = str(l[0])
    for i in range(1, len(l)):
     tline = tline + ", " + str(l[i])
    
    return tline + "\n"

def make_image(data, out):
    
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
feature_fields = {'x', 'y'}
# feature_fields = {'age'}
kis = {"r": 10, "g": 10, "b" : 10}
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
colors, features = utils.read_CSV("./pixels.csv", allFields, color_field, feature_fields)
assert (len(colors) == len(features))

# "normalize" features
# Should happen before coreset construction
# features = features / features.max(axis=0)


print(f'Size of data = {len(features)}')
import coreset as CORESET
#l = len(feature_fields) # doubling dimension = d 
e_coreset = 20
coreset_constructor = CORESET.Coreset_FMM(features, colors, k, e_coreset, len(feature_fields))
features, colors = coreset_constructor.compute()
print(f'Coreset size = {len(features)}')



f = open("pixels_coreset.csv", "w")
data_coreset = []
for i in range(0, len(features)):
   color = colors[i]
   x = int(features[i][0])
   y = int(features[i][1])
   data_coreset.append([color, x, y])
   f.write(line([color, x, y]))

# plot the images
make_image(data, "data.png")
make_image(data_coreset, "coreset.png")