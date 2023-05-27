import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def make_image(name, width, height, data):

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
    
    img.save(name + '.png')

def line(l):
    
    tline = str(l[0])
    for i in range(1, len(l)):
     tline = tline + ", " + str(l[i])
    
    return tline + "\n"

def make_data(name, width, height):

    datafile = open(name + "_data.csv", "w")

    labels = ["color", "x", "y"]

    colors = ["r", "g", "b"]

    data = []

    for x in range (width):
        for y in range (height):
            color =random.choice(colors)
            data.append([color, x, y])
            datafile.write(line([color, x, y]))
    datafile.close()

    return data

def main():
    print('Generating image dataset')
    
    if len(sys.argv) != 4:
        print('Usage: python3 gen_pixel_dataset.py [<dataname>] [<width>] [<height>]')
    
    dataname = sys.argv[1]
    width = int(sys.argv[2])
    heigth = int(sys.argv[3])

    data = make_data(dataname, width, heigth)
    make_image(dataname, width, heigth, data)


if __name__ == "__main__":
    main()