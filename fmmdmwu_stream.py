
import math

import numpy as np
import sys

from tqdm import trange

from datastructures.WeightedTree import WeightedTree
import datastructures.BallTree as BallTree
from algorithms.rounding import rand_round
import algorithms.coreset as CORESET
import algorithms.utils as algsU
import datasets.utils as datsU


from fmmdmwu_nyoom import epsilon_falloff as FMMDMWU

def fmmdmwu_stream(gen, features, colors, kis, gamma_upper, mwu_epsilon, falloff_epsilon, sample_percentage, return_unadjusted, percent_theoretical_limit=1.0):
    
    # Get values for m (number of colors) and k
    m = 0
    k = 0
    color_bins = {}
    for color in kis:
        m=m+1
        k = k + kis[color]
        # Initialize bins for each color
        color_bins[color] = []

    # Stream the data
    for feature, color in zip(features, colors):
        # TODO: Calculate the coreset for the streaming setting
        # Notes
        # Size of the coreset should be k*m, where m is the number of colors
        # For each color we run the clustering algorithm to get k points

        # Check if the bin has sufficient points
        if len(color_bins[color]) < k:
            # If not simply add the new point to the bin
            color_bins[color].append(feature)
        else:
            # TODO: Run k-center on set: color_bins[color] U {new point}
            pass
    
    # Merge the color bins to create the coreset
    core_features = []
    core_colors = []
    for color in color_bins:
        for feature in color_bins[color]:
            core_features.append(feature)
            core_colors.append(color)

    
    # Run MWU on the calculated coreset
    FMMDMWU(
        gen=gen,
        features = core_features, 
        colors = core_colors, 
        kis = kis,
        gamma_upper = gamma_upper,
        mwu_epsilon = mwu_epsilon,
        falloff_epsilon = falloff_epsilon,
        percent_theoretical_limit = percent_theoretical_limit,
        return_unadjusted = False
    )

    pass