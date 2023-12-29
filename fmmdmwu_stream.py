
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
    
    core_features = []
    core_colors = []

    # Stream the data
    for feature, color in zip(features, colors):
        # TODO: Calculate the coreset for the streaming setting
        pass
    
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