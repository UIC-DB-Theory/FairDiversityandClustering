import sys
import numpy as np
from scipy.spatial import distance

def coreset(features, colors, coreset_size):
    out_colors = []
    out_features = []
    
    all_colors, inverse = np.unique(colors, return_inverse=True)
    C = len(all_colors)

    # The Gonzalez's algorithm

    # Candidate set
    cand = set()

    # Distances of points 
    cand_dists = dict()

    cand_div = sys.float_info.max
    array_dists = [sys.float_info.max] * len(features)

    # Select first point as part of the candidate set
    cand.add(features[0])

    # Calculate the distances from the first point to all other points
    cand_dists[0] = dict()
    for i in range(len(features)):
        array_dists[i] = distance.euclidean(features[0], features[i])

    # Run till the size of the coreset is satisfied
    while len(cand) < coreset_size:
        max_idx = np.argmax(array_dists)
        max_dist = np.max(array_dists)
        cand.add(max_idx)
        cand_dists[max_idx] = dict()
        for idx in cand:
            if idx < max_idx:
                cand_dists[idx][max_idx] = dist(V[idx], V[max_idx])
            elif idx > max_idx:
                cand_dists[max_idx][idx] = dist(V[idx], V[max_idx])
        cand_div = min(cand_div, max_dist)
        for i in range(len(V)):
            array_dists[i] = min(array_dists[i], dist(V[i], V[max_idx]))

    # Divide candidates by colors
    cand_colors = list()
    for c in range(C):
        cand_colors.append(set())
    for idx in cand:
        c = V[idx].color
        cand_colors[c].add(idx)

    return out_features, out_colors