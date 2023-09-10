import numpy as np # linear algebra
import math
from scipy.spatial import distance

'''
Coreset for FMMD using Gonzalez Algorithm for k-Center Clustering

Example produced from:
    https://www.kaggle.com/code/barelydedicated/gonzalez-algorithm
'''

def gonzalez(data, num_centers):
    
    if(len(data) <= num_centers):
        print('\t\tlen(data) =', len(data), ", num_centers = ", num_centers)
        return data
    
    clusters = []
    clusters.append(data[0]) # let us assign the first cluster point to be first point of the data
    while len(clusters) is not num_centers:
        print(len(clusters))
        clusters.append(max_dist(data, clusters)) 
        # we add the furthest point from ALL current clusters
    return (clusters)

def max_dist(data, clusters):
    distances = np.zeros(len(data)) # we will keep a cumulative distance measure for all points
    for cluster_id, cluster in enumerate(clusters):
        for point_id, point in enumerate(data):
            if distance.euclidean(point,cluster) == 0.0:
                distances[point_id] = -math.inf # this point is already a cluster (obselete)
            if not math.isinf(distances[point_id]):
                # if a point is not obselete, then we add the distance to its specific bin
                distances[point_id] = distances[point_id] + distance.euclidean(point,cluster) 
                # return the point which is furthest away from all the other clusters
    return data[np.argmax(distances)]



def coreset(features, colors, coreset_size):
    out_colors = []
    out_features = []

    # run k-center on all possible colors
    all_colors, inverse = np.unique(colors, return_inverse=True)
    coreset_per_color = coreset_size/len(all_colors)
    for color in range(len(all_colors)):
        print('\t color: ', color)

        # The fetures for current color
        color_features = features[inverse == color]
        print('\t num features in color: ', len(color_features))

        # Calculate the GMM for colored set
        color_coreset = gonzalez(color_features, coreset_per_color)

        # The coressponding color list
        color_colors = np.array([all_colors[color]]*len(color_coreset))

        out_colors.append(color_colors)
        out_features.append(color_coreset)

    out_colors = np.concatenate(out_colors)
    out_features = np.concatenate(out_features)

    return out_features, out_colors




