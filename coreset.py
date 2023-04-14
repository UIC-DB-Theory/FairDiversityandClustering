import numpy as np
import numpy.typing as npt
import helper_functions as hf
import time, random
from tqdm import trange, tqdm

# TODO: add types later :)
class Coreset_FMM:
    """
    Class to compute Fair Max Min Coreset.
    
    It computes the (1+e)-coreset for FAIR-MAX-MIN in metric spaces of low doubling dimension (l).
    
    The paper [2] cited below guarantees that in the resulting coreset there would be enough points
    from each color group to satisfy fairness while these points are atleast l*/(1+e) apart.
    TODO: (l* is the optimal diversity score? -- double check this)

    Parameters
    ----------
        features : ndarry of points.
        colors : list of colors corresponding to the points.
        k : cardinality of the result set for FMMD.
        e : epsilon value for coreset.
        l: the doubling dimension.
        
    ----------
    References
    [1] Agarwal, Pankaj K., Sariel Har-Peled, and Kasturi R. Varadarajan. "Geometric approximation via coresets."
    Combinatorial and computational geometry 52.1-30 (2005): 3.

    [2] Addanki, R., McGregor, A., Meliou, A., & Moumoulidou, Z. (2022). Improved approximation and scalability 
    for fair max-min diversification. arXiv preprint arXiv:2201.06678.

    [3] https://github.com/kvombatkere/CoreSets-Algorithms
    """

    # Initialize with parameters
    def __init__(self, features, colors, k , e, l):

        if isinstance(features, np.ndarray):
            self.features = features
        else:
            self.features = np.array(features)

        self.colors = colors
        self.k = k
        self.e = e

        # TODO: double check this
        (_, c) = features.shape
        self.d = c
        
        # The result set size while running gmm for each color
        self.gmm_result_size = np.ceil(pow(((4*(e+1))/(e)), l) * k).astype(int)


    # Compute Greedy k-center/GMM with polynomial 2-approximation
    def GMM(self, input_set):
        from scipy.spatial.distance import cdist

        if len(input_set) < self.gmm_result_size:
            return np.array(input_set)

        # Randomly select a point.
        randomPointIndex = np.random.randint(0, len(input_set) + 1)
        s_1 = input_set[randomPointIndex]

        # Initialize all distances initially to s_1.
        # we need to reshape the point vector into a row vector
        dim = s_1.shape[0]  # this is a tuple (reasons!)
        point = np.reshape(s_1, (1, dim)) # point is the current "max_dist" point we're working with

        # comes out as a Nx1 matrix
        point_distances = cdist(input_set, point)

        # Result set for GMM
        result = np.zeros((self.gmm_result_size, self.d), np.float64)
        result[0] = point

        for i in trange(1, self.gmm_result_size):

            # Get the farthest point from current point
            # max_point_index = point_distances.index(max(point_distances))
            # maximum_dist_point = input_set[max_point_index]
            maximum_dist_point = input_set[point_distances.argmax()]

            result[i] = maximum_dist_point

            # Update point distances with respect to the maximum_dis_point.
            # we keep the minimum distance any point is to our selected point
            point = np.reshape(maximum_dist_point, (1, dim))
            new_point_distances = cdist(input_set, point)
            point_distances = np.minimum(point_distances, new_point_distances)

        # TODO: What is this and is this relevant to us?
        # Get the cost, R
        # self.R_val = max(point_distances)
        # print("Cost (max) of k-center clustering={:.3f}".format(self.R_val))

        return result

        


    # Compute the coreset for FMM
    def compute(self):
        
        print("No. of points selected by GMM per color:", self.gmm_result_size)

        # TODO: Not sure if we can do this using numpy
        # First we segregate the feautures by their colors
        features_per_color = {}
        m = 0
        for i in range(0, len(self.features)):
            if self.colors[i] not in features_per_color:
                features_per_color[self.colors[i]] = [self.features[i]]
                m+=1
            else:
                features_per_color[self.colors[i]].append(self.features[i])
        
        print("No. of points per color:")
        for color in features_per_color:
            print(f'\t{color}: {len(features_per_color[color])}')

        # Calcualte the coreset
        coreset = np.empty((0, self.d), np.float64)
        colors = []
        for color in features_per_color:

            print("Calculating coreset for color:", color)

            # Concatenate the result sets of GMM run on each colored set.
            # Had to change this to account for failing color lengths
            gmm_res = self.GMM(features_per_color[color])
            coreset = np.append(coreset, gmm_res, axis=0)
            colors = colors + ([color]*len(gmm_res))

        colors = np.array(colors)

        return coreset, colors