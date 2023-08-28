import numpy as np
import numpy.typing as npt
import typing as t
from tqdm import trange
import time
class Coreset_FMM:
    """
    Class to compute Fair Max Min Coreset.

    It computes the (1+e)-coreset for FAIR-MAX-MIN in metric spaces of low doubling dimension (l).

    Parameters
    ----------
        features : ndarry of points.
        colors : list of colors corresponding to the points.
        k : cardinality of the result set for FMMD.
        m : number of unique colors
        d : data of the dimension
        coreset_size : required size of the coreset
        
    ----------
    References
    [1] Agarwal, Pankaj K., Sariel Har-Peled, and Kasturi R. Varadarajan. "Geometric approximation via coresets."
    Combinatorial and computational geometry 52.1-30 (2005): 3.

    [2] Addanki, R., McGregor, A., Meliou, A., & Moumoulidou, Z. (2022). Improved approximation and scalability 
    for fair max-min diversification. arXiv preprint arXiv:2201.06678.

    [3] https://github.com/kvombatkere/CoreSets-Algorithms
    """

    # Initialize with parameters
    def __init__(self, features: npt.NDArray[np.float64], colors: npt.NDArray[t.AnyStr], k, m, d, coreset_size):

        # Convert features and colors to numpy arrays
        if isinstance(features, np.ndarray):
            self.features = features
        else:
            self.features = np.array(features)
        
        if isinstance(colors, np.ndarray):
            self.colors = colors
        else:
            self.colors = np.array(colors)

        # Minimum size of the result set (FMMD)
        self.k = k

        # Number of unique colors
        self.m  = m

        # Required coreset size
        self.coreset_size = coreset_size

        # Dimensions - number of features per point
        self.d = d
       
        # GMM selects equal number of points per color
        self.gmm_result_size = int(coreset_size/m)

        # error value for calculated coreset
        self.e = pow(((k*m)/coreset_size),(1/d)) * 8

        self.coreset_compute_time = 0
        self.gamma_upper_bound_compute_time = 0
    
    def update_coreset_size(self, coreset_size):
        self.coreset_size = coreset_size
        self.e = pow(((self.k*self.m)/coreset_size),(1/self.d)) * 8
        self.gmm_result_size = int(coreset_size/self.m)


    # Compute Greedy k-center/GMM with polynomial 2-approximation
    def GMM(self, input_set: npt.NDArray[np.float64]):
        from scipy.spatial.distance import cdist

        if len(input_set) < self.gmm_result_size:
            return np.array(input_set)

        # Randomly select a point.
        s_1 = np.random.default_rng().choice(input_set)

        # Initialize all distances initially to s_1.
        # we need to reshape the point vector into a row vector
        dim = s_1.shape[0]  # this is a tuple (reasons!)
        point = np.reshape(s_1, (1, dim)) # point is the current "max_dist" point we're working with

        # comes out as a Nx1 matrix
        point_distances = cdist(input_set, point)

        # Result set for GMM
        result = np.zeros((self.gmm_result_size, self.d), np.float64)
        result[0] = point

        for i in range(1, self.gmm_result_size):

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

        return result
    
    # returns the upper bound for gamma as 2*div(U)
    def compute_gamma_upper_bound(self):

        gamma_high = float('inf')

        from scipy.spatial.distance import cdist

        # Randomly select a point.
        s_1 = np.random.default_rng().choice(self.features)

        # Initialize all distances initially to s_1.
        # we need to reshape the point vector into a row vector
        dim = s_1.shape[0]  # this is a tuple (reasons!)
        point = np.reshape(s_1, (1, dim)) # point is the current "max_dist" point we're working with

        # comes out as a Nx1 matrix
        point_distances = cdist(self.features, point)

        # Result set for GMM
        result = np.zeros((self.k, self.d), np.float64)
        result[0] = point

        for i in range(1, self.k):

            # Get the farthest point from current point
            # max_point_index = point_distances.index(max(point_distances))
            # maximum_dist_point = input_set[max_point_index]
            maximum_dist_point = self.features[point_distances.argmax()]
            
            distance = point_distances[point_distances.argmax()][0]

            if distance < gamma_high:
                gamma_high = distance

            result[i] = maximum_dist_point

            # Update point distances with respect to the maximum_dis_point.
            # we keep the minimum distance any point is to our selected point
            point = np.reshape(maximum_dist_point, (1, dim))
            new_point_distances = cdist(self.features, point)
            point_distances = np.minimum(point_distances, new_point_distances)
        
        return 2*gamma_high
    
    # Compute the coreset for FMM
    def compute(self):
        t0 = time.perf_counter()
        out_colors = []
        out_features = []

        # run k-center on all possible colors
        all_colors, inverse = np.unique(self.colors, return_inverse=True)
        for color in range(len(all_colors)):

            # The fetures for current color
            color_features = self.features[inverse == color]

            # Calculate the GMM for colored set
            color_coreset = self.GMM(color_features)

            # The coressponding color list
            color_colors = np.array([all_colors[color]]*len(color_coreset))

            out_colors.append(color_colors)
            out_features.append(color_coreset)

        out_colors = np.concatenate(out_colors)
        out_features = np.concatenate(out_features)
        t1 = time.perf_counter()
        self.coreset_compute_time = t1- t0
        return out_features, out_colors