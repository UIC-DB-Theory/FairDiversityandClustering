
import typing as T
import numpy.typing as NPT
import numpy as np

from scipy.spatial.distance import pdist
from scipy.spatial import KDTree

from collections import defaultdict



class centerer:
    def __init__(self, size : int, point_dim : int) -> None:
        self.centers = np.empty((size, point_dim)) # up to k center points we've selected
        self.prev_centers = None # the centers from the previous iteration
        self.k = size
        self.point_dim = point_dim
        self.points_seen = 0
        self.R = np.Inf # starts out as far as possible before initialization

    def _calc_initial_R(self):
        """
        Computes the minium distance between all points in the centers

        """
        dists = pdist(self.centers)
        self.R = min(dists)

    def _to_generator(self, points : NPT.NDArray):
        """
            returns a generator that iterates over the given points
            and increments the point counter
        """
        for point in points:
            self.points_seen += 1
            point = np.reshape(point, (1, -1))
            yield point


    def add(self, points : NPT.NDArray):
        # assumes that this is a 2d array of (n points, point dim)
        (_, dims) = points.shape
        assert(self.point_dim == dims)

        # gets a generator to loop
        pgen = self._to_generator(points)

        # if we're yet to see the first k points, just add it
        while self.points_seen < self.k and pgen:
            cur_index = self.points_seen
            self.centers[cur_index] = next(pgen)

        # if we have seen k points, compute the first R radius
        # and fall through to the rest
        if self.points_seen == self.k:
            self._calc_initial_R()

        # at this point we do the main algorithm
        try:
            while True:
                # while |T| <= k
                while len(self.centers) <= self.k:
                    cur_p = next(pgen)
                    tree = KDTree(self.centers)
                    nearest_dist, _ = tree.query(cur_p)
                    if nearest_dist > 2 * self.R:
                        self.centers = np.append(self.centers, cur_p, axis=0)

                # store this set of centers as the old centers before the next loop
                self.prev_centers = np.copy(self.centers)

                # while there exists z in T such that p(z,T') > 2R
                # compute the tree and update all at once
                tree = KDTree(self.centers)
                too_close_pairs = tree.query_pairs(2.0 * self.R)
                # remove any point in centers (T) from centers (T)
                indecies_to_delete = [i for (i, _) in too_close_pairs]
                self.centers = np.delete(self.centers, indecies_to_delete, axis=0)
                # update R
                self.R *= 2
        # we're out of points, so whatever we had is what we had
        except StopIteration:
            return self.get_centers()

    def get_centers(self):
        """
            returns the currently computed centers, ensuring that
            we always return k points if we are able to.

            Otherwise, the centers may have fewer points in it,
            as we may be in the portion of the algorithm that is adding new points.
        """

        # if we have no other points, just return
        if self.prev_centers is None:
            return self.centers

        # if not, grab all the points in self.prev_centers not already in self.centers
        # for now, do this the slow way and convert back since these are small
        # TODO? faster?
        addables = np.array(set(map(tuple, self.prev_centers)).difference(map(tuple, self.centers)))
        # adds points to the until we have K points
        return np.append(self.centers, addables)[:self.k]

class color_centerer:
    """
        centers points split up by color.
    """
    def __init__(self, size_per_color, point_dim):
        self.size_per_color = size_per_color
        self.point_dim = point_dim
        self.centerers = defaultdict(lambda: 
                            centerer(self.size_per_color, self.point_dim))
    
    def add(self, features, colors):
        """
            adds a given array of features and colors to the centerer
        """
        # split features into unique colors
        all_colors, inverse = np.unique(colors, return_inverse=True)

        # add to the correct centerer
        for color in range(len(all_colors)):
            # The fetures for current color
            color_features = features[inverse == color]
            self.centerers[color].add(color_features)

        # TODO: this is technically a waste since the previous add call
        # already computes this for each individually, but this is nicer for now
        return self.get_centers()

    def get_centers(self):
        """
            returns the k-center solution as of current
        """
        # grab the solution for every color we've seen
        all_features = []
        all_colors = []
        for color in self.centerers:
            features = self.centerers[color].get_centers()
            colors = np.repeat(color, len(features))
            assert(len(features) == len(colors))
            all_features.append(features)
            all_colors.append(colors)

        # return the solutions all combined
        return np.concatenate(all_features), np.concatenate((all_colors))


if __name__ == '__main__':
    from utils import Stopwatch

    insert_time = 0
    return_time = 0
    pointcnts = []

    c = color_centerer(50, 2)

    for _ in range(10):
        points = np.random.rand(10000, 2)
        colors = np.random.choice(['Red', 'Blue', 'Green', 'fat', 'skinny', 'twitter user'], len(points))
        timer = Stopwatch('Centering')

        c.add(points, colors)
        timer.split('returning')
        features, colors = c.get_centers()
        print(len(features))
        [(_, insertion_times), (_, other_times)], _ = timer.stop()
        insert_time += insertion_times / 1000
        return_time += other_times
        pointcnts.append(len(features))

    print(f'average: {sum(pointcnts)/len(pointcnts)}')
    print(f'insertion time: {insert_time / 100}')
    print(f'finalize time: {return_time / 100}')
