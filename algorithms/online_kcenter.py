
import typing as T
import numpy.typing as NPT
import numpy as np

from scipy.spatial.distance import pdist, cdist
from scipy.spatial import KDTree

from collections import defaultdict

from algorithms.utils import Stopwatch

class centerer:
    def __init__(self, size : int) -> None:
        self.centers = set() # k centers we've collected
                          # originally this is a dictionary to avoid duplicates
                          # then a numpy array later on
        self.prev_centers = None # the centers from the previous iteration
        self.k = size
        self.R = np.Inf # starts out as far as possible before initialization

        self.point_dim = None

        self.points_seen = 0
        self.total_insertion_time = 0
        self.last_finalize_time = 0

        self.in_startup = True

    def _calc_initial_R(self):
        """
        Computes the minimum nonzero distance between all points in the centers
        """
        assert(type(self.centers) is np.ndarray)
        dists = pdist(self.centers)
        self.R = np.min(dists)

    def _to_generator(self, points : NPT.NDArray):
        """
            returns a generator that iterates over the given points
            and increments the point counter
        """
        for point in points:
            point = np.reshape(point, (1, -1))
            yield point
            # we yielded a point so we've seen that point
            self.points_seen += 1


    def add(self, points : NPT.NDArray):
        # assumes that this is a 2d array of (n points, point dim)
        (_, dims) = points.shape
        if self.point_dim == None:
            self.point_dim = dims
        else:
            assert(self.point_dim == dims)

        # start timer
        timer = Stopwatch("Compute all points")

        # gets a generator to loop
        pgen = self._to_generator(points)

        # at this point we do the main algorithm
        try:
            # startup we find the first k unique points
            if self.in_startup:
                assert(type(self.centers) is set)
                # if we're yet to see the first k points, just add it
                while len(self.centers) < self.k:
                    next_point = next(pgen)
                    tuple_point = tuple(next_point[0])
                    # if we haven't seen it add it
                    if tuple_point not in self.centers:
                        self.centers.add(tuple_point)

                # we have k points, make the np array (I have no clue why this is a type error)
                points = [np.reshape(np.asarray(t), (1, -1)) for t in self.centers]
                self.centers = np.concatenate(points, axis=0)
                assert(type(self.centers) is np.ndarray)

                # Compute the first R radius and fall through to the rest
                self._calc_initial_R()
                self.in_startup = False

            # afterwards, we can do the normal algorithm
            assert(type(self.centers) is np.ndarray)

            while True:
                # pre-compute a tree of centers
                tree = KDTree(self.centers)
                # while |T| <= k
                while len(self.centers) <= self.k:
                    cur_p = next(pgen)
                    nearest_dist, _ = tree.query(cur_p)
                    if nearest_dist > 2 * self.R:
                        self.centers = np.append(self.centers, cur_p, axis=0)
                        # we added a point so we rebuild the tree
                        tree = KDTree(self.centers)

                # store this set of centers as the old centers before the next loop
                # should not need to be a copy as we refer to the old array
                self.prev_centers = self.centers

                # while there exists z in T such that p(z,T') > 2R
                new_centers = [np.reshape(self.centers[0], (1, -1))]
                candidates = self.centers[1:]
                for point in candidates:
                    point = np.reshape(point, (1, -1))
                    # get minimum distance to the new centers
                    min_dist = np.min(cdist(np.concatenate(new_centers, axis=0), point).ravel())
                    if min_dist > 2 * self.R:
                        new_centers.append(point)

                # remake the numpy array (it's faster to add to the list earlier)
                self.centers = np.concatenate(new_centers, axis=0)

                # update R
                self.R *= 2
        # we're out of points, so whatever we had is what we had
        except StopIteration:
            # add the time processing these points to the total time
            _, time = timer.stop()
            self.total_insertion_time += time

    def get_centers(self):
        """
            returns the currently computed centers, ensuring that
            we always return k points if we are able to.

            Otherwise, the centers may have fewer points in it,
            as we may be in the portion of the algorithm that is adding new points.
        """
        timer = Stopwatch("Finalize centers")

        if type(self.centers) is set:
            l = [np.reshape(np.asarray(t), (1, -1)) for t in self.centers]
            out = np.concatenate(l, axis=0)
            _, time = timer.stop()
            self.last_finalize_time = time
            return out
        
        assert(type(self.centers) is np.ndarray)

        # if we saw less than k points, return that many points
        if self.points_seen < self.k:
            out = self.centers[:self.points_seen]
            # sets the time (essentially 0 here)
            _, time = timer.stop()
            self.last_finalize_time = time
            return out
        
        # if we have no other points, just return
        if self.prev_centers is None:
            _, time = timer.stop()
            self.last_finalize_time = time
            return self.centers

        # if we have extra points, add the best ones (furthest distance) from all the points in the existing set

        # get the minimum distance to a point in S
        # cdist returns an array where element i j is distance:
        #   d(prev_centers[i], centers[j])
        # so we want to get the minimum along the second axis
        nearest_dists = np.min(cdist(self.prev_centers, self.centers), axis=1)
        
        # get the indices we need to take from prev_centers
        points_to_take = self.k - len(self.centers)

        # get the points_to_take furthest points
        # as k - points_to_take === k - (k - len(centers)) === len(centers)
        # so if we reverse the argpartition it'll give us the points_to_take furthest points
        indices = np.argpartition(nearest_dists, len(self.centers) - 1)[::-1]
        additionals = self.prev_centers[indices][:points_to_take]

        # adds those in
        output = np.concatenate([self.centers, additionals])

        """ old approach
        # if not, grab all the points in self.prev_centers not already in self.centers
        # for now, do this the slow way and convert back since these are small
        # TODO? faster?
        center_set = set(map(tuple, self.centers))
        addables = np.array([x for x in self.prev_centers if tuple(x) not in center_set])
        # adds points to the until we have K points
        output = np.concatenate([self.centers, addables])[:self.k]
        """

        # stops the time
        _, time = timer.stop()
        self.last_finalize_time = time
        return output

    def get_times(self):
        """
            returns a tuple of the average time spent per element processed and the last time taken to finalize the centers
        """
        return self.total_insertion_time / self.points_seen, self.last_finalize_time

class color_centerer:
    """
        centers points split up by color.
    """
    def __init__(self, size_per_color):
        self.size_per_color = size_per_color
        self.centerers = defaultdict(lambda: 
                            centerer(self.size_per_color))
    
    def add(self, features, colors):
        """
            adds a given array of features and colors to the centerer
        """
        # split features into unique colors
        all_colors, inverse = np.unique(colors, return_inverse=True)

        # add to the correct centerer
        for color_index in range(len(all_colors)):
            # The fetures for current color
            color_features = features[inverse == color_index]
            self.centerers[all_colors[color_index]].add(color_features)

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
        return np.concatenate(all_features), np.concatenate(all_colors)
    
    def get_times(self):
        """
            returns a tuple of the average time spent per element processed and the last time taken to finalize the centers
        """
        total_points = 0
        insertion_time = 0
        finalization_time = 0

        for color in self.centerers:
            # we can't use the helper method since we want to average across
            # ALL points, rather than across centerers
            total_points += self.centerers[color].points_seen
            insertion_time += self.centerers[color].total_insertion_time
            finalization_time += self.centerers[color].last_finalize_time

        return insertion_time / total_points, finalization_time


if __name__ == '__main__':

    insert_time = 0
    return_time = 0
    pointcnts = []

    c = color_centerer(25)

    num_points_per_round = 30000
    num_rounds = 10

    for r in range(num_rounds):
        print(f'round: {r}')
        points = np.random.rand(num_points_per_round, 2) * 10
        color_choices = np.array(['Red', 'Blue', 'Green'])
        colors = np.random.choice(color_choices, len(points))
        timer = Stopwatch('Centering')

        c.add(points, colors)
        timer.split('returning')
        features, colors = c.get_centers()
        print(colors)
        print(features.shape)
        [(_, insertion_times), (_, other_times)], _ = timer.stop()
        insert_time += insertion_times
        return_time += other_times
        pointcnts.append(len(features))

    print(f'Inserted {num_rounds} of {num_points_per_round} points each')
    print(f'average number of points after each round: {sum(pointcnts)/len(pointcnts)}')
    insert_t, last_center_t = c.get_times()
    print('times: ')
    print(f'\taverage point insertion: {insert_t}')
    print(f'\t      last finalization: {last_center_t}')
