from sklearn.neighbors import BallTree
import numpy as np
import numpy.typing as npt

def create(points):
    """
    Creates a BallTree data structure over the "points"
    :param points: NP Array of features
    :return: BallTree of the features
    """
    return BallTree(points)


def get_ind(structure, r: np.float64, point) -> npt.NDArray[int]:
    """
    Queries the BallTree for neighbors within a radius r of point
    :param structure: BallTree to query over
    :param r: radius
    :param point: a point to query
    :return: a BallTree of indices
    """
    dim = point.shape[0]  # this is a tuple (reasons!)
    point_reshaped = np.reshape(point, (1, dim))

    return structure.query_radius(point_reshaped, r).flatten()[0]


def get_counts_in_range(structure : BallTree, points : npt.NDArray, r : float):
    return structure.query_radius(points, r, count_only=True)