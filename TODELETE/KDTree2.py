from scipy.spatial import KDTree
import numpy as np
import numpy.typing as npt


def create(points):
    """
    Creates a KDTree data structure over the "points"
    :param points: NP Array of features
    :return: KDTree of the features
    """
    return KDTree(points)


def get_ind_range(structure, r: np.float64, point) -> npt.NDArray[int]:
    """
    Queries the KDTree for neighbors within a radius r of point
    :param structure: KDTree to query over
    :param r: radius
    :param point: a point to query
    :return: a NPArray of indices
    """
    dim = point.shape[0]  # this is a tuple (reasons!)
    point_reshaped = np.reshape(point, (1, dim))

    return np.array(structure.query_ball_point(point_reshaped, r).flatten()[0])

def get_self_range(structure, r: np.float64) -> npt.NDArray[int]:
    """
    Queries the KDTree for all neighbors of itself within radius gamma
    :param structure: the tree to search
    :param r: radius for points to be close
    :return: list of lists aligning points in structure to their neighbors
    """
    return structure.query_ball_tree(structure, r)

def get_ind(structure, k : int, point) -> npt.NDArray[int]:
    dim = point.shape[0]  # this is a tuple (reasons!)
    point_reshaped = np.reshape(point, (1, dim))
    return structure.query(point_reshaped, k)

def get_weight_ranges(structure, weights, gamma):
    ind = structure.query_ball_tree(structure, gamma)
    weights = [np.sum(weights[i]) for i in ind]
    return np.array(weights)

def get_count_in_range(structure : KDTree, point : npt.NDArray, r : float):
    ind = get_ind_range(structure, r, point)
    return ind.size