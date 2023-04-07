from sklearn.neighbors import KDTree
import numpy as np
import numpy.typing as npt


def create(points):
    return KDTree(points)


def get_ind(structure, r: np.float64, point) -> npt.NDArray[int]:
    dim = point.shape[0]  # this is a tuple (reasons!)
    point_reshaped = np.reshape(point, (1, dim))

    return structure.query_radius(point_reshaped, r).flatten()[0]
