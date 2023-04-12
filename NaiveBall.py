import numpy as np
import numpy.typing as npt


def create(points):
    return points


def get_ind(points: npt.NDArray[np.float64], r: np.float64, point) -> npt.NDArray[int]:
    """naive_ball.

    Returns the indices of points inside the ball

    :param dF: the distance function point -> point -> num
    :param points: a list of all points in the dataset
    :type points: t.List[t.Any]
    :param r: the radius of the sphere to bound
    :param point: the circle's center
    :rtype: t.List[t.Any]
    """
    from scipy.spatial.distance import cdist

    # we need to reshape the point vector into a row vector
    dim = point.shape[0]  # this is a tuple (reasons!)
    point = np.reshape(point, (1, dim))

    # comes out as a Nx1 matrix
    dists = cdist(points, point)

    return (dists.flatten() <= r).nonzero()[0]  # Also a tuple!
