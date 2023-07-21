import sys
from typing import Any, Callable, List

import numpy as np
import scipy.sparse as sp
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_chunked


class Elem:
    def __init__(self, idx: int, color: int, features: List[float]):
        self.idx = idx
        self.color = color
        self.features = features


class ElemSparse:
    def __init__(self, idx: int, color: int, features: sp.csr_matrix):
        self.idx = idx
        self.color = color
        self.features = features


def euclidean_dist(elem1: Elem, elem2: Elem) -> float:
    if len(elem1.features) != len(elem2.features):
        raise AssertionError("dimension not match")
    return distance.euclidean(elem1.features, elem2.features)


def euclidean_dist_sparse(elem1: ElemSparse, elem2: ElemSparse) -> float:
    return pairwise_distances(elem1.features, elem2.features, 'euclidean')[0, 0]


def manhattan_dist(elem1: Elem, elem2: Elem) -> float:
    if len(elem1.features) != len(elem2.features):
        raise AssertionError("dimension not match")
    return distance.cityblock(elem1.features, elem2.features)


def manhattan_dist_sparse(elem1: ElemSparse, elem2: ElemSparse) -> float:
    return pairwise_distances(elem1.features, elem2.features, 'cityblock')[0, 0]


def cosine_dist(elem1: Elem, elem2: Elem) -> float:
    if len(elem1.features) != len(elem2.features):
        raise AssertionError("dimension not match")
    return np.arccos(1.0 - distance.cosine(elem1.features, elem2.features))


def cosine_dist_sparse(elem1: ElemSparse, elem2: ElemSparse) -> float:
    distance = pairwise_distances(elem1.features, elem2.features, 'cosine')[0, 0]
    return np.arccos(1.0 - distance)


def diversity(elements: List[Elem], dist: Callable[[Any, Any], float]) -> float:
    div = sys.float_info.max
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            div = min(div, dist(elements[i], elements[j]))
    return div


def div_subset(elements: List[Elem], indices: List[int], dist: Callable[[Any, Any], float]) -> float:
    div = sys.float_info.max
    for i in indices:
        for j in indices:
            if i < j:
                div = min(div, dist(elements[i], elements[j]))
    return div


def get_id_lt_threshold(R: List[Elem], D: List[Elem], threshold, metric_name) -> list:
    if not R:
        return []
    else:
        R_features = [x.features for x in R]
        D_features = [x.features for x in D]
        if metric_name.endswith('_sparse'):
            R_features = sp.vstack(R_features)
            D_features = sp.vstack(D_features)
            metric_name = metric_name[:-7]
        # d_RD = distance.cdist(R_features, D_features, metric_name)
        bool_list = []
        gen = pairwise_distances_chunked(R_features, D_features, metric=metric_name)
        for chunk in gen:
            bool_chunk = list([np.any(chunk < threshold, axis=1)][0])
            bool_list += bool_chunk
        return [x.idx for x in np.array(R)[np.array(bool_list)]]
