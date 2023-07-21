import itertools
import sys
import time
from typing import Any, Callable, List, Union

import networkx as nx
import numpy as np
from scipy.special import comb
import math
import utilsfdm
import random

ElemList = Union[List[utilsfdm.Elem], List[utilsfdm.ElemSparse]]

def FairGreedyFlowWrapped(features, colors, kis, epsilon, gammahigh, gammalow, normalize=False):
    '''
    A wrapper for FairFlow
    Adjust the problem instance for a different set of parameters
    '''
    '''
    A wrapper for FairFlow
    Adjust the problem instance for a different set of parameters
    '''

    c = len(kis)

    # Create a map of indices to colors
    color_number_map = list(kis.keys())

    # As done in the paper we will pre-process the data
    # by normalizing to have zero mean and unit standard deviation.
    features = np.array(features)
    features_normalized = features.copy()
    mean = np.mean(features_normalized, axis=0)
    std = np.std(features_normalized, axis=0)
    features_normalized = features_normalized - mean
    features_normalized = features_normalized/std
    features = features.tolist()
    features_normalized = features_normalized.tolist()

    elements = []
    elements_normalized = []
    for i in range(0, len(features_normalized)):
        elem_normalized = utilsfdm.Elem(i, color_number_map.index(colors[i]), features_normalized[i])
        elem = utilsfdm.Elem(i, color_number_map.index(colors[i]), features[i])
        elements.append(elem)
        elements_normalized.append(elem_normalized)

    # Adjust the constraints as a list
    kis_list = []
    for color in color_number_map:
        kis_list.append(kis[color])
    
    if normalize:
        return FairGreedyFlow(
                                X=elements_normalized, 
                                k=kis_list, 
                                m=c,
                                dist=utilsfdm.euclidean_dist,
                                eps=epsilon,
                                dmax=gammahigh,
                                dmin=gammalow,
                                metric_name='euclidean'
                            )
    else:
        return FairGreedyFlow(
                                X=elements, 
                                k=kis_list, 
                                m=c,
                                dist=utilsfdm.euclidean_dist,
                                eps=epsilon,
                                dmax=gammahigh,
                                dmin=gammalow,
                                metric_name='euclidean'
                            )

def FairGreedyFlow(X: ElemList, k: List[int], m: int, dist: Callable[[Any, Any], float], eps: float, dmax: float, dmin: float, metric_name) -> (List[int], float, float):
    t0 = time.perf_counter()
    sol, div_sol = None, 0.0
    sum_k = sum(k)
    # gammas to be searched
    list_of_gamma = [((1 + eps) ** i) * dmin for i in range(math.ceil(math.log(dmax / dmin, 1 + eps)) + 1)]
    lower, upper = 0, len(list_of_gamma) - 1
    while lower < upper - 1:
        mid = (lower + upper) // 2
        gamma = list_of_gamma[mid]
        d = gamma / (m + 1)
        # construct C
        R = X.copy()
        C = []
        while len(R) > 0 and len(C) <= sum_k * m:
            D, D_color = [], set()
            R_not_in_D_color = [x for x in R if x.color not in D_color]
            while R_not_in_D_color:
                if len(D) == 0:
                    p = random.choice(R_not_in_D_color)
                    D.append(p)
                    D_color = D_color.union({p.color})
                    R_not_in_D_color = [x for x in R if x.color not in D_color]
                else:
                    idx_lower_than_d = utilsfdm.get_id_lt_threshold(R_not_in_D_color, D, d, metric_name)
                    if idx_lower_than_d:
                        p_list = [x for x in R_not_in_D_color if x.idx in idx_lower_than_d]
                        p = random.choice(p_list)
                        D.append(p)
                        D_color = D_color.union({p.color})
                        R_not_in_D_color = [x for x in R if x.color not in D_color]
                    else:
                        break  # len(D) < m : dist(elem,D) >= d ,for all elem whose color is in {1,2,...,m} - D_color
            idx_lower_than_d = utilsfdm.get_id_lt_threshold(R, D, d, metric_name)
            R = [x for x in R if x.idx not in idx_lower_than_d]
            C.append(D)
            color_to_delete = []
            for i in range(m):
                D_i_count = 0
                for D in C:
                    for x in D:
                        if x.color == i:
                            D_i_count += 1
                            break
                if D_i_count >= sum_k:
                    color_to_delete.append(i)
            R = [x for x in R if x.color not in color_to_delete]
        # max ab-flow
        FlowG = nx.DiGraph()
        FlowG.add_node("a")
        FlowG.add_node("b")
        for i in range(m):
            FlowG.add_node("u" + str(i))
            FlowG.add_edge("a", "u" + str(i), capacity=k[i])
        for j in range(len(C)):
            FlowG.add_node("v" + str(j))
            FlowG.add_edge("v" + str(j), "b", capacity=1)
            for i in range(m):
                for x in C[j]:
                    if x.color == i:
                        FlowG.add_edge("u" + str(i), "v" + str(j), capacity=1)
                        break
        flow_size, flow_dict = nx.maximum_flow(FlowG, "a", "b")
        # print(flow_size, flow_dict)
        # next search or return result
        if flow_size < sum_k - 0.5:
            upper = mid
        else:
            lower = mid
            cur_sol = []
            for i in range(m):
                for j in range(len(C)):
                    node1 = "u" + str(i)
                    node2 = "v" + str(j)
                    if node1 in flow_dict.keys() and node2 in flow_dict[node1].keys() and flow_dict[node1][node2] > 0.5:
                        for x in C[j]:
                            if x.color == i:
                                cur_sol.append(x.idx)
                                break
            if len(cur_sol) != sum_k:
                print("There are some errors in flow_dict")
            else:
                cur_div = diversity(X, cur_sol, dist)
                if cur_div > div_sol:
                    sol = cur_sol
                    div_sol = cur_div
    t1 = time.perf_counter()
    return sol, div_sol, t1 - t0

def FairFlowWrapped(features, colors, kis, normalize=False):
    '''
    A wrapper for FairFlow
    Adjust the problem instance for a different set of parameters
    '''

    c = len(kis)

    # Create a map of indices to colors
    color_number_map = list(kis.keys())

    # As done in the paper we will pre-process the data
    # by normalizing to have zero mean and unit standard deviation.
    features = np.array(features)
    features_normalized = features.copy()
    mean = np.mean(features_normalized, axis=0)
    std = np.std(features_normalized, axis=0)
    features_normalized = features_normalized - mean
    features_normalized = features_normalized/std
    features = features.tolist()
    features_normalized = features_normalized.tolist()

    elements = []
    elements_normalized = []
    for i in range(0, len(features_normalized)):
        elem_normalized = utilsfdm.Elem(i, color_number_map.index(colors[i]), features_normalized[i])
        elem = utilsfdm.Elem(i, color_number_map.index(colors[i]), features[i])
        elements.append(elem)
        elements_normalized.append(elem_normalized)

    # Adjust the constraints as a list
    kis_list = []
    for color in color_number_map:
        kis_list.append(kis[color])
    
    if normalize:
        return FairFlow(X=elements_normalized, k=kis_list, m=c ,dist=utilsfdm.euclidean_dist)
    else:
        return FairFlow(X=elements, k=kis_list, m=c ,dist=utilsfdm.euclidean_dist)


def FairFlow(X: ElemList, m: int, k: List[int], dist: Callable[[Any, Any], float]) -> (List[int], float, float):
    t0 = time.perf_counter()
    sum_k = sum(k)
    S = []
    Div = []
    for c in range(m):
        Sc, divc = GMMC(X, color=c, k=sum_k, init=[], dist=dist)
        S.append(Sc)
        Div.append(divc)
    dist_matrix = np.empty([sum_k * m, sum_k * m])
    for c1 in range(m):
        for i1 in range(sum_k):
            for c2 in range(m):
                for i2 in range(sum_k):
                    dist_matrix[c1 * sum_k + i1][c2 * sum_k + i2] = dist(X[S[c1][i1]], X[S[c2][i2]])
    dist_array = np.sort(list(set(dist_matrix.flatten())))
    lower = 0
    upper = len(dist_array) - 1
    sol = None
    div_sol = 0.0
    while lower < upper - 1:
        mid = (lower + upper) // 2
        gamma = dist_array[mid]
        dist1 = m * gamma / (3 * m - 1)
        dist2 = gamma / (3 * m - 1)
        # print(mid, gamma, dist1, dist2)
        Z = []
        GZ = nx.Graph()
        for c in range(m):
            Zc = []
            for i in range(sum_k):
                if Div[c][i] >= dist1:
                    Zc.append(S[c][i])
                    GZ.add_node(S[c][i])
                else:
                    break
            Z.append(Zc)
        for c1 in range(m):
            for i1 in range(len(Z[c1])):
                for c2 in range(m):
                    for i2 in range(len(Z[c2])):
                        if c1 * sum_k + i1 != c2 * sum_k + i2 and dist_matrix[c1 * sum_k + i1][c2 * sum_k + i2] < dist2:
                            GZ.add_edge(Z[c1][i1], Z[c2][i2])
        C = []
        for cc in nx.connected_components(GZ):
            C.append(set(cc))
        FlowG = nx.DiGraph()
        FlowG.add_node("a")
        FlowG.add_node("b")
        for c in range(m):
            FlowG.add_node("u" + str(c))
            FlowG.add_edge("a", "u" + str(c), capacity=k[c])
        for j in range(len(C)):
            FlowG.add_node("v" + str(j))
            FlowG.add_edge("v" + str(j), "b", capacity=1)
            for c in range(m):
                for i in range(len(Z[c])):
                    if Z[c][i] in C[j]:
                        FlowG.add_edge("u" + str(c), "v" + str(j), capacity=1)
                        break
        flow_size, flow_dict = nx.maximum_flow(FlowG, "a", "b")
        # print(flow_size, flow_dict)
        if flow_size < sum_k - 0.5:
            upper = mid
        else:
            lower = mid
            cur_sol = []
            for c in range(m):
                for j in range(len(C)):
                    node1 = "u" + str(c)
                    node2 = "v" + str(j)
                    if node1 in flow_dict.keys() and node2 in flow_dict[node1].keys() and flow_dict[node1][node2] > 0.5:
                        for s_idx in Z[c]:
                            if s_idx in C[j]:
                                cur_sol.append(s_idx)
                                break
            if len(cur_sol) != sum_k:
                print("There are some errors in flow_dict")
            else:
                cur_div = diversity(X, cur_sol, dist)
                if cur_div > div_sol:
                    sol = cur_sol
                    div_sol = cur_div
    t1 = time.perf_counter()
    return sol, div_sol, t1 - t0

def diversity(X: ElemList, idxs: List[int], dist: Callable[[Any, Any], float]) -> float:
    '''
    Calculates the diversity
    '''
    div_val = sys.float_info.max
    for id1 in idxs:
        for id2 in idxs:
            if id1 != id2:
                div_val = min(div_val, dist(X[id1], X[id2]))
    return div_val

def GMMC(X: ElemList, color: int, k: int, init: List[int], dist: Callable[[Any, Any], float]) -> (List[int], List[float]):
    S = []
    div = []
    dist_array = np.full(len(X), sys.float_info.max)
    if len(init) == 0:
        first = -1
        for i in range(len(X)):
            if X[i].color == color:
                first = i
                break
        S.append(first)
        div.append(sys.float_info.max)
        for i in range(len(X)):
            if X[i].color == color:
                dist_array[i] = dist(X[first], X[i])
            else:
                dist_array[i] = 0.0
    else:
        for i in range(len(init)):
            S.append(init[i])
            div.append(sys.float_info.max)
        for i in range(len(X)):
            for j in S:
                if X[i].color == color:
                    dist_array[i] = min(dist_array[i], dist(X[i], X[j]))
                else:
                    dist_array[i] = 0.0

    while len(S) < k:
        max_idx = np.argmax(dist_array)
        max_dist = np.max(dist_array)
        S.append(max_idx)
        div.append(max_dist)
        for i in range(len(X)):
            if X[i].color == color:
                dist_array[i] = min(dist_array[i], dist(X[i], X[max_idx]))
    return S, div