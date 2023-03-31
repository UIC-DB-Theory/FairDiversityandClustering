import sys
import time
from typing import Any, Callable, Iterable, List, Set, Union

import networkx as nx
import numpy as np

import utils

ElemList = Union[List[utils.Elem], List[utils.ElemSparse]]


class Instance:
    def __init__(self, k: int, mu: float, m: int):
        self.k = k
        self.mu = mu
        self.div = sys.float_info.max
        self.idxs = set()
        if m > 1:
            self.group_idxs = []
            for c in range(m):
                self.group_idxs.append(set())


def StreamDivMax(X: ElemList, k: int, dist: Callable[[Any, Any], float], eps: float, dmax: float, dmin: float) -> (List[int], float):
    Ins = []
    cur_d = dmax - 0.01
    while cur_d > dmin:
        ins = Instance(k, mu=cur_d, m=1)
        Ins.append(ins)
        cur_d *= (1.0 - eps)
    for x in X:
        for ins in Ins:
            if len(ins.idxs) == 0:
                ins.idxs.add(x.idx)
            elif len(ins.idxs) < ins.k:
                div_x = sys.float_info.max
                flag_x = True
                for y_idx in ins.idxs:
                    div_x = min(div_x, dist(x, X[y_idx]))
                    if div_x < ins.mu:
                        flag_x = False
                        break
                if flag_x:
                    ins.idxs.add(x.idx)
                    ins.div = min(ins.div, div_x)
    max_inst = None
    max_div = 0
    for ins in Ins:
        if len(ins.idxs) == k and ins.div > max_div:
            max_inst = ins
            max_div = ins.div
    return max_inst.idxs, max_inst.div


def StreamFairDivMax1(X: ElemList, k: List[int], dist: Callable[[Any, Any], float], eps: float, dmax: float, dmin: float) -> (set, float, int, float, float):
    if len(k) != 2:
        print("The length of k must be 2")
        return set(), 0.0, 0, 0.0, 0.0
    t0 = time.perf_counter()
    # Initialization
    all_ins = []
    group_ins = [list(), list()]
    cur_d = dmax - 0.01
    while cur_d > dmin:
        ins = Instance(k=k[0] + k[1], mu=cur_d, m=2)
        all_ins.append(ins)
        for c in range(2):
            gins = Instance(k=k[c], mu=cur_d, m=1)
            group_ins[c].append(gins)
        cur_d *= (1.0 - eps)
    # Stream processing
    for x in X:
        for ins in all_ins:
            if len(ins.idxs) == 0:
                ins.idxs.add(x.idx)
                ins.group_idxs[x.color].add(x.idx)
            elif len(ins.idxs) < ins.k:
                div_x = sys.float_info.max
                flag_x = True
                for idx_y in ins.idxs:
                    div_x = min(div_x, dist(x, X[idx_y]))
                    if div_x < ins.mu:
                        flag_x = False
                        break
                if flag_x:
                    ins.idxs.add(x.idx)
                    ins.group_idxs[x.color].add(x.idx)
                    ins.div = min(ins.div, div_x)
        for gins in group_ins[x.color]:
            if len(gins.idxs) == 0:
                gins.idxs.add(x.idx)
            elif len(gins.idxs) < gins.k:
                div_x = sys.float_info.max
                flag_x = True
                for idx_y in gins.idxs:
                    div_x = min(div_x, dist(x, X[idx_y]))
                    if div_x < gins.mu:
                        flag_x = False
                        break
                if flag_x:
                    gins.idxs.add(x.idx)
                    gins.div = min(gins.div, div_x)
    # t1 = time.perf_counter()
    # stream_time_per_elem = (t1 - t0) / len(X)
    stored_elements = set()
    for ins_id in range(len(all_ins)):
        stored_elements.update(all_ins[ins_id].idxs)
        for c in range(2):
            stored_elements.update(group_ins[c][ins_id].idxs)
    num_elements = len(stored_elements)
    # Post-processing (1): Find the lower index
    # t0 = time.perf_counter()
    lower = 0
    upper = len(all_ins) - 1
    while lower < upper - 1:
        mid = (lower + upper) // 2
        if len(all_ins[mid].idxs) == k[0] + k[1] and len(group_ins[0][mid].idxs) == k[0] and len(group_ins[1][mid].idxs) == k[1]:
            upper = mid
        else:
            lower = mid
    # Post-processing (2): Balance each instance so that it is a fair solution.
    sol_div = 0.0
    sol_idx = -1
    for ins_id in range(upper + 1):
        if len(all_ins[ins_id].group_idxs[0].union(group_ins[0][ins_id].idxs)) < k[0] or len(all_ins[ins_id].group_idxs[1].union(group_ins[1][ins_id].idxs)) < k[1]:
            continue
        elif len(all_ins[ins_id].idxs) < k[0] + k[1]:
            while len(all_ins[ins_id].group_idxs[0]) < k[0]:
                max_div = 0.0
                max_idx = -1
                for idx1 in group_ins[0][ins_id].idxs:
                    if idx1 not in all_ins[ins_id].group_idxs[0]:
                        div1 = sys.float_info.max
                        for idx2 in all_ins[ins_id].idxs:
                            div1 = min(div1, dist(X[idx1], X[idx2]))
                        if div1 > max_div:
                            max_div = div1
                            max_idx = idx1
                all_ins[ins_id].idxs.add(max_idx)
                all_ins[ins_id].group_idxs[0].add(max_idx)
            while len(all_ins[ins_id].group_idxs[1]) < k[1]:
                max_div = 0.0
                max_idx = -1
                for idx1 in group_ins[1][ins_id].idxs:
                    if idx1 not in all_ins[ins_id].group_idxs[1]:
                        div1 = sys.float_info.max
                        for idx2 in all_ins[ins_id].idxs:
                            div1 = min(div1, dist(X[idx1], X[idx2]))
                        if div1 > max_div:
                            max_div = div1
                            max_idx = idx1
                all_ins[ins_id].idxs.add(max_idx)
                all_ins[ins_id].group_idxs[1].add(max_idx)
            while len(all_ins[ins_id].group_idxs[0]) > k[0]:
                min_div = sys.float_info.max
                min_idx = -1
                for idx1 in all_ins[ins_id].group_idxs[0]:
                    div1 = sys.float_info.max
                    for idx2 in all_ins[ins_id].idxs:
                        if idx1 != idx2:
                            div1 = min(div1, dist(X[idx1], X[idx2]))
                    if div1 < min_div:
                        min_div = div1
                        min_idx = idx1
                all_ins[ins_id].idxs.remove(min_idx)
                all_ins[ins_id].group_idxs[0].remove(min_idx)
            while len(all_ins[ins_id].group_idxs[1]) > k[1]:
                min_div = sys.float_info.max
                min_idx = -1
                for idx1 in all_ins[ins_id].group_idxs[1]:
                    div1 = sys.float_info.max
                    for idx2 in all_ins[ins_id].idxs:
                        if idx1 != idx2:
                            div1 = min(div1, dist(X[idx1], X[idx2]))
                    if div1 < min_div:
                        min_div = div1
                        min_idx = idx1
                all_ins[ins_id].idxs.remove(min_idx)
                all_ins[ins_id].group_idxs[1].remove(min_idx)
            ins_div = diversity(X, all_ins[ins_id].idxs, dist)
            if ins_div > sol_div:
                sol_idx = ins_id
                sol_div = ins_div
        else:
            while len(all_ins[ins_id].group_idxs[0]) < k[0]:
                max_div = 0.0
                max_idx = -1
                for idx1 in group_ins[0][ins_id].idxs:
                    if idx1 not in all_ins[ins_id].group_idxs[0]:
                        div1 = sys.float_info.max
                        for idx2 in all_ins[ins_id].group_idxs[0]:
                            div1 = min(div1, dist(X[idx1], X[idx2]))
                        if div1 > max_div:
                            max_div = div1
                            max_idx = idx1
                all_ins[ins_id].idxs.add(max_idx)
                all_ins[ins_id].group_idxs[0].add(max_idx)
            while len(all_ins[ins_id].group_idxs[1]) < k[1]:
                max_div = 0.0
                max_idx = -1
                for idx1 in group_ins[1][ins_id].idxs:
                    if idx1 not in all_ins[ins_id].group_idxs[1]:
                        div1 = sys.float_info.max
                        for idx2 in all_ins[ins_id].group_idxs[1]:
                            div1 = min(div1, dist(X[idx1], X[idx2]))
                        if div1 > max_div:
                            max_div = div1
                            max_idx = idx1
                all_ins[ins_id].idxs.add(max_idx)
                all_ins[ins_id].group_idxs[1].add(max_idx)
            while len(all_ins[ins_id].group_idxs[0]) > k[0]:
                min_div = sys.float_info.max
                min_idx = -1
                for idx1 in all_ins[ins_id].group_idxs[0]:
                    div1 = sys.float_info.max
                    for idx2 in all_ins[ins_id].group_idxs[1]:
                        div1 = min(div1, dist(X[idx1], X[idx2]))
                    if div1 < min_div:
                        min_div = div1
                        min_idx = idx1
                all_ins[ins_id].idxs.remove(min_idx)
                all_ins[ins_id].group_idxs[0].remove(min_idx)
            while len(all_ins[ins_id].group_idxs[1]) > k[1]:
                min_div = sys.float_info.max
                min_idx = -1
                for idx1 in all_ins[ins_id].group_idxs[1]:
                    div1 = sys.float_info.max
                    for idx2 in all_ins[ins_id].group_idxs[0]:
                        div1 = min(div1, dist(X[idx1], X[idx2]))
                    if div1 < min_div:
                        min_div = div1
                        min_idx = idx1
                all_ins[ins_id].idxs.remove(min_idx)
                all_ins[ins_id].group_idxs[1].remove(min_idx)
            ins_div = diversity(X, all_ins[ins_id].idxs, dist)
            if ins_div > sol_div:
                sol_idx = ins_id
                sol_div = ins_div
    t1 = time.perf_counter()
    # post_time = t1 - t0
    return all_ins[sol_idx].idxs, sol_div, num_elements, 0, t1 - t0


def StreamFairDivMax2(X: ElemList, k: List[int], m: int, dist: Callable[[Any, Any], float], eps: float, dmax: float, dmin: float) -> (Set[int], float, int, float, float):
    t0 = time.perf_counter()
    # Initialization
    sum_k = sum(k)
    # print(zmin, zmax)
    all_ins = []
    group_ins = []
    for c in range(m):
        group_ins.append(list())
    cur_d = dmax - 0.01
    while cur_d > dmin:
        ins = Instance(k=sum_k, mu=cur_d, m=m)
        all_ins.append(ins)
        for c in range(m):
            gins = Instance(k=sum_k, mu=cur_d, m=1)
            group_ins[c].append(gins)
        cur_d *= (1.0 - eps)
    # Stream processing
    for x in X:
        for ins in all_ins:
            if len(ins.idxs) == 0:
                ins.idxs.add(x.idx)
                ins.group_idxs[x.color].add(x.idx)
            elif len(ins.idxs) < ins.k:
                div_x = sys.float_info.max
                flag_x = True
                for idx_y in ins.idxs:
                    div_x = min(div_x, dist(x, X[idx_y]))
                    if div_x < ins.mu:
                        flag_x = False
                        break
                if flag_x:
                    ins.idxs.add(x.idx)
                    ins.group_idxs[x.color].add(x.idx)
                    ins.div = min(ins.div, div_x)
        for gins in group_ins[x.color]:
            if len(gins.idxs) == 0:
                gins.idxs.add(x.idx)
            elif len(gins.idxs) < gins.k:
                div_x = sys.float_info.max
                flag_x = True
                for idx_y in gins.idxs:
                    div_x = min(div_x, dist(x, X[idx_y]))
                    if div_x < gins.mu:
                        flag_x = False
                        break
                if flag_x:
                    gins.idxs.add(x.idx)
                    gins.div = min(gins.div, div_x)
    # t1 = time.perf_counter()
    # stream_time_per_elem = (t1 - t0) / len(X)
    stored_elements = set()
    for ins_id in range(len(all_ins)):
        stored_elements.update(all_ins[ins_id].idxs)
        for c in range(m):
            stored_elements.update(group_ins[c][ins_id].idxs)
    num_elements = len(stored_elements)
    # post-processing
    # t0 = time.perf_counter()
    sol = None
    sol_div = 0.0
    for ins_id in range(len(all_ins)):
        hasValidSol = True
        for c in range(m):
            if len(group_ins[c][ins_id].idxs) < k[c]:
                hasValidSol = False
                break
        if not hasValidSol:
            continue
        S_all = set()
        S_all.update(all_ins[ins_id].idxs)
        for c in range(m):
            S_all.update(group_ins[c][ins_id].idxs)
        G1 = nx.Graph()
        for idx1 in S_all:
            G1.add_node(idx1)
            for idx2 in S_all:
                if idx1 < idx2 and dist(X[idx1], X[idx2]) < all_ins[ins_id].mu / (m + 1):
                    G1.add_edge(idx1, idx2)
        P = []
        for p in nx.connected_components(G1):
            P.append(set(p))
        dict_par = dict()
        for j in range(len(P)):
            for s_idx in P[j]:
                dict_par[s_idx] = j
        S_prime = set()
        num_elem_col = np.zeros(m)
        for c in range(m):
            if len(all_ins[ins_id].group_idxs[c]) <= k[c]:
                S_prime.update(all_ins[ins_id].group_idxs[c])
                num_elem_col[c] = len(all_ins[ins_id].group_idxs[c])
            else:
                for s_idx in all_ins[ins_id].group_idxs[c]:
                    S_prime.add(s_idx)
                    num_elem_col[c] += 1
                    if num_elem_col[c] == k[c]:
                        break
        X1 = set()
        X2 = set()
        P_prime = set()
        if len(S_prime) < sum_k:
            for s_idx in S_prime:
                P_prime.add(dict_par[s_idx])
            for s_idx in S_all:
                s_col = X[s_idx].color
                s_par = dict_par[s_idx]
                if s_idx not in S_prime and num_elem_col[s_col] < k[s_col]:
                    X1.add(s_idx)
                if s_idx not in S_prime and s_par not in P_prime:
                    X2.add(s_idx)
            X12 = X1.intersection(X2)
            while len(X12) > 0:
                max_idx = -1
                max_div = 0.0
                for s_idx1 in X12:
                    s_div1 = sys.float_info.max
                    for s_idx2 in S_prime:
                        s_div1 = min(s_div1, dist(X[s_idx1], X[s_idx2]))
                    if s_div1 > max_div:
                        max_idx = s_idx1
                        max_div = s_div1
                max_col = X[max_idx].color
                max_par = dict_par[max_idx]
                S_prime.add(max_idx)
                num_elem_col[max_col] += 1
                # print(max_idx, max_col, max_par, S_prime, num_elem_col)
                if num_elem_col[max_col] == k[max_col]:
                    for s_idx in group_ins[max_col][ins_id].idxs:
                        X1.discard(s_idx)
                for s_idx in P[max_par]:
                    X2.discard(s_idx)
                X12 = X1.intersection(X2)
        while len(S_prime) < sum_k and len(X1) > 0 and len(X2) > 0:
            GA = nx.DiGraph()
            GA.add_node(-1)
            GA.add_node(len(X))
            for s_idx in X1:
                GA.add_node(s_idx)
                GA.add_edge(-1, s_idx)
            for s_idx in X2:
                GA.add_node(s_idx)
                GA.add_edge(s_idx, len(X))
            for s_idx1 in S_prime:
                GA.add_node(s_idx1)
                for s_idx2 in X1:
                    if X[s_idx1].color == X[s_idx2].color:
                        GA.add_edge(s_idx1, s_idx2)
                    if dict_par[s_idx1] == dict_par[s_idx2]:
                        GA.add_edge(s_idx2, s_idx1)
                for s_idx2 in X2:
                    if X[s_idx1].color == X[s_idx2].color:
                        GA.add_edge(s_idx1, s_idx2)
                    if dict_par[s_idx1] == dict_par[s_idx2]:
                        GA.add_edge(s_idx2, s_idx1)
            try:
                s_path = nx.shortest_path(GA, source=-1, target=len(X))
                for s_idx in s_path:
                    if -1 < s_idx < len(X):
                        if s_idx in S_prime:
                            S_prime.remove(s_idx)
                        else:
                            S_prime.add(s_idx)
                if len(S_prime) == sum_k:
                    break
                P_prime.clear()
                X1.clear()
                X2.clear()
                for s_idx in S_prime:
                    P_prime.add(dict_par[s_idx])
                for s_idx in S_all:
                    s_col = X[s_idx].color
                    s_par = dict_par[s_idx]
                    if s_idx not in S_prime and num_elem_col[s_col] < k[s_col]:
                        X1.add(s_idx)
                    if s_idx not in S_prime and s_par not in P_prime:
                        X2.add(s_idx)
            except nx.NetworkXNoPath:
                break
        if len(S_prime) == sum_k:
            div_s = diversity(X, S_prime, dist)
            if div_s > sol_div:
                sol = S_prime
                sol_div = div_s
    t1 = time.perf_counter()
    post_time = t1 - t0
    return sol, sol_div, num_elements, 0, t1 - t0


def diversity(X: ElemList, idxs: Iterable[int], dist: Callable[[Any, Any], float]) -> float:
    div_val = sys.float_info.max
    for idx1 in idxs:
        for idx2 in idxs:
            if idx1 != idx2:
                div_val = min(div_val, dist(X[idx1], X[idx2]))
    return div_val
