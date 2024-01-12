import sys
import time
from typing import Any, Callable, List, Union

import networkx as nx
import numpy as np
from scipy.special import comb
import math
import algorithms.utilsfdm as utilsfdm
import random
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from algorithms.utils import Stopwatch

ElemList = Union[List[utilsfdm.Elem], List[utilsfdm.ElemSparse]]
TIME_LIMIT_ILP = 300

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

def StreamFairDivMax2(X: ElemList, k: List[int], m: int, dist: Callable[[Any, Any], float], eps: float, dmax: float, dmin: float):

    stream_size = len(X)
    # print(f'[SFDM2] Stream size = {stream_size}')

    timer = Stopwatch("Stream time")

    # Start of streaming
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
    # End of Streaming
             
    _, stream_time = timer.stop()
    # print(f'Streaming in time ', stream_time, " of size ",stream_size)  

    timer = Stopwatch("Post Time")
    # Start post processing
    stored_elements = set()
    for ins_id in range(len(all_ins)):
        stored_elements.update(all_ins[ins_id].idxs)
        for c in range(m):
            stored_elements.update(group_ins[c][ins_id].idxs)
    num_elements = len(stored_elements)
    # post-processing
    t0 = time.perf_counter()
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
    # End post processing    
    _, post_time = timer.stop()

    stream_time_per_elem = float(stream_time)/float(stream_size)

    total_time = stream_time + post_time

    # print(f'[SFDM2] total stream time = {stream_time}')
    # print(f'[SFDM2] post time = {post_time}')
    # print(f'[SFDM2] stream size = {stream_size}')

    return sol, sol_div, stream_time_per_elem, post_time, total_time

def scalable_fmmd_ILP(V: ElemList, EPS: float, k: int, C: int, constr: List[List[int]], dist: Callable[[Any, Any], float]) -> (List[int], float, float):
    t0 = time.perf_counter()

    # Initialization
    sol = list()
    div_sol = 0.0

    # The Gonzalez's algorithm
    cand = set()
    cand_dists = dict()
    cand_div = sys.float_info.max
    array_dists = [sys.float_info.max] * len(V)

    cand.add(0)
    cand_dists[0] = dict()
    for i in range(len(V)):
        array_dists[i] = dist(V[0], V[i])
    while len(cand) < k:
        max_idx = np.argmax(array_dists)
        max_dist = np.max(array_dists)
        cand.add(max_idx)
        cand_dists[max_idx] = dict()
        for idx in cand:
            if idx < max_idx:
                cand_dists[idx][max_idx] = dist(V[idx], V[max_idx])
            elif idx > max_idx:
                cand_dists[max_idx][idx] = dist(V[idx], V[max_idx])
        cand_div = min(cand_div, max_dist)
        for i in range(len(V)):
            array_dists[i] = min(array_dists[i], dist(V[i], V[max_idx]))

    # Divide candidates by colors
    cand_colors = list()
    for c in range(C):
        cand_colors.append(set())
    for idx in cand:
        c = V[idx].color
        cand_colors[c].add(idx)

    # Compute the solution
    div = cand_div
    while len(sol) == 0:
        under_capped = False
        for c in range(C):
            # Add an arbitrary element of color c when there is not anyone in the candidate.
            if len(cand_colors[c]) == 0:
                for i in range(len(V)):
                    if V[i].color == c:
                        cand_colors[c].add(i)
                        cand.add(i)
                        cand_dists[i] = dict()
                        for idx in cand:
                            if idx < i:
                                cand_dists[idx][i] = dist(V[idx], V[i])
                            elif idx > i:
                                cand_dists[i][idx] = dist(V[idx], V[i])
                        break
            # The Gonzalez's algorithm starting from cand_colors[c] on all elements of color c
            array_dists_color = [sys.float_info.max] * len(V)
            for i in range(len(V)):
                for j in cand_colors[c]:
                    if V[i].color == c:
                        array_dists_color[i] = min(array_dists_color[i], dist(V[i], V[j]))
                    else:
                        array_dists_color[i] = 0.0
            max_idx_c = np.argmax(array_dists_color)
            max_dist_c = np.max(array_dists_color)
            while len(cand_colors[c]) < k and max_dist_c > div:
                cand_colors[c].add(max_idx_c)
                cand.add(max_idx_c)
                cand_dists[max_idx_c] = dict()
                for idx in cand:
                    if idx < max_idx_c:
                        cand_dists[idx][max_idx_c] = dist(V[idx], V[max_idx_c])
                    elif idx > max_idx_c:
                        cand_dists[max_idx_c][idx] = dist(V[idx], V[max_idx_c])
                for i in range(len(V)):
                    if V[i].color == c:
                        array_dists_color[i] = min(array_dists_color[i], dist(V[i], V[max_idx_c]))
                max_idx_c = np.argmax(array_dists_color)
                max_dist_c = np.max(array_dists_color)
            if len(cand_colors[c]) < constr[c][0]:
                under_capped = True
                break
        if under_capped:
            div = div * (1.0 - EPS)
            continue

        # Build a graph G w.r.t. cand_div
        dict_cand = dict()
        new_idx = 0
        for idx in cand:
            dict_cand[idx] = new_idx
            new_idx += 1
        G = nx.Graph()
        G.add_nodes_from(range(len(cand)))
        for i in cand_dists.keys():
            for j in cand_dists[i].keys():
                if cand_dists[i][j] < div:
                    G.add_edge(dict_cand[i], dict_cand[j])

        # Find an independent set S of G using ILP
        try:
            model = gp.Model("mis_" + str(div))
            model.setParam(GRB.Param.TimeLimit, TIME_LIMIT_ILP)

            size = [1] * len(cand)
            vars_x = model.addVars(len(cand), vtype=GRB.BINARY, obj=size, name="x")

            model.modelSense = GRB.MAXIMIZE

            eid = 0
            for e in G.edges:
                model.addConstr(vars_x[e[0]] + vars_x[e[1]] <= 1, "edge_" + str(eid))
                eid += 1

            expr = gp.LinExpr()
            for j in range(len(cand)):
                expr.addTerms(1, vars_x[j])
            model.addConstr(expr <= k, "size")

            for c in range(C):
                expr = gp.LinExpr()
                for j in cand_colors[c]:
                    expr.addTerms(1, vars_x[dict_cand[j]])
                model.addConstr(expr >= constr[c][0], "lb_color_" + str(c))
                model.addConstr(expr <= constr[c][1], "ub_color_" + str(c))

            model.optimize()

            S = set()
            for j in range(len(cand)):
                if vars_x[j].X > 0.5:
                    S.add(j)

            if len(S) >= k:
                for key, value in dict_cand.items():
                    if value in S:
                        sol.append(key)
                div_sol = utilsfdm.div_subset(V, sol, dist)
                break
            else:
                div = div * (1.0 - EPS)

        except gp.GurobiError as error:
            print("Error code " + str(error))
            exit(0)

        except AttributeError:
            div = div * (1.0 - EPS)

    t1 = time.perf_counter()
    return sol, div_sol, t1 - t0


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
