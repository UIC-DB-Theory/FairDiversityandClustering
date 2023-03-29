import functools
import sys
import time
from typing import Any, Callable, List, Union

import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB

import utils

ElemList = Union[List[utils.Elem], List[utils.ElemSparse]]
TIME_LIMIT_ILP = 300


@functools.total_ordering
class DistToken:

    def __init__(self, u: int, v: int, d_uv: float):
        self.u, self.v, self.d_uv = u, v, d_uv

    def __lt__(self, other):
        if self.d_uv - other.d_uv > 1.0e-6:
            return True
        elif self.d_uv - other.d_uv < -1.0e-6:
            return False
        elif self.u != other.u:
            return self.u < other.u
        else:
            return self.v < other.v

    def __gt__(self, other):
        if self.d_uv - other.d_uv < -1.0e-6:
            return True
        elif self.d_uv - other.d_uv > 1.0e-6:
            return False
        elif self.u != other.u:
            return self.u > other.u
        else:
            return self.v > other.v

    def __eq__(self, other):
        return (self.u, self.v) == (other.u, other.v)


def ILP(V: ElemList, k: int, dist: Callable[[Any, Any], float]) -> (List[int], float, float):
    t0 = time.perf_counter()

    # Initialization
    sol = list()
    div_sol = 0.0

    # Compute and sort the pairwise distances between elements
    dists = list()
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            dists.append(DistToken(i, j, dist(V[i], V[j])))

    dists.sort()

    # Binary search to find the optimal diversity
    idx_max = 0
    idx_min = len(dists) - 1
    idx = (idx_max + idx_min) // 2
    # print(idx_max, dists[idx_max].d_uv, idx_min, dists[idx_min].d_uv, idx, dists[idx].d_uv)
    while idx_max < idx_min - 1:
        # Build a graph G w.r.t. idx
        G = nx.Graph()
        G.add_nodes_from(range(len(V)))
        for i in range(len(dists) - 1, idx, -1):
            G.add_edge(dists[i].u, dists[i].v)
        # Find an independent set S of G using ILP
        try:
            model = gp.Model("mis_" + str(idx))
            model.setParam(GRB.Param.TimeLimit, TIME_LIMIT_ILP)

            size = [1] * len(V)
            vars_x = model.addVars(len(V), vtype=GRB.BINARY, obj=size, name="x")

            model.modelSense = GRB.MAXIMIZE

            eid = 0
            for e in G.edges:
                model.addConstr(vars_x[e[0]] + vars_x[e[1]] <= 1, "edge_" + str(eid))
                eid += 1

            expr = gp.LinExpr()
            for j in range(len(V)):
                expr.addTerms(1, vars_x[j])
            model.addConstr(expr <= k, "size")

            model.optimize()

            S = list()
            for j in range(len(V)):
                if vars_x[j].X > 0.5:
                    S.append(j)
            print(S)
            print("Obj: %g" % model.objVal)

            if len(S) >= k:
                if len(sol) == 0:
                    sol.extend(S)
                    div_sol = utils.div_subset(V, S, dist)
                else:
                    div_S = utils.div_subset(V, S, dist)
                    if div_S > div_sol:
                        sol.clear()
                        sol.extend(S)
                        div_sol = div_S
                idx_min = idx
                idx = (idx_max + idx_min) // 2
            else:
                idx_max = idx
                idx = (idx_max + idx_min) // 2

        except gp.GurobiError as error:
            print("Error code " + str(error))
            exit(0)

        except AttributeError:
            idx_max = idx
            idx = (idx_max + idx_min) // 2

    t1 = time.perf_counter()
    return sol, div_sol, t1 - t0


def fmmd_ILP(V: ElemList, k: int, C: int, constr: List[List[int]], dist: Callable[[Any, Any], float]) -> (List[int], float, float):
    t0 = time.perf_counter()

    # Initialization
    sol = list()
    div_sol = 0.0

    # Divide elements by colors
    V_colors = list()
    for c in range(C):
        V_colors.append(list())
    for i in range(len(V)):
        V_colors[V[i].color].append(i)

    # Compute and sort the pairwise distances between elements
    dists = list()
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            dists.append(DistToken(i, j, dist(V[i], V[j])))
    dists.sort()

    # Binary search to find the optimal diversity
    idx_max = 0
    idx_min = len(dists) - 1
    idx = (idx_max + idx_min) // 2
    # print(idx_max, dists[idx_max].d_uv, idx_min, dists[idx_min].d_uv, idx, dists[idx].d_uv)
    while idx_max < idx_min - 1:
        # Build a graph G w.r.t. idx
        G = nx.Graph()
        G.add_nodes_from(range(len(V)))
        for i in range(len(dists) - 1, idx, -1):
            G.add_edge(dists[i].u, dists[i].v)
        # Find an independent set S of G using ILP
        try:
            model = gp.Model("mis_" + str(idx))
            model.setParam(GRB.Param.TimeLimit, TIME_LIMIT_ILP)

            size = [1] * len(V)
            vars_x = model.addVars(len(V), vtype=GRB.BINARY, obj=size, name="x")

            model.modelSense = GRB.MAXIMIZE

            eid = 0
            for e in G.edges:
                model.addConstr(vars_x[e[0]] + vars_x[e[1]] <= 1, "edge_" + str(eid))
                eid += 1

            expr = gp.LinExpr()
            for j in range(len(V)):
                expr.addTerms(1, vars_x[j])
            model.addConstr(expr <= k, "size")

            for c in range(C):
                expr = gp.LinExpr()
                for j in V_colors[c]:
                    expr.addTerms(1, vars_x[j])
                model.addConstr(expr >= constr[c][0], "lb_color_" + str(c))
                model.addConstr(expr <= constr[c][1], "ub_color_" + str(c))

            model.optimize()

            S = list()
            for j in range(len(V)):
                if vars_x[j].X > 0.5:
                    S.append(j)
            # print(S)
            # print("Obj: %g" % model.objVal)

            if len(S) >= k:
                if len(sol) == 0:
                    sol.extend(S)
                    div_sol = utils.div_subset(V, S, dist)
                else:
                    div_S = utils.div_subset(V, S, dist)
                    if div_S > div_sol:
                        sol.clear()
                        sol.extend(S)
                        div_sol = div_S
                idx_min = idx
                idx = (idx_max + idx_min) // 2
            else:
                idx_max = idx
                idx = (idx_max + idx_min) // 2

        except gp.GurobiError as error:
            print("Error code " + str(error))
            exit(0)

        except AttributeError:
            idx_max = idx
            idx = (idx_max + idx_min) // 2

        # print(idx_max, dists[idx_max].d_uv, idx_min, dists[idx_min].d_uv, idx, dists[idx].d_uv)
    t1 = time.perf_counter()
    return sol, div_sol, t1 - t0


def fmmd_modified_greedy(V: ElemList, k: int, C: int, constr: List[List[int]], dist: Callable[[Any, Any], float]) -> (List[int], float, float):
    t0 = time.perf_counter()

    # Initialization
    sol = list()
    div_sol = 0.0

    # Compute and sort the pairwise distances between elements
    dists = list()
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            dists.append(DistToken(i, j, dist(V[i], V[j])))
    dists.sort()

    # Binary search to find an appropriate diversity
    idx_max = 0
    idx_min = len(dists) - 1
    idx = (idx_max + idx_min) // 2
    # print(idx_max, dists[idx_max].d_uv, idx_min, dists[idx_min].d_uv, idx, dists[idx].d_uv)
    while idx_max < idx_min - 1:
        # Build a graph G w.r.t. idx
        G = nx.Graph()
        G.add_nodes_from(range(len(V)))
        for i in range(len(dists) - 1, idx, -1):
            G.add_edge(dists[i].u, dists[i].v)
        # Find an independent set S of G using the modified greedy algorithm
        S = list()
        S_colors = [0] * C
        Q = 0
        for c in range(C):
            Q += constr[c][0]
        non_ext_colors = set()
        under_capped_colors = set()
        for c in range(C):
            if constr[c][0] > 0:
                under_capped_colors.add(c)

        min_deg = sys.maxsize
        min_node = -1
        for i in G.nodes:
            deg_i = G.degree[i]
            if deg_i < min_deg:
                min_node = i
                min_deg = deg_i
            if min_deg == 0:
                break
        c = V[min_node].color
        S.append(min_node)
        S_colors[c] += 1
        if constr[c][0] <= 1:
            under_capped_colors.remove(c)
        if constr[c][1] == 1:
            non_ext_colors.add(c)
        to_be_removed = list()
        for n in G.neighbors(min_node):
            to_be_removed.append(n)
        G.remove_node(min_node)
        for n in to_be_removed:
            G.remove_node(n)

        while len(S) < k and G.number_of_nodes() > 0:
            min_deg = sys.maxsize
            min_node = -1
            uc_min_deg = sys.maxsize
            uc_min_node = -1
            for i in G.nodes:
                deg_i = G.degree[i]
                c_i = V[i].color
                if c_i in under_capped_colors:
                    if deg_i < uc_min_deg:
                        uc_min_node = i
                        uc_min_deg = deg_i
                    if uc_min_deg == 0:
                        break
                elif c_i not in non_ext_colors:
                    if deg_i < min_deg:
                        min_node = i
                        min_deg = deg_i
            if uc_min_node >= 0:
                c = V[uc_min_node].color
                S.append(uc_min_node)
                S_colors[c] += 1
                if S_colors[c] >= constr[c][0]:
                    under_capped_colors.remove(c)
                if S_colors[c] == constr[c][1] or (S_colors[c] >= constr[c][0] and Q >= k):
                    non_ext_colors.add(c)
                to_be_removed = list()
                for n in G.neighbors(uc_min_node):
                    to_be_removed.append(n)
                G.remove_node(uc_min_node)
                for n in to_be_removed:
                    G.remove_node(n)
            elif len(under_capped_colors) > 0 or min_node < 0:
                break
            else:
                c = V[min_node].color
                S.append(min_node)
                S_colors[c] += 1
                if S_colors[c] > constr[c][0]:
                    Q += 1
                if S_colors[c] == constr[c][1] or (S_colors[c] >= constr[c][0] and Q >= k):
                    non_ext_colors.add(c)
                to_be_removed = list()
                for n in G.neighbors(min_node):
                    to_be_removed.append(n)
                G.remove_node(min_node)
                for n in to_be_removed:
                    G.remove_node(n)
        # print(S, S_colors)
        if len(S) == k:
            if len(sol) == 0:
                sol.extend(S)
                div_sol = utils.div_subset(V, S, dist)
            else:
                div_S = utils.div_subset(V, S, dist)
                if div_S > div_sol:
                    sol.clear()
                    sol.extend(S)
                    div_sol = div_S
            idx_min = idx
            idx = (idx_max + idx_min) // 2
        else:
            idx_max = idx
            idx = (idx_max + idx_min) // 2
        # print(idx_max, dists[idx_max].d_uv, idx_min, dists[idx_min].d_uv, idx, dists[idx].d_uv)
    t1 = time.perf_counter()
    return sol, div_sol, t1 - t0


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
                div_sol = utils.div_subset(V, sol, dist)
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


def scalable_fmmd_modified_greedy(V: ElemList, EPS: float, k: int, C: int, constr: List[List[int]], dist: Callable[[Any, Any], float]) -> (List[int], float, float):
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
        G = nx.Graph()
        G.add_nodes_from(cand)
        for i in cand_dists.keys():
            for j in cand_dists[i].keys():
                if cand_dists[i][j] < div:
                    G.add_edge(i, j)

        # Find an independent set S of G using modified greedy
        S = list()
        S_colors = [0] * C
        Q = 0
        for c in range(C):
            Q += constr[c][0]
        non_ext_colors = set()
        under_capped_colors = set()
        for c in range(C):
            if constr[c][0] > 0:
                under_capped_colors.add(c)

        min_deg = sys.maxsize
        min_node = -1
        for i in G.nodes:
            deg_i = G.degree[i]
            if deg_i < min_deg:
                min_node = i
                min_deg = deg_i
            if min_deg == 0:
                break
        c = V[min_node].color
        S.append(min_node)
        S_colors[c] += 1
        if constr[c][0] <= 1:
            under_capped_colors.remove(c)
        if constr[c][1] == 1:
            non_ext_colors.add(c)
        to_be_removed = list()
        for n in G.neighbors(min_node):
            to_be_removed.append(n)
        G.remove_node(min_node)
        for n in to_be_removed:
            G.remove_node(n)

        while len(S) < k and G.number_of_nodes() > 0:
            min_deg = sys.maxsize
            min_node = -1
            uc_min_deg = sys.maxsize
            uc_min_node = -1
            for i in G.nodes:
                deg_i = G.degree[i]
                c_i = V[i].color
                if c_i in under_capped_colors:
                    if deg_i < uc_min_deg:
                        uc_min_node = i
                        uc_min_deg = deg_i
                    if uc_min_deg == 0:
                        break
                elif c_i not in non_ext_colors:
                    if deg_i < min_deg:
                        min_node = i
                        min_deg = deg_i
            if uc_min_node >= 0:
                c = V[uc_min_node].color
                S.append(uc_min_node)
                S_colors[c] += 1
                if S_colors[c] >= constr[c][0]:
                    under_capped_colors.remove(c)
                if S_colors[c] == constr[c][1] or (S_colors[c] >= constr[c][0] and Q >= k):
                    non_ext_colors.add(c)
                to_be_removed = list()
                for n in G.neighbors(uc_min_node):
                    to_be_removed.append(n)
                G.remove_node(uc_min_node)
                for n in to_be_removed:
                    G.remove_node(n)
            elif len(under_capped_colors) > 0 or min_node < 0:
                break
            else:
                c = V[min_node].color
                S.append(min_node)
                S_colors[c] += 1
                if S_colors[c] > constr[c][0]:
                    Q += 1
                if S_colors[c] == constr[c][1] or (S_colors[c] >= constr[c][0] and Q >= k):
                    non_ext_colors.add(c)
                to_be_removed = list()
                for n in G.neighbors(min_node):
                    to_be_removed.append(n)
                G.remove_node(min_node)
                for n in to_be_removed:
                    G.remove_node(n)
        if len(S) >= k:
            sol.extend(S)
            div_sol = utils.div_subset(V, sol, dist)
            break
        else:
            div = div * (1.0 - EPS)

    t1 = time.perf_counter()
    return sol, div_sol, t1 - t0
