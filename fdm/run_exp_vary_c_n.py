import csv
import random
import numpy as np
import algorithms_offline as algo
import alg
import utils
import itertools
import algorithms_streaming as algs

'''
    experiment-vary-c-synthetic:
    n = 1000
    k = 20
    c = 2,4,6,8,...,20
    eps = 0.05
'''

output = open("results_vary_c_n_together.csv", "a")
writer = csv.writer(output)
writer.writerow(["dataset", "group", "m", "k", "algorithm", "param_eps", "div", "num_elem", "time1", "time2", "time3"])
output.flush()

# experiments for varying c
num_runs = 10
EPS = 0.05
k = 20
elements = []

group_c = {2: [10, 10],
           4: [6, 5, 4, 5],
           6: [4, 3, 4, 3, 3, 3],
           8: [2, 3, 2, 3, 3, 3, 2, 2],
           10: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
           12: [2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2],
           14: [1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2],
           16: [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1],
           18: [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
           20: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
constr = {2: [[8, 12], [8, 12]], 4: [[4, 8], [4, 6], [3, 5], [4, 6]], 6: [[3, 5], [2, 4], [3, 5], [2, 4], [2, 4], [2, 4]],
          8: [[1, 3], [2, 4], [1, 3], [2, 4], [2, 4], [2, 4], [1, 3], [1, 3]], 10: [[1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3]],
          12: [[1, 3], [1, 2], [1, 2], [1, 3], [1, 2], [1, 3], [1, 3], [1, 2], [1, 3], [1, 3], [1, 3], [1, 3]],
          14: [[1, 2], [1, 2], [1, 3], [1, 2], [1, 2], [1, 3], [1, 3], [1, 2], [1, 3], [1, 2], [1, 2], [1, 2], [1, 3], [1, 3]],
          16: [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 3], [1, 3], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 3], [1, 3], [1, 2], [1, 2]],
          18: [[1, 2], [1, 2], [1, 2], [1, 2], [1, 3], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 3]],
          20: [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]}

range_d = [1.8, 4.6]
for c in range(2, 21, 2):
    elements.clear()
    with open("./data/blobs_n1000_c" + str(c) + ".csv", "r") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in csvreader:
            features = []
            for i in range(2, len(row)):
                features.append(float(row[i]))
            elem = utils.Elem(int(row[0]), int(row[1]), features)
            elements.append(elem)
    alg1 = np.zeros([4, num_runs])
    alg2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    fair_swap = np.zeros([2, num_runs])
    fair_flow = np.zeros([2, num_runs])
    fmmd_ILP = np.zeros(2)
    scalable_fmmd_ILP = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        if c == 2:
            sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements, k=group_c[c], dist=utils.euclidean_dist)
            sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=group_c[c], dist=utils.euclidean_dist, eps=EPS, dmax=range_d[1], dmin=range_d[0])
            print(sol)
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_c[c], m=c, dist=utils.euclidean_dist)
        sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, k=k, C=c, EPS=EPS, constr=constr[c], dist=utils.euclidean_dist)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_c[c], m=c, dist=utils.euclidean_dist, eps=EPS, dmax=range_d[1], dmin=range_d[0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_c[c], m=c, dist=utils.euclidean_dist, eps=EPS, dmax=range_d[1], dmin=range_d[0], metric_name='euclidean')
        print(sol)
    sol, fmmd_ILP[0], fmmd_ILP[1] = alg.fmmd_ILP(V=elements, k=k, C=c, constr=constr[c], dist=utils.euclidean_dist)

    if c == 2:
        writer.writerow(["Blobs_1000", "-", c, k, "Alg1", EPS, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]), np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
        writer.writerow(["Blobs_1000", "-", c, k, "FairSwap", np.average(fair_swap[0]), np.average(fair_swap[1])])

    writer.writerow(["Blobs_1000", "-", c, k, "Alg2", EPS, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    writer.writerow(["Blobs_1000", "-", c, k, "FairGreedyFlow", EPS, np.average(fair_greedy_flow[0]), "-", "-", "-", np.average(fair_greedy_flow[1])])
    writer.writerow(["Blobs_1000", "-", c, k, "FairFlow", EPS, np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["Blobs_1000", "-", c, k, "fmmd_ILP", EPS, fmmd_ILP[0], "-", "-", "-", fmmd_ILP[1]])
    writer.writerow(["Blobs_1000", "-", c, k, "scalable_fmmd_ILP", EPS, np.average(scalable_fmmd_ILP[0]), "-", "-", "-", np.average(scalable_fmmd_ILP[1])])

'''
    experiment-vary-n-synthetic:
    n = 100,1000,10000,...,10000000
    k = 20
    eps = 0.05
    c = 2
'''

num_runs = 10
k = 20
c = 2
elements.clear()
elements = []
group = [10, 10]
constr = [[8, 12], [8, 12]]
num_elem = [100, 1000, 10000, 100000, 1000000, 10000000]
range_d_n = {100: [1.23, 3.08], 1000: [1.70, 4.25], 10000: [1.99, 4.97],
             100000: [2.10, 5.26], 1000000: [2.42, 6.06], 10000000: [2.36, 5.9]}
for n in num_elem:
    elements.clear()
    with open("./data/blobs_n10000000_c2.csv", "r") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in itertools.islice(csvreader, n):
            features = []
            for i in range(2, len(row)):
                features.append(float(row[i]))
            elem = utils.Elem(int(row[0]), int(row[1]), features)
            elements.append(elem)
    fair_swap = np.zeros([2, num_runs])
    fair_flow = np.zeros([2, num_runs])
    fmmd_ILP = np.zeros(2)
    scalable_fmmd_ILP = np.zeros([2, num_runs])
    alg1 = np.zeros([4, num_runs])
    alg2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])

    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group, m=c, dist=utils.euclidean_dist)
        sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, k=k, C=c, EPS=EPS, constr=constr, dist=utils.euclidean_dist)
        sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements, k=group, dist=utils.euclidean_dist)
        sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=group, dist=utils.euclidean_dist, eps=EPS, dmax=range_d_n[n][1], dmin=range_d_n[n][0])
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group, m=c, dist=utils.euclidean_dist, eps=EPS, dmax=range_d_n[n][1], dmin=range_d_n[n][0])
        print(sol)

        if n <= 100_000:
            sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group, m=c, dist=utils.euclidean_dist, eps=EPS, dmax=range_d_n[n][1], dmin=range_d_n[n][0], metric_name='euclidean')
            print(sol)
    writer.writerow(["Blobs", n, c, k, "Alg1", EPS, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]), np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
    writer.writerow(["Blobs", n, c, k, "Alg2", EPS, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    writer.writerow(["Blobs", n, c, k, "scalable_fmmd_ILP", EPS, np.average(scalable_fmmd_ILP[0]), "-", "-", "-", np.average(scalable_fmmd_ILP[1])])
    writer.writerow(["Blobs", n, c, k, "FairFlow", EPS, np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["Blobs", n, c, k, "FairSwap", EPS, np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
    if n <= 100_000:
        writer.writerow(["Blobs", n, c, k, "FairGreedyFlow", EPS, np.average(fair_greedy_flow[0]), "-", "-", "-", np.average(fair_greedy_flow[1])])
    if n <= 1_000:
        sol, fmmd_ILP[0], fmmd_ILP[1] = alg.fmmd_ILP(V=elements, k=k, C=c, constr=constr, dist=utils.euclidean_dist)
        writer.writerow(["Blobs", n, c, k, "fmmd_ILP", EPS, fmmd_ILP[0], "-", "-", "-", fmmd_ILP[1]])
    output.flush()

'''
    experiment-vary-n-synthetic:
    n = 100,1000,10000,...,10000000
    k = 20
    eps = 0.05
    c = 10
'''

num_runs = 10
k = 20
c = 10
elements = []
group = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
constr = [[1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3]]
num_elem = [100, 1000, 10000, 100000, 1000000, 10000000]
range_d_n = {100: [1.23, 3.08], 1000: [1.70, 4.25], 10000: [1.99, 4.97],
             100000: [2.10, 5.26], 1000000: [2.42, 6.06], 10000000: [2.36, 5.9]}
for n in num_elem:
    elements.clear()
    with open("./data/blobs_n10000000_c10.csv", "r") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in itertools.islice(csvreader, n):
            features = []
            for i in range(2, len(row)):
                features.append(float(row[i]))
            elem = utils.Elem(int(row[0]), int(row[1]), features)
            elements.append(elem)
    alg2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    fair_flow = np.zeros([2, num_runs])
    fmmd_ILP = np.zeros(2)
    scalable_fmmd_ILP = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group, m=c, dist=utils.euclidean_dist, eps=EPS, dmax=range_d_n[n][1], dmin=range_d_n[n][0])
        print(sol)
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group, m=c, dist=utils.euclidean_dist)
        sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, k=k, C=c, EPS=EPS, constr=constr, dist=utils.euclidean_dist)
        if n <= 100_000:
            sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group, m=c, dist=utils.euclidean_dist, eps=EPS, dmax=range_d_n[n][1], dmin=range_d_n[n][0], metric_name='euclidean')
            print(sol)
    writer.writerow(["Blobs", n, c, k, "Alg2", EPS, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    writer.writerow(["Blobs", n, c, k, "scalable_fmmd_ILP", EPS, np.average(scalable_fmmd_ILP[0]), "-", "-", "-", np.average(scalable_fmmd_ILP[1])])
    writer.writerow(["Blobs", n, c, k, "FairFlow", EPS, np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    if n <= 100_000:
        writer.writerow(["Blobs", n, c, k, "FairGreedyFlow", EPS, np.average(fair_greedy_flow[0]), "-", "-", "-",
                         np.average(fair_greedy_flow[1])])
    if n <= 1_000:
        sol, fmmd_ILP[0], fmmd_ILP[1] = alg.fmmd_ILP(V=elements, k=k, C=c, constr=constr, dist=utils.euclidean_dist)
        writer.writerow(["Blobs", n, c, k, "fmmd_ILP", EPS, fmmd_ILP[0], "-", "-", "-", fmmd_ILP[1]])
    output.flush()
