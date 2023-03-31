import csv
import random
import numpy as np
import algorithms_offline as algo
import alg
import utils
import scipy.sparse as sp

'''
    experiment-vary-eps-small:
    n = 1000
    eps = 0.001,0.005,0.01,0.05,0.1,0.5

'''
output = open("results_vary_k_eps.csv", "a")
writer = csv.writer(output)
writer.writerow(["dataset", "group", "c", "k", "EPS", "algorithm", "num", "div", "time"])
output.flush()

# read the Census dataset grouped by sex (c=2)
elements = []
with open("./data/census_small.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Elem(int(row[0]), int(row[1]), features)
        elements.append(elem)
# experiments on varying k
num_runs = 10
k = 10
c = 2
group_k = [5, 5]
constr = [[4, 6], [4, 6]]
values_eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
for eps in values_eps:
    scalable_fmmd_ILP = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, EPS=eps, k=k, C=c, constr=constr, dist=utils.manhattan_dist)
        writer.writerow(["Census_small", "Sex", c, k, eps, "scalable_fmmd_ILP", run, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run]])
    output.flush()

# read the Census dataset grouped by age (c=7)
elements.clear()
with open("./data/census_small.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Elem(int(row[0]), int(row[2]), features)
        elements.append(elem)

c = 7
group_k = [2, 1, 2, 2, 1, 1, 1]
constr = [[1, 3], [1, 2], [1, 3], [1, 3], [1, 2], [1, 2], [1, 2]]
values_eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
for eps in values_eps:
    scalable_fmmd_ILP = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, EPS=eps, k=k, C=c, constr=constr, dist=utils.manhattan_dist)
        writer.writerow(["Census_small", "Age", c, k, eps, "scalable_fmmd_ILP", run, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run]])
    output.flush()

# read the twitter dataset grouped by gender (c=3)
elements.clear()
elements = []
with open("./data/twitter_small.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Elem(int(row[0]), int(row[1]), features)
        elements.append(elem)

num_runs = 10
k = 10
c = 3
group_k = [3, 4, 3]
constr = [[2, 4], [3, 5], [2, 4]]
values_eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
for eps in values_eps:
    scalable_fmmd_ILP = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, EPS=eps, k=k, C=c, constr=constr, dist=utils.cosine_dist)
        writer.writerow(["twitter_small", "Sex", c, k, eps, "scalable_fmmd_ILP", run, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run]])
    output.flush()

# read the celebA dataset grouped by sex (c=2)
elements.clear()
elements = []
csr_sparse = sp.load_npz('../data/celebA_small_csr_sparse.npz')
for i in range(1000):
    if int(csr_sparse[i, 1]) == -1:
        elem = utils.ElemSparse(int(csr_sparse[i, 0]), 0, csr_sparse[i, 4:])
    else:
        elem = utils.ElemSparse(int(csr_sparse[i, 0]), int(csr_sparse[i, 1]), csr_sparse[i, 4:])
    elements.append(elem)

num_runs = 10
k = 10
c = 2
group_k = [6, 4]
constr = [[4, 8], [3, 5]]
values_eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
for eps in values_eps:
    scalable_fmmd_ILP = np.zeros([2, num_runs])
    scalable_fmmd_modified_greedy = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, scalable_fmmd_modified_greedy[0][run], scalable_fmmd_modified_greedy[1][run] = alg.scalable_fmmd_modified_greedy(V=elements, EPS=eps, k=k, C=c, constr=constr,
                                                                                                                              dist=utils.manhattan_dist_sparse)
        writer.writerow(["celebA_small", "Sex", c, k, eps, "scalable_fmmd_modified_greedy", run, scalable_fmmd_modified_greedy[0][run], scalable_fmmd_modified_greedy[1][run]])
        sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, EPS=eps, k=k, C=c, constr=constr, dist=utils.manhattan_dist_sparse)
        writer.writerow(["celebA_small", "Sex", c, k, eps, "scalable_fmmd_ILP", run, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run]])
    output.flush()

# read the celebA dataset grouped by age (c=2)
elements.clear()
elements = []
csr_sparse = sp.load_npz('../data/celebA_small_csr_sparse.npz')
for i in range(1000):
    if int(csr_sparse[i, 2]) == -1:
        elem = utils.ElemSparse(int(csr_sparse[i, 0]), 0, csr_sparse[i, 4:])
    else:
        elem = utils.ElemSparse(int(csr_sparse[i, 0]), int(csr_sparse[i, 2]), csr_sparse[i, 4:])
    elements.append(elem)

c = 2
group_k = [2, 8]
constr = [[1, 3], [6, 10]]

values_eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
for eps in values_eps:
    scalable_fmmd_ILP = np.zeros([2, num_runs])
    scalable_fmmd_modified_greedy = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, scalable_fmmd_modified_greedy[0][run], scalable_fmmd_modified_greedy[1][run] = alg.scalable_fmmd_modified_greedy(V=elements, EPS=eps, k=k, C=c, constr=constr,
                                                                                                                              dist=utils.manhattan_dist_sparse)
        writer.writerow(["celebA_small", "Age", c, k, eps, "scalable_fmmd_modified_greedy", run, scalable_fmmd_modified_greedy[0][run], scalable_fmmd_modified_greedy[1][run]])
        sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, EPS=eps, k=k, C=c, constr=constr, dist=utils.manhattan_dist_sparse)
        writer.writerow(["celebA_small", "Age", c, k, eps, "scalable_fmmd_ILP", run, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run]])
    output.flush()

# read the celebA dataset grouped by sex+age (c=4)
elements.clear()
elements = []
csr_sparse = sp.load_npz('../data/celebA_small_csr_sparse.npz')
for i in range(1000):
    if int(csr_sparse[i, 3]) == -1:
        elem = utils.ElemSparse(int(csr_sparse[i, 0]), 0, csr_sparse[i, 4:])
    else:
        elem = utils.ElemSparse(int(csr_sparse[i, 0]), int(csr_sparse[i, 3]), csr_sparse[i, 4:])
    elements.append(elem)

c = 4
group_k = [1, 4, 2, 3]
constr = [[1, 2], [3, 5], [1, 3], [2, 4]]
values_eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
for eps in values_eps:
    scalable_fmmd_ILP = np.zeros([2, num_runs])
    scalable_fmmd_modified_greedy = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, scalable_fmmd_modified_greedy[0][run], scalable_fmmd_modified_greedy[1][run] = alg.scalable_fmmd_modified_greedy(V=elements, EPS=eps, k=k, C=c, constr=constr,
                                                                                                                              dist=utils.manhattan_dist_sparse)
        writer.writerow(["celebA_small", "Both", c, k, eps, "scalable_fmmd_modified_greedy", run, scalable_fmmd_modified_greedy[0][run], scalable_fmmd_modified_greedy[1][run]])
        sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, EPS=eps, k=k, C=c, constr=constr, dist=utils.manhattan_dist_sparse)
        writer.writerow(["celebA_small", "Both", c, k, eps, "scalable_fmmd_ILP", run, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run]])
    output.flush()

# read the Adult dataset grouped by sex (c=2)
elements.clear()
elements = []
with open("./data/adult_small.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Elem(int(row[0]), int(row[1]), features)
        elements.append(elem)

num_runs = 10
k = 10
c = 2
group_k = [7, 3]
constr = [[5, 9], [2, 4]]
values_eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
for eps in values_eps:
    scalable_fmmd_ILP = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, EPS=eps, k=k, C=c, constr=constr, dist=utils.euclidean_dist)
        writer.writerow(["adult_small", "Sex", c, k, eps, "scalable_fmmd_ILP", run, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run]])
    output.flush()

# read the Adult dataset grouped by race (c=5)
elements.clear()
with open("./data/adult_small.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Elem(int(row[0]), int(row[2]), features)
        elements.append(elem)

c = 5
group_k = [6, 1, 1, 1, 1]
constr = [[4, 8], [1, 2], [1, 2], [1, 2], [1, 2]]
values_eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
for eps in values_eps:
    scalable_fmmd_ILP = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, EPS=eps, k=k, C=c, constr=constr, dist=utils.euclidean_dist)
        writer.writerow(["adult_small", "Race", c, k, eps, "scalable_fmmd_ILP", run, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run]])
    output.flush()

# read the Adult dataset grouped by sex+race (c=10)
elements.clear()
with open("./data/adult_small.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Elem(int(row[0]), int(row[3]), features)
        elements.append(elem)

c = 10
group_k = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
constr = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
values_eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
for eps in values_eps:
    scalable_fmmd_ILP = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, EPS=eps, k=k, C=c, constr=constr, dist=utils.euclidean_dist)
        writer.writerow(["adult_small", "Both", c, k, eps, "scalable_fmmd_ILP", run, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run]])
    output.flush()

'''
fairGmm
'''

# fairGmm
output = open("results_very_k_eps.csv", "a")
writer = csv.writer(output)
writer.writerow(["dataset", "group", "c", "k", "algorithm", "num", "div", "time"])
output.flush()

# read the Census dataset grouped by sex (c=2)
elements.clear()
with open("./data/census_small.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Elem(int(row[0]), int(row[1]), features)
        elements.append(elem)
# experiments on varying k
num_runs = 10
c = 2
values_k = range(5, 11, 5)
group_k = {5: [2, 3], 10: [5, 5]}  # , 15: [7, 8]
for k in values_k:
    fair_gmm = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, m=c, k=group_k[k], dist=utils.manhattan_dist)
        writer.writerow(["Census_small", "Sex", c, k, "FairGMM", run, fair_gmm[0][run], fair_gmm[1][run]])
    output.flush()

# read the twitter dataset grouped by gender (c=3)
elements.clear()
elements = []
with open("./data/twitter_small.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Elem(int(row[0]), int(row[1]), features)
        elements.append(elem)

num_runs = 10
values_k = [5]
c = 3
group_k = [2, 2, 1]
for k in values_k:
    fair_gmm = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, k=group_k, m=c, dist=utils.cosine_dist)
    writer.writerow(["twitter_small", "Sex", c, k, "FairGMM", np.average(fair_gmm[0]), np.average(fair_gmm[1])])
    output.flush()

# read the Adult dataset grouped by sex (c=2)
elements.clear()
elements = []
with open("./data/adult_small.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Elem(int(row[0]), int(row[1]), features)
        elements.append(elem)

num_runs = 10
values_k = [5]
c = 2
group_k = [3, 2]
for k in values_k:
    fair_gmm = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, m=c, k=group_k, dist=utils.euclidean_dist)
    writer.writerow(["adult_small", "Sex", c, k, "FairGMM", np.average(fair_gmm[0]), np.average(fair_gmm[1])])
    output.flush()

# read the Adult dataset grouped by race (c=5)
elements.clear()
with open("./data/adult_small.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Elem(int(row[0]), int(row[2]), features)
        elements.append(elem)

num_runs = 10
values_k = [5]
c = 5
group_k = [1, 1, 1, 1, 1]
for k in values_k:
    fair_gmm = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, k=group_k, m=c, dist=utils.euclidean_dist)
    writer.writerow(["adult_small", "Race", c, k, "FairGMM", np.average(fair_gmm[0]), np.average(fair_gmm[1])])
    output.flush()

# read the celebA dataset grouped by sex (c=2)
elements.clear()
elements = []
csr_sparse = sp.load_npz('./data/celebA_small_csr_sparse.npz')
for i in range(1000):
    if int(csr_sparse[i, 1]) == -1:
        elem = utils.ElemSparse(int(csr_sparse[i, 0]), 0, csr_sparse[i, 4:])
    else:
        elem = utils.ElemSparse(int(csr_sparse[i, 0]), int(csr_sparse[i, 1]), csr_sparse[i, 4:])
    elements.append(elem)

values_k = [5]
c = 2
group_k = [3, 2]
for k in values_k:
    fair_gmm = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, k=group_k, m=c,
                                                               dist=utils.manhattan_dist_sparse)
    writer.writerow(["celebA_small", "Sex", c, k, "FairGMM", np.average(fair_gmm[0]), np.average(fair_gmm[1])])
    output.flush()

# read the celebA dataset grouped by age (c=2)
elements.clear()
elements = []
csr_sparse = sp.load_npz('./data/celebA_small_csr_sparse.npz')
for i in range(1000):
    if int(csr_sparse[i, 2]) == -1:
        elem = utils.ElemSparse(int(csr_sparse[i, 0]), 0, csr_sparse[i, 4:])
    else:
        elem = utils.ElemSparse(int(csr_sparse[i, 0]), int(csr_sparse[i, 2]), csr_sparse[i, 4:])
    elements.append(elem)

values_k = [5]
c = 2
group_k = [1, 4]
for k in values_k:
    fair_gmm = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, m=c, k=group_k,
                                                               dist=utils.manhattan_dist_sparse)
    writer.writerow(["celebA_small", "Age", c, k, "FairGMM", np.average(fair_gmm[0]), np.average(fair_gmm[1])])
    output.flush()

# read the celebA dataset grouped by sex+age (c=4)
elements.clear()
elements = []
csr_sparse = sp.load_npz('./data/celebA_small_csr_sparse.npz')
for i in range(1000):
    if int(csr_sparse[i, 3]) == -1:
        elem = utils.ElemSparse(int(csr_sparse[i, 0]), 0, csr_sparse[i, 4:])
    else:
        elem = utils.ElemSparse(int(csr_sparse[i, 0]), int(csr_sparse[i, 3]), csr_sparse[i, 4:])
    elements.append(elem)

values_k = [5]
c = 4
group_k = [1, 2, 1, 1]
for k in values_k:
    fair_gmm = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, k=group_k, m=c,
                                                               dist=utils.manhattan_dist_sparse)
    writer.writerow(["celebA_small", "Both", c, k, "FairGMM", np.average(fair_gmm[0]), np.average(fair_gmm[1])])
    output.flush()
