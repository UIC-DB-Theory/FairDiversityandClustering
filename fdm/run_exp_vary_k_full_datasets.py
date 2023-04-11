import csv
import random
import numpy as np
import algorithms_offline as algo
import algorithms_streaming as algs
import alg
import utils
import scipy.sparse as sp
import sys, os
import multiprocessing as mp

# Disable
def blockPrint():
	sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
	sys.stdout = sys.__stdout__
	
def print_o(s):
	enablePrint()
	print(s)
	blockPrint()



def run_celebA_dataset(writer, output):

	print_o("\tgrouped by sex (c=2)")
	# read the celebA dataset grouped by sex (c=2)
	elements = []
	csr_sparse = sp.load_npz('./data/celebA_csr_sparse.npz')
	for i in range(202599):
		if int(csr_sparse[i, 1]) == -1:
			elem = utils.ElemSparse(int(csr_sparse[i, 0]), 0, csr_sparse[i, 4:])
		else:
			elem = utils.ElemSparse(int(csr_sparse[i, 0]), int(csr_sparse[i, 1]), csr_sparse[i, 4:])
		elements.append(elem)

	values_k = range(10, 101, 10)
	c = 2
	num_runs = 10
	EPS = 0.05
	range_d_sex = {10: [72154, 180387], 20: [68702, 171755], 30: [66912, 167281], 40: [65378, 163446], 50: [64417, 161043],
				60: [63581, 158954], 70: [62969, 157423], 80: [62067, 155168], 90: [61287, 153219], 100: [60918, 152295]}
	group_k = {
		10: [6, 4],
		20: [12, 8],
		30: [13, 13],
		40: [23, 17],
		50: [29, 21],
		60: [35, 25],
		70: [41, 29],
		80: [47, 33],
		90: [52, 38],
		100: [58, 42]}
	constr = {
		10: [[4, 8], [3, 5]], 20: [[9, 15], [6, 10]], 30: [[10, 16], [10, 16]],
		40: [[18, 28], [13, 21]], 50: [[23, 35], [16, 26]], 60: [[28, 42], [20, 30]], 70: [[32, 50], [23, 35]], 80: [[37, 57], [26, 40]], 90: [[41, 63], [30, 46]],
		100: [[46, 70], [33, 51]]}

	for k in values_k:
		alg1 = np.zeros([4, num_runs])
		alg2 = np.zeros([4, num_runs])
		fair_greedy_flow = np.zeros([2, num_runs])
		fair_swap = np.zeros([2, num_runs])
		fair_flow = np.zeros([2, num_runs])
		scalable_fmmd_ILP = np.zeros([2, num_runs])
		scalable_fmmd_modified_greedy = np.zeros([2, num_runs])
		for run in range(num_runs):
			random.Random(run).shuffle(elements)
			for new_idx in range(len(elements)):
				elements[new_idx].idx = new_idx
			sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=group_k[k],
																								dist=utils.manhattan_dist_sparse,
																								eps=EPS,
																								dmax=range_d_sex[k][1],
																								dmin=range_d_sex[k][0])
			print(sol)
			sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k[k],
																								m=c,
																								dist=utils.manhattan_dist_sparse,
																								eps=EPS,
																								dmax=range_d_sex[k][1],
																								dmin=range_d_sex[k][0])
			print(sol)
			sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k[k], m=c,
																						dist=utils.manhattan_dist_sparse,
																						eps=EPS,
																						dmax=range_d_sex[k][1],
																						dmin=range_d_sex[k][0],
																						metric_name='cityblock')
			print(sol)
			sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k[k], m=c,
																	dist=utils.manhattan_dist_sparse)
			sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements, k=group_k[k],
																	dist=utils.manhattan_dist_sparse)
			sol, scalable_fmmd_modified_greedy[0][run], scalable_fmmd_modified_greedy[1][
				run] = alg.scalable_fmmd_modified_greedy(V=elements, k=k, EPS=EPS, C=c, constr=constr[k],
														dist=utils.manhattan_dist_sparse)
			sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, k=k, EPS=EPS, C=c,
																							constr=constr[k],
																							dist=utils.manhattan_dist_sparse)
		writer.writerow(["celebA", "Sex", c, k, "FairSwap", EPS, np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
		writer.writerow(["celebA", "Sex", c, k, "FairFlow", EPS, np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
		writer.writerow(["celebA", "Sex", c, k, "scalable_fmmd_ILP", EPS, np.average(scalable_fmmd_ILP[0]), "-", "-", "-",
						np.average(scalable_fmmd_ILP[1])])
		writer.writerow(
			["celebA", "Sex", c, k, "scalable_fmmd_modified_greedy", EPS, np.average(scalable_fmmd_modified_greedy[0]), "-", "-", "-",
			np.average(scalable_fmmd_modified_greedy[1])])
		writer.writerow(["celebA", "Sex", c, k, "FairGreedyFlow", EPS, np.average(fair_greedy_flow[0]), "-", "-", "-",
						np.average(fair_greedy_flow[1])])
		writer.writerow(
			["celebA", "Sex", c, k, "Alg1", EPS, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]),
			np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
		writer.writerow(
			["celebA", "Sex", c, k, "Alg2", EPS, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
			np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
		output.flush()

	print_o("\tgrouped by age (c=2)")
	# read the celebA dataset grouped by age (c=2)
	elements.clear()
	elements = []
	csr_sparse = sp.load_npz('./data/celebA_csr_sparse.npz')
	for i in range(202599):
		if int(csr_sparse[i, 2]) == -1:
			elem = utils.ElemSparse(int(csr_sparse[i, 0]), 0, csr_sparse[i, 4:])
		else:
			elem = utils.ElemSparse(int(csr_sparse[i, 0]), int(csr_sparse[i, 2]), csr_sparse[i, 4:])
		elements.append(elem)

	values_k = range(10, 101, 10)
	c = 2
	range_d_age = {10: [72154, 180387], 20: [68702, 171755], 30: [66912, 167281], 40: [65378, 163446], 50: [64417, 161043],
				60: [63581, 158954], 70: [62969, 157423], 80: [62067, 155168], 90: [61287, 153219], 100: [60918, 152295]}
	group_k = {10: [2, 8],
			20: [5, 15],
			30: [7, 23],
			40: [9, 31],
			50: [11, 39],
			60: [14, 46],
			70: [16, 54],
			80: [18, 62],
			90: [20, 70],
			100: [23, 77]}
	constr = {10: [[1, 3], [6, 10]], 20: [[4, 6], [12, 18]], 30: [[5, 9], [18, 28]], 40: [[7, 11], [24, 38]], 50: [[8, 14], [31, 47]], 60: [[11, 17], [36, 56]],
			70: [[12, 20], [43, 65]], 80: [[14, 22], [49, 75]], 90: [[16, 24], [56, 84]], 100: [[18, 28], [61, 93]]}
	for k in values_k:
		alg1 = np.zeros([4, num_runs])
		alg2 = np.zeros([4, num_runs])
		fair_greedy_flow = np.zeros([2, num_runs])
		fair_swap = np.zeros([2, num_runs])
		fair_flow = np.zeros([2, num_runs])
		scalable_fmmd_ILP = np.zeros([2, num_runs])
		scalable_fmmd_modified_greedy = np.zeros([2, num_runs])
		for run in range(num_runs):
			random.Random(run).shuffle(elements)
			for new_idx in range(len(elements)):
				elements[new_idx].idx = new_idx
			sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=group_k[k],
																								dist=utils.manhattan_dist_sparse,
																								eps=EPS,
																								dmax=range_d_age[k][1],
																								dmin=range_d_age[k][0])
			print(sol)
			sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k[k],
																								m=c,
																								dist=utils.manhattan_dist_sparse,
																								eps=EPS,
																								dmax=range_d_age[k][1],
																								dmin=range_d_age[k][0])
			print(sol)
			sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k[k], m=c,
																						dist=utils.manhattan_dist_sparse,
																						eps=EPS,
																						dmax=range_d_age[k][1],
																						dmin=range_d_age[k][0],
																						metric_name='cityblock')
			print(sol)
			sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements, k=group_k[k],
																	dist=utils.manhattan_dist_sparse)
			sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k[k], m=c,
																	dist=utils.manhattan_dist_sparse)
			sol, scalable_fmmd_modified_greedy[0][run], scalable_fmmd_modified_greedy[1][
				run] = alg.scalable_fmmd_modified_greedy(V=elements, k=k, EPS=EPS, C=c, constr=constr[k],
														dist=utils.manhattan_dist_sparse)
			sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, k=k, EPS=EPS, C=c,
																							constr=constr[k],
																							dist=utils.manhattan_dist_sparse)

		writer.writerow(["celebA", "Age", c, k, "FairSwap", EPS, np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
		writer.writerow(["celebA", "Age", c, k, "FairFlow", EPS, np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
		writer.writerow(["celebA", "Age", c, k, "scalable_fmmd_ILP", EPS, np.average(scalable_fmmd_ILP[0]), "-", "-", "-",
						np.average(scalable_fmmd_ILP[1])])
		writer.writerow(
			["celebA", "Age", c, k, "scalable_fmmd_modified_greedy", EPS, np.average(scalable_fmmd_modified_greedy[0]), "-", "-", "-",
			np.average(scalable_fmmd_modified_greedy[1])])
		writer.writerow(["celebA", "Age", c, k, "FairGreedyFlow", EPS, np.average(fair_greedy_flow[0]), "-", "-", "-",
						np.average(fair_greedy_flow[1])])
		writer.writerow(
			["celebA", "Age", c, k, "Alg1", EPS, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]),
			np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
		writer.writerow(
			["celebA", "Age", c, k, "Alg2", EPS, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
			np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
		output.flush()

	print_o("\tgrouped by sex+age (c=4)")
	# read the celebA dataset grouped by sex+age (c=4)
	elements.clear()
	elements = []
	csr_sparse = sp.load_npz('./data/celebA_csr_sparse.npz')
	for i in range(202599):
		if int(csr_sparse[i, 3]) == -1:
			elem = utils.ElemSparse(int(csr_sparse[i, 0]), 0, csr_sparse[i, 4:])
		else:
			elem = utils.ElemSparse(int(csr_sparse[i, 0]), int(csr_sparse[i, 3]), csr_sparse[i, 4:])
		elements.append(elem)

	values_k = range(10, 101, 10)
	range_d_both = {10: [72154, 180387], 20: [68702, 171755], 30: [66912, 167281], 40: [65378, 163446], 50: [64417, 161043],
					60: [63581, 158954], 70: [62969, 157423], 80: [62067, 155168], 90: [61287, 153219], 100: [60918, 152295]}
	c = 4
	group_k = {10: [1, 4, 2, 3],
			20: [2, 10, 3, 5],
			30: [2, 15, 5, 8],
			40: [3, 20, 6, 11],
			50: [4, 25, 8, 13],
			60: [4, 31, 9, 16],
			70: [5, 36, 11, 18],
			80: [6, 41, 12, 21],
			90: [7, 45, 14, 24],
			100: [8, 51, 15, 26]}
	constr = {10: [[1, 2], [3, 5], [1, 3], [2, 4]], 20: [[1, 3], [8, 12], [2, 4], [4, 6]], 30: [[1, 3], [12, 18], [4, 6], [6, 10]], 40: [[2, 4], [16, 24], [4, 8], [8, 14]],
			50: [[3, 5], [20, 30], [6, 10], [10, 16]], 60: [[3, 5], [24, 38], [7, 11], [12, 20]], 70: [[4, 6], [28, 44], [8, 14], [14, 22]],
			80: [[4, 8], [32, 50], [9, 15], [16, 26]], 90: [[5, 9], [36, 54], [11, 17], [19, 29]], 100: [[6, 10], [40, 62], [12, 18], [20, 32]]}

	for k in values_k:
		alg2 = np.zeros([4, num_runs])
		fair_greedy_flow = np.zeros([2, num_runs])
		fair_flow = np.zeros([2, num_runs])
		scalable_fmmd_ILP = np.zeros([2, num_runs])
		scalable_fmmd_modified_greedy = np.zeros([2, num_runs])
		for run in range(num_runs):
			random.Random(run).shuffle(elements)
			for new_idx in range(len(elements)):
				elements[new_idx].idx = new_idx
			sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k[k],
																								m=c,
																								dist=utils.manhattan_dist_sparse,
																								eps=EPS,
																								dmax=range_d_both[k][1],
																								dmin=range_d_both[k][0])
			print(sol)
			sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k[k], m=c,
																						dist=utils.manhattan_dist_sparse,
																						eps=EPS,
																						dmax=range_d_both[k][1],
																						dmin=range_d_both[k][0],
																						metric_name='cityblock')
			print(sol)
			sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k[k], m=c,
																	dist=utils.manhattan_dist_sparse)
			sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, k=k, EPS=EPS, C=c,
																							constr=constr[k],
																							dist=utils.manhattan_dist_sparse)
			sol, scalable_fmmd_modified_greedy[0][run], scalable_fmmd_modified_greedy[1][
				run] = alg.scalable_fmmd_modified_greedy(V=elements, k=k, EPS=EPS, C=c, constr=constr[k],
														dist=utils.manhattan_dist_sparse)
		writer.writerow(["celebA", "Both", c, k, "FairFlow", EPS, np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
		writer.writerow(["celebA", "Both", c, k, "scalable_fmmd_ILP", EPS, np.average(scalable_fmmd_ILP[0]), "-", "-", "-",
						np.average(scalable_fmmd_ILP[1])])
		writer.writerow(
			["celebA", "Both", c, k, "scalable_fmmd_modified_greedy", EPS, np.average(scalable_fmmd_modified_greedy[0]), "-", "-", "-",
			np.average(scalable_fmmd_modified_greedy[1])])
		writer.writerow(["celebA", "both", c, k, "FairGreedyFlow", EPS, np.average(fair_greedy_flow[0]), "-", "-", "-",
						np.average(fair_greedy_flow[1])])
		writer.writerow(
			["celebA", "both", c, k, "Alg2", EPS, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
			np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
		output.flush()

def run_adult_dataset(writer, output):
	# adult

	print_o("\tgrouped by sex (c=2)")
	# read the Adult dataset grouped by sex (c=2)
	elements = []
	with open("./data/adult.csv", "r") as fileobj:
		csvreader = csv.reader(fileobj, delimiter=',')
		for row in csvreader:
			features = []
			for i in range(4, len(row)):
				features.append(float(row[i]))
			elem = utils.Elem(int(row[0]), int(row[1]), features)
			elements.append(elem)

	num_runs = 10
	EPS = 0.05
	values_k = range(10, 101, 10)
	c = 2
	range_d_sex = {10: [3.4, 8.5], 20: [2.5, 6.3], 30: [2.2, 5.5], 40: [1.94, 4.85], 50: [1.79, 4.46],
				60: [1.66, 4, 15], 70: [1.57, 3.93], 80: [1.50, 3.74], 90: [1.42, 3.56], 100: [1.37, 3.43]}
	group_k = {10: [7, 3],
			20: [13, 7],
			30: [20, 10],
			40: [27, 13],
			50: [33, 17],
			60: [40, 20],
			70: [47, 23],
			80: [53, 27],
			90: [60, 30],
			100: [67, 33]}
	constr = {10: [[5, 9], [2, 4]], 20: [[10, 16], [5, 9]], 30: [[16, 24], [8, 12]], 40: [[21, 33], [10, 16]], 50: [[26, 40], [13, 21]], 60: [[32, 48], [16, 24]],
			70: [[37, 57], [18, 28]], 80: [[42, 64], [21, 33]], 90: [[48, 72], [24, 36]], 100: [[53, 81], [26, 40]]}
	for k in values_k:
		alg1 = np.zeros([4, num_runs])
		alg2 = np.zeros([4, num_runs])
		fair_greedy_flow = np.zeros([2, num_runs])
		fair_swap = np.zeros([2, num_runs])
		fair_flow = np.zeros([2, num_runs])
		scalable_fmmd_ILP = np.zeros([2, num_runs])
		for run in range(num_runs):
			random.Random(run).shuffle(elements)
			for new_idx in range(len(elements)):
				elements[new_idx].idx = new_idx
			sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=group_k[k],
																								dist=utils.euclidean_dist,
																								eps=EPS,
																								dmax=range_d_sex[k][1],
																								dmin=range_d_sex[k][0])
			print(sol)
			sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k[k], m=c,
																								dist=utils.euclidean_dist,
																								eps=EPS,
																								dmax=range_d_sex[k][1],
																								dmin=range_d_sex[k][0])
			print(sol)
			sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k[k], m=c,
																						dist=utils.euclidean_dist,
																						eps=EPS,
																						dmax=range_d_sex[k][1],
																						dmin=range_d_sex[k][0],
																						metric_name='euclidean')
			print(sol)
			sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements, k=group_k[k], dist=utils.euclidean_dist)
			sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k[k], m=c,
																	dist=utils.euclidean_dist)
			sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, k=k, EPS=EPS, C=c,
																							constr=constr[k],
																							dist=utils.euclidean_dist)
		writer.writerow(["adult", "Sex", c, k, "FairSwap", EPS, np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
		writer.writerow(["adult", "Sex", c, k, "FairFlow", EPS, np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
		writer.writerow(
			["adult", "Sex", c, k, "scalable_fmmd_ILP", EPS, np.average(scalable_fmmd_ILP[0]), "-", "-", "-", np.average(scalable_fmmd_ILP[1])])
		writer.writerow(["Adult", "Sex", c, k, "Alg1", EPS, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]),
						np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
		writer.writerow(["Adult", "Sex", c, k, "Alg2", EPS, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
						np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
		writer.writerow(["Adult", "Sex", c, k, "FairGreedyFlow", EPS, np.average(fair_greedy_flow[0]), "-", "-", "-", np.average(fair_greedy_flow[1])])
		output.flush()

	print_o("\tgrouped by race (c=5)")
	# read the Adult dataset grouped by race (c=5)
	elements.clear()
	with open("./data/adult.csv", "r") as fileobj:
		csvreader = csv.reader(fileobj, delimiter=',')
		for row in csvreader:
			features = []
			for i in range(4, len(row)):
				features.append(float(row[i]))
			elem = utils.Elem(int(row[0]), int(row[2]), features)
			elements.append(elem)

	num_runs = 10
	values_k = range(10, 101, 10)
	c = 5
	range_d_race = {10: [3.4, 8.5], 20: [2.5, 6.3], 30: [2.2, 5.5], 40: [1.94, 4.85], 50: [1.79, 4.46],
					60: [1.66, 4, 15], 70: [1.57, 3.93], 80: [1.50, 3.74], 90: [1.42, 3.56], 100: [1.37, 3.43]}
	group_k = {10: [6, 1, 1, 1, 1],
			20: [15, 1, 1, 2, 1],
			30: [24, 1, 1, 3, 1],
			40: [33, 1, 1, 4, 1],
			50: [41, 2, 1, 5, 1],
			60: [50, 2, 1, 6, 1],
			70: [59, 2, 1, 7, 1],
			80: [68, 2, 1, 8, 1],
			90: [76, 3, 1, 9, 1],
			100: [85, 3, 1, 10, 1]}
	constr = {10: [[4, 8], [1, 2], [1, 2], [1, 2], [1, 2]], 20: [[12, 18], [1, 2], [1, 2], [1, 3], [1, 2]], 30: [[19, 29], [1, 2], [1, 2], [2, 4], [1, 2]],
			40: [[26, 40], [1, 2], [1, 2], [3, 5], [1, 2]], 50: [[32, 50], [1, 3], [1, 2], [4, 6], [1, 2]], 60: [[40, 60], [1, 3], [1, 2], [4, 8], [1, 2]],
			70: [[47, 71], [1, 3], [1, 2], [5, 9], [1, 2]], 80: [[54, 82], [1, 3], [1, 2], [6, 10], [1, 2]], 90: [[60, 92], [2, 4], [1, 2], [7, 11], [1, 2]],
			100: [[68, 102], [2, 4], [1, 2], [8, 12], [1, 2]]}

	for k in values_k:
		alg2 = np.zeros([4, num_runs])
		fair_greedy_flow = np.zeros([2, num_runs])
		fair_flow = np.zeros([2, num_runs])
		scalable_fmmd_ILP = np.zeros([2, num_runs])
		for run in range(num_runs):
			random.Random(run).shuffle(elements)
			for new_idx in range(len(elements)):
				elements[new_idx].idx = new_idx
			sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k[k], m=c,
																								dist=utils.euclidean_dist,
																								eps=EPS,
																								dmax=range_d_race[k][1],
																								dmin=range_d_race[k][0])
			print(sol)
			sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k[k], m=c,
																						dist=utils.euclidean_dist,
																						eps=EPS,
																						dmax=range_d_race[k][1],
																						dmin=range_d_race[k][0],
																						metric_name='euclidean')
			print(sol)
			sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k[k], m=c,
																	dist=utils.euclidean_dist)
			sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, k=k, EPS=EPS, C=c,
																							constr=constr[k],
																							dist=utils.euclidean_dist)

		writer.writerow(["adult", "Race", c, k, "FairFlow", EPS, np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
		writer.writerow(["adult", "Race", c, k, "scalable_fmmd_ILP", EPS, np.average(scalable_fmmd_ILP[0]), "-", "-", "-",
						np.average(scalable_fmmd_ILP[1])])
		writer.writerow(["Adult", "Race", c, k, "FairGreedyFlow", EPS, np.average(fair_greedy_flow[0]), "-", "-", "-",
						np.average(fair_greedy_flow[1])])
		writer.writerow(
			["Adult", "Race", c, k, "Alg2", EPS, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
			np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
		output.flush()

	print_o("\tgrouped by sex+race (c=10)")
	# read the Adult dataset grouped by sex+race (c=10)
	elements.clear()
	with open("./data/adult.csv", "r") as fileobj:
		csvreader = csv.reader(fileobj, delimiter=',')
		for row in csvreader:
			features = []
			for i in range(4, len(row)):
				features.append(float(row[i]))
			elem = utils.Elem(int(row[0]), int(row[3]), features)
			elements.append(elem)

	values_k = range(10, 101, 10)
	c = 10
	range_d_both = {10: [3.4, 8.5], 20: [2.5, 6.3], 30: [2.2, 5.5], 40: [1.94, 4.85], 50: [1.79, 4.46],
					60: [1.66, 4, 15], 70: [1.57, 3.93], 80: [1.50, 3.74], 90: [1.42, 3.56], 100: [1.37, 3.43]}
	group_k = {10: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
			20: [7, 1, 1, 1, 1, 5, 1, 1, 1, 1],
			30: [14, 1, 1, 1, 1, 8, 1, 1, 1, 1],
			40: [19, 1, 1, 2, 1, 11, 1, 1, 2, 1],
			50: [27, 1, 1, 2, 1, 13, 1, 1, 2, 1],
			60: [32, 1, 1, 3, 1, 16, 1, 1, 3, 1],
			70: [39, 1, 1, 3, 1, 19, 1, 1, 3, 1],
			80: [44, 2, 1, 4, 1, 21, 1, 1, 4, 1],
			90: [52, 2, 1, 4, 1, 24, 1, 0, 4, 1],
			100: [56, 2, 1, 5, 1, 27, 1, 1, 5, 1]}
	constr = {10: [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
			20: [[5, 9], [1, 2], [1, 2], [1, 2], [1, 2], [4, 6], [1, 2], [1, 2], [1, 2], [1, 2]],
			30: [[11, 17], [1, 2], [1, 2], [1, 2], [1, 2], [6, 10], [1, 2], [1, 2], [1, 2], [1, 2]],
			40: [[15, 23], [1, 2], [1, 2], [1, 3], [1, 2], [8, 14], [1, 2], [1, 2], [1, 3], [1, 2]],
			50: [[21, 33], [1, 2], [1, 2], [1, 3], [1, 2], [10, 16], [1, 2], [1, 2], [1, 3], [1, 2]],
			60: [[25, 39], [1, 2], [1, 2], [2, 4], [1, 2], [12, 20], [1, 2], [1, 2], [2, 4], [1, 2]],
			70: [[31, 47], [1, 2], [1, 2], [2, 4], [1, 2], [15, 23], [1, 2], [1, 2], [2, 4], [1, 2]],
			80: [[35, 53], [1, 3], [1, 2], [3, 5], [1, 2], [16, 26], [1, 2], [1, 2], [3, 5], [1, 2]],
			90: [[41, 63], [1, 3], [1, 2], [3, 5], [1, 2], [19, 29], [1, 2], [0, 0], [3, 5], [1, 2]],
			100: [[44, 68], [1, 3], [1, 2], [4, 6], [1, 2], [21, 33], [1, 2], [1, 2], [4, 6], [1, 2]]}

	for k in values_k:
		alg2 = np.zeros([4, num_runs])
		fair_greedy_flow = np.zeros([2, num_runs])
		fair_flow = np.zeros([2, num_runs])
		scalable_fmmd_ILP = np.zeros([2, num_runs])
		for run in range(num_runs):
			random.Random(run).shuffle(elements)
			for new_idx in range(len(elements)):
				elements[new_idx].idx = new_idx
			sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements,
																								k=group_k[k], m=c,
																								dist=utils.euclidean_dist,
																								eps=EPS,
																								dmax=range_d_both[k][
																									1],
																								dmin=range_d_both[k][
																									0])
			print(sol)
			sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k[k], m=c,
																						dist=utils.euclidean_dist,
																						eps=EPS,
																						dmax=range_d_race[k][1],
																						dmin=range_d_race[k][0],
																						metric_name='euclidean')
			print(sol)
			sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k[k], m=c,
																	dist=utils.euclidean_dist)
			sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, k=k, EPS=EPS, C=c,
																							constr=constr[k],
																							dist=utils.euclidean_dist)
		writer.writerow(["adult", "Both", c, k, "FairFlow", EPS, np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
		writer.writerow(["adult", "Both", c, k, "scalable_fmmd_ILP", EPS, np.average(scalable_fmmd_ILP[0]), "-", "-", "-",
						np.average(scalable_fmmd_ILP[1])])
		writer.writerow(["Adult", "both", c, k, "FairGreedyFlow", EPS, np.average(fair_greedy_flow[0]), "-", "-", "-",
						np.average(fair_greedy_flow[1])])
		writer.writerow(
			["Adult", "both", c, k, "Alg2", EPS, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
			np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
		output.flush()

def run_twitter_dataset(writer, output):

	print_o("\tgrouped by gender (c=3)")
	# read the twitter dataset grouped by gender (c=3)
	elements = []
	with open("./data/twitter.csv", "r") as fileobj:
		csvreader = csv.reader(fileobj, delimiter=',')
		for row in csvreader:
			features = []
			for i in range(2, len(row)):
				features.append(float(row[i]))
			elem = utils.Elem(int(row[0]), int(row[1]), features)
			elements.append(elem)

	num_runs = 10
	values_k = range(10, 101, 10)
	c = 3
	EPS = 0.05
	range_d_sex = {10: [0.74, 1.86], 20: [0.71, 1.8], 30: [0.70, 1.76], 40: [0.69, 1.74], 50: [0.69, 1.74],
				60: [0.69, 1.74], 70: [0.68, 1.70], 80: [0.69, 1.74], 90: [0.67, 1.68], 100: [0.67, 1.68]}
	group_k = {10: [3, 4, 3],
			20: [7, 7, 6],
			30: [10, 11, 9],
			40: [13, 14, 13],
			50: [16, 18, 16],
			60: [20, 21, 19],
			70: [23, 25, 22],
			80: [26, 28, 26],
			90: [30, 32, 28],
			100: [33, 36, 31]}
	constr = {10: [[2, 4], [3, 5], [2, 4]], 20: [[5, 9], [5, 9], [4, 8]], 30: [[8, 12], [8, 14], [7, 11]], 40: [[10, 16], [11, 17], [10, 16]], 50: [[12, 20], [14, 22], [12, 20]],
			60: [[16, 24], [16, 26], [15, 23]], 70: [[18, 28], [20, 30], [17, 27]], 80: [[20, 32], [22, 34], [20, 32]], 90: [[24, 36], [25, 39], [22, 34]],
			100: [[26, 40], [28, 44], [24, 38]]}
	for k in values_k:
		alg2 = np.zeros([4, num_runs])
		fair_greedy_flow = np.zeros([2, num_runs])
		fair_flow = np.zeros([2, num_runs])
		scalable_fmmd_ILP = np.zeros([2, num_runs])
		for run in range(num_runs):
			random.Random(run).shuffle(elements)
			for new_idx in range(len(elements)):
				elements[new_idx].idx = new_idx
			sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements,
																								k=group_k[k], m=c,
																								dist=utils.cosine_dist,
																								eps=EPS,
																								dmax=range_d_sex[k][
																									1],
																								dmin=range_d_sex[k][
																									0])
			print(sol)
			sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k[k], m=c,
																						dist=utils.cosine_dist,
																						eps=EPS,
																						dmax=range_d_sex[k][1],
																						dmin=range_d_sex[k][0],
																						metric_name='cosine')
			print(sol)
			sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k[k], m=c, dist=utils.cosine_dist)
			sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, k=k, EPS=EPS, C=c,
																							constr=constr[k],
																							dist=utils.cosine_dist)
		writer.writerow(["twitter", "Sex", c, k, "FairFlow", EPS, np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
		writer.writerow(["twitter", "Sex", c, k, "scalable_fmmd_ILP", EPS, np.average(scalable_fmmd_ILP[0]), "-", "-", "-",
						np.average(scalable_fmmd_ILP[1])])
		writer.writerow(["twitter", "sex", c, k, "FairGreedyFlow", EPS, np.average(fair_greedy_flow[0]), "-", "-", "-",
						np.average(fair_greedy_flow[1])])
		writer.writerow(
			["Twitter", "sex", c, k, "Alg2", EPS, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
			np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
		output.flush()


def run_census_dataset(writer, output):
	# all

	print_o("\tgrouped by sex (c=2)")
	# read the Census dataset grouped by sex (c=2)
	elements = []
	with open("./data/census.csv", "r") as fileobj:
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
	EPS = 0.05
	values_k = range(10, 101, 10)
	range_d_sex = {10: [21, 52.5], 20: [17.5, 43.75], 30: [16.5, 41.25], 40: [15.5, 38.75], 50: [14.5, 36.25],
				60: [14, 35], 70: [13.5, 33.75], 80: [13.5, 33.75], 90: [13, 32.5], 100: [13, 32.5]}
	group_k = {10: [5, 5], 20: [10, 10], 30: [15, 15], 40: [19, 21],
			50: [24, 26], 60: [29, 31], 70: [34, 36],
			80: [39, 41], 90: [44, 46], 100: [48, 52]}
	constr = {10: [[4, 6], [4, 6]], 20: [[8, 12], [8, 12]], 30: [[12, 18], [12, 18]],
			40: [[15, 23], [16, 26]], 50: [[19, 29], [20, 32]], 60: [[23, 35], [24, 38]],
			70: [[27, 41], [28, 44]], 80: [[31, 47], [32, 50]], 90: [[35, 53], [36, 56]],
			100: [[38, 58], [41, 63]]}
	for k in values_k:
		alg1 = np.zeros([4, num_runs])
		alg2 = np.zeros([4, num_runs])
		fair_swap = np.zeros([2, num_runs])
		fair_flow = np.zeros([2, num_runs])
		scalable_fmmd_ILP = np.zeros([2, num_runs])
		# fair_greedy_flow = np.zeros([2, num_runs])
		for run in range(num_runs):
			random.Random(run).shuffle(elements)
			for new_idx in range(len(elements)):
				elements[new_idx].idx = new_idx
			sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=group_k[k], dist=utils.manhattan_dist, eps=EPS, dmax=range_d_sex[k][1],
																								dmin=range_d_sex[k][0])
			print(sol)
			sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k[k], m=c, dist=utils.manhattan_dist, eps=EPS,
																								dmax=range_d_sex[k][1], dmin=range_d_sex[k][0])
			print(sol)
			# sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k[k], m=c,
			#                                                                           dist=utils.manhattan_dist,
			#                                                                           eps=EPS,
			#                                                                               dmax=range_d_sex[k][1],
			#                                                                               dmin=range_d_sex[k][0],
			#                                                                          metric_name='cityblock')
			sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements, k=group_k[k], dist=utils.manhattan_dist)
			sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k[k], m=c,
																	dist=utils.manhattan_dist)
			sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, k=k, EPS=EPS, C=c,
																							constr=constr[k],
																							dist=utils.manhattan_dist)
		writer.writerow(
			["Census", "Sex", c, k, "FairSwap", EPS, np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
		writer.writerow(["Census", "Sex", c, k, "FairFlow", EPS, np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
		writer.writerow(
			["Census", "Sex", c, k, "scalable_fmmd_ILP", EPS, np.average(scalable_fmmd_ILP[0]), "-", "-", "-", np.average(scalable_fmmd_ILP[1])])
		writer.writerow(
			["Census", "Sex", c, k, "Alg1", EPS, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]),
			np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
		writer.writerow(
			["Census", "Sex", c, k, "Alg2", EPS, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
			np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
		# writer.writerow(["Census", "Sex", c, k, "FairGreedyFlow", EPS, np.average(fair_greedy_flow[0]), "-", "-", "-",
		#                  np.average(fair_greedy_flow[1])])
		output.flush()

	print_o("\tgrouped by age (c=7)")
	# read the Census dataset grouped by age (c=7)
	elements.clear()
	elements = []
	with open("./data/census.csv", "r") as fileobj:
		csvreader = csv.reader(fileobj, delimiter=',')
		for row in csvreader:
			features = []
			for i in range(4, len(row)):
				features.append(float(row[i]))
			elem = utils.Elem(int(row[0]), int(row[2]), features)
			elements.append(elem)
	# experiments on varying k
	num_runs = 10
	EPS = 0.05
	c = 7
	values_k = range(10, 101, 10)
	range_d_age = {10: [14, 52.5], 20: [11.7, 43.75], 30: [11, 41.25], 40: [10.3, 38.75], 50: [9.67, 36.25],
				60: [9.3, 35], 70: [9, 33.75], 80: [9, 33.75], 90: [8.67, 32.5], 100: [8.67, 32.5]}
	group_k = {10: [2, 1, 2, 2, 1, 1, 1], 20: [4, 2, 2, 3, 3, 3, 3], 30: [5, 3, 5, 5, 4, 4, 4], 40: [7, 5, 6, 7, 5, 5, 5],
			50: [9, 5, 8, 8, 6, 7, 7], 60: [11, 6, 9, 10, 8, 8, 8], 70: [12, 7, 11, 12, 9, 10, 9],
			80: [15, 8, 12, 13, 10, 11, 11], 90: [16, 9, 14, 15, 12, 12, 12], 100: [18, 10, 15, 17, 13, 14, 13]}
	constr = {10: [[1, 3], [1, 2], [1, 3], [1, 3], [1, 2], [1, 2], [1, 2]], 20: [[3, 5], [1, 3], [1, 3], [2, 4], [2, 4], [2, 4], [2, 4]],
			30: [[4, 6], [2, 4], [4, 6], [4, 6], [3, 5], [3, 5], [3, 5]], 40: [[5, 9], [4, 6], [4, 8], [5, 9], [4, 6], [4, 6], [4, 6]],
			50: [[7, 11], [4, 6], [6, 10], [6, 10], [4, 8], [5, 9], [5, 9]], 60: [[8, 14], [4, 8], [7, 11], [8, 12], [6, 10], [6, 10], [6, 10]],
			70: [[9, 15], [5, 9], [8, 14], [9, 15], [7, 11], [8, 12], [7, 11]], 80: [[12, 18], [6, 10], [9, 15], [10, 16], [8, 12], [8, 14], [8, 14]],
			90: [[12, 20], [7, 11], [11, 17], [12, 18], [9, 15], [9, 15], [9, 15]], 100: [[14, 22], [8, 12], [12, 18], [13, 21], [10, 16], [11, 17], [10, 16]]}
	for k in values_k:
		alg2 = np.zeros([4, num_runs])
		fair_greedy_flow = np.zeros([2, num_runs])
		fair_flow = np.zeros([2, num_runs])
		scalable_fmmd_ILP = np.zeros([2, num_runs])
		for run in range(num_runs):
			random.Random(run).shuffle(elements)
			for new_idx in range(len(elements)):
				elements[new_idx].idx = new_idx
			sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k[k],
																								m=c,
																								dist=utils.manhattan_dist,
																								eps=EPS,
																								dmax=range_d_age[k][1],
																								dmin=range_d_age[k][0])
			print(sol)
			# sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k[k], m=c,
			#                                                                               dist=utils.manhattan_dist,
			#                                                                               eps=EPS,
			#                                                                               dmax=range_d_age[k][1],
			#                                                                               dmin=range_d_age[k][0],
			#                                                                               metric_name='cityblock')
			# print(sol)
			sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k[k], m=c,
																	dist=utils.manhattan_dist)
			sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, k=k, EPS=EPS, C=c,
																							constr=constr[k],
																							dist=utils.manhattan_dist)
		writer.writerow(["census", "Age", c, k, "scalable_fmmd_ILP", EPS, np.average(scalable_fmmd_ILP[0]), "-", "-", "-",
						np.average(scalable_fmmd_ILP[1])])
		writer.writerow(["census", "Age", c, k, "FairFlow", EPS, np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
		# writer.writerow(["Census", "Age", c, k, "FairGreedyFlow", EPS, np.average(fair_greedy_flow[0]), "-", "-", "-",
		#                  np.average(fair_greedy_flow[1])])
		writer.writerow(
			["Census", "Age", c, k, "Alg2", EPS, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
			np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
		output.flush()

	print_o("\tgrouped by both (c=14)")
	# read the Census dataset grouped by both (c=14)
	elements.clear()
	with open("./data/census.csv", "r") as fileobj:
		csvreader = csv.reader(fileobj, delimiter=',')
		for row in csvreader:
			features = []
			for i in range(4, len(row)):
				features.append(float(row[i]))
			elem = utils.Elem(int(row[0]), int(row[3]), features)
			elements.append(elem)

	c = 14
	num_runs = 10
	values_k = range(20, 101, 10)
	group_k = {
		20: [2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1],
		30: [3, 2, 2, 2, 2, 2, 2, 3, 1, 2, 3, 2, 2, 2],
		40: [4, 2, 3, 3, 3, 3, 2, 4, 2, 3, 2, 3, 3, 3],
		50: [5, 3, 4, 4, 3, 3, 3, 4, 2, 4, 4, 3, 4, 4],
		60: [6, 3, 5, 4, 4, 4, 3, 5, 3, 5, 5, 4, 4, 5],
		70: [6, 4, 5, 6, 4, 5, 4, 6, 3, 5, 6, 5, 5, 6],
		80: [7, 4, 6, 7, 5, 5, 4, 7, 5, 6, 7, 5, 6, 6],
		90: [8, 5, 7, 7, 6, 6, 5, 8, 4, 7, 8, 6, 6, 7],
		100: [9, 5, 8, 8, 6, 6, 5, 9, 6, 8, 8, 7, 7, 8],
	}
	constr = {20: [[1, 3], [1, 2], [1, 3], [1, 3], [1, 2], [1, 2], [1, 2], [1, 3], [1, 2], [1, 3], [1, 3], [1, 2], [1, 2], [1, 2]],
			30: [[2, 4], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [2, 4], [1, 2], [1, 3], [2, 4], [1, 3], [1, 3], [1, 3]],
			40: [[3, 5], [1, 3], [2, 4], [2, 4], [2, 4], [2, 4], [1, 3], [3, 5], [1, 3], [2, 4], [1, 3], [2, 4], [2, 4], [2, 4]],
			50: [[4, 6], [2, 4], [3, 5], [3, 5], [2, 4], [2, 4], [2, 4], [3, 5], [1, 3], [3, 5], [3, 5], [2, 4], [3, 5], [3, 5]],
			60: [[4, 8], [2, 4], [4, 6], [3, 5], [3, 5], [3, 5], [2, 4], [4, 6], [2, 4], [4, 6], [4, 6], [3, 5], [3, 5], [4, 6]],
			70: [[4, 8], [3, 5], [4, 6], [4, 8], [3, 5], [4, 6], [3, 5], [4, 8], [2, 4], [4, 6], [4, 8], [4, 6], [4, 6], [4, 8]],
			80: [[5, 9], [3, 5], [4, 8], [5, 9], [4, 6], [4, 6], [3, 5], [5, 9], [4, 6], [4, 8], [5, 9], [4, 6], [4, 8], [4, 8]],
			90: [[6, 10], [4, 6], [5, 9], [5, 9], [4, 8], [4, 8], [4, 6], [6, 10], [3, 5], [5, 9], [6, 10], [4, 8], [4, 8], [5, 9]],
			100: [[7, 11], [4, 6], [6, 10], [6, 10], [4, 8], [4, 8], [4, 6], [7, 11], [4, 8], [6, 10], [6, 10], [5, 9], [5, 9], [6, 10]]}
	range_d_both = {10: [16.8, 31.5], 20: [14, 26.25], 30: [13.2, 24.75], 40: [12.4, 23.25], 50: [11.6, 21.75],
					60: [11.2, 21], 70: [10.8, 20.25], 80: [10.8, 20.25], 90: [10.4, 19.5], 100: [10.4, 19.5]}
	for k in values_k:
		alg2 = np.zeros([4, num_runs])
		fair_greedy_flow = np.zeros([2, num_runs])
		fair_flow = np.zeros([2, num_runs])
		scalable_fmmd_ILP = np.zeros([2, num_runs])
		for run in range(num_runs):
			random.Random(run).shuffle(elements)
			for new_idx in range(len(elements)):
				elements[new_idx].idx = new_idx
			sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k[k],
																								m=c,
																								dist=utils.manhattan_dist,
																								eps=EPS,
																								dmax=range_d_both[k][1],
																								dmin=range_d_both[k][0])
			print(sol)
			# sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k[k], m=c,
			#                                                                               dist=utils.manhattan_dist,
			#                                                                               eps=EPS,
			#                                                                               dmax=range_d_both[k][1],
			#                                                                               dmin=range_d_both[k][0],
			#                                                                               metric_name='cityblock')
			# print(sol)
			sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k[k], m=c,
																	dist=utils.manhattan_dist)
			sol, scalable_fmmd_ILP[0][run], scalable_fmmd_ILP[1][run] = alg.scalable_fmmd_ILP(V=elements, k=k, EPS=EPS, C=c,
																							constr=constr[k],
																							dist=utils.manhattan_dist)
		writer.writerow(["census", "both", c, k, "FairFlow", EPS, np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
		writer.writerow(["census", "both", c, k, "scalable_fmmd_ILP", EPS, np.average(scalable_fmmd_ILP[0]), "-", "-", "-",
						np.average(scalable_fmmd_ILP[1])])
		# writer.writerow(["Census", "both", c, k, "FairGreedyFlow", EPS, np.average(fair_greedy_flow[0]), "-", "-", "-",
		#                  np.average(fair_greedy_flow[1])])
		writer.writerow(
			["Census", "both", c, k, "Alg2", EPS, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
			np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
		output.flush()

def resultlistener(q, fn):
	with open(fn, 'w') as f:
		writer = csv.writer(f)
		while 1:
			m = q.get()
			if m == 'kill':
				break
			writer.writerow(m)
			f.flush()

def main():
	'''
	experiment-vary-k-small:
	n = 1000
	k = 5,10,15, .,45,50
	eps = 0.05
	'''
	print("***Varying K for full datasets***")
	
	outfile = "results_vary_k.csv"
	output = open("results_vary_k.csv", "a")
	writer = csv.writer(output)
	writer.writerow(["dataset", "group", "m", "k", "algorithm", "param_eps", "div", "num_elem", "time1", "time2", "time3"])
	output.flush()
	# output.close()

	# manager = mp.Manager()
	# q = manager.Queue()
	# pool = mp.Pool(mp.cpu_count() + 2)
	
	# watcher = pool.apply_async(resultlistener, (q, outfile))

	blockPrint()

	#print_o("Running adult dataset")
	#run_adult_dataset(writer, output)

	# missing datafile: celebA_csr_sparse.npz
	#print_o("Running celebA dataset")
	#run_celebA_dataset(writer, output)


	print_o("Running census dataset")
	run_census_dataset(writer, output)


	print_o("Running twitter dataset")
	run_twitter_dataset(writer, output)

	enablePrint()
	



if __name__ == "__main__":
	main()
