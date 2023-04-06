import matplotlib.pyplot as plt
import os
import sys
import csv


def color(alg_name):
	if alg_name == "fmmd_ILP":
		# Red
		return 'r-'
	elif alg_name == "FairFlow":
		# Yellow
		return 'y-'
	elif alg_name == "scalable_fmmd_ILP":
		# Green
		return 'g-'
	elif alg_name == "FairSwap":
		# Black
		return 'k-'
	elif alg_name == "Alg1":
		# Pink
		return 'm-'
	elif alg_name == "Alg2":
		# Blue
		return 'b-'
	elif alg_name == "FairGreedyFlow":
		# Cyan
		return 'c-'


# main
def main():
	# Load the data into a dictionary 

	data_per_alg = {}
	filepath = "./results/results_vary_k copy.csv"
	with open(filepath) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		lc = 0
		for row in csv_reader:
			if lc == 0:
				lc += 1
			else:
				if row[4] not in data_per_alg:
					data_per_alg[row[4]] = {"k":[int(row[3])], "time":[float(row[10])]}
				else:
					data_per_alg[row[4]]["k"].append(int(row[3]))
					data_per_alg[row[4]]["time"].append(float(row[10]))

	for a in data_per_alg:
		print("Alg = ", a)
		print("\tX=", data_per_alg[a]["k"])
		print("\tY=", data_per_alg[a]["time"])
		plt.plot(data_per_alg[a]["k"], data_per_alg[a]["time"], color(a), label=a)
	
	plt.yscale("log")
	plt.legend(title = "Census Small (Sex)")
	plt.xlabel("k")
	plt.ylabel("time")
	#fig, ax = plt.subplots()
	#ax.set_ylim(bottom=0)
	plt.savefig('censmallsex.png', dpi=300, bbox_inches='tight')








if __name__ == "__main__":
	main()