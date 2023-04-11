import matplotlib.pyplot as plt
import csv
import os


def labels(filetype):
	if filetype == "vary_k":
		return "k", "time"
	elif filetype == "vary_k_div":
		return "k", "diversity"


def color(alg_name):
	'''
	Returns the color used for plotting the graph of the given algorithm
	
	alg_name - name of the algorithm
	'''
	if alg_name == "fmmd_ilp": # FMMD-E
		# Blue
		return 'tab:blue'
	elif alg_name == "fairflow": 
		# Yellow
		return 'y-'
	elif alg_name == "scalable_fmmd_ilp": #FMMD-S
		# Red
		return 'tab:red'
	elif alg_name == "fairswap":
		# Black
		return 'k'
	elif alg_name == "sfdm1": #agl1
		# Purple
		return 'tab:purple'
	elif alg_name == "sfdm2": #alg2
		# Green
		return 'tab:green'
	elif alg_name == "fairgreedyflow":
		# Cyan
		return 'tab:cyan'
	elif alg_name == "scalable_fmmd_modified_greedy":
		# Brown
		return 'tab:brown'


def plot_graph(graphname,filetype, data_per_alg, outputdir):

	'''
	plots a graph for a single dataset and group variation.

	graphname - the name of the graph/also the name of the file the graph is saved to.
	x_label - label for x axis
	y_label - label for y axis
	data_per_alg - teh data of k vs runtime for each algorithm for the particular dataset and group.
	outputdir - the directory where the geaph is saved
	'''

	plt.clf()
	for a in data_per_alg:
		plt.plot(data_per_alg[a]["x"], data_per_alg[a]["y"], color(a), label=a)
	
	plt.yscale("log")
	plt.legend(title = graphname, bbox_to_anchor=(1.05, 1.0), loc='upper left')
	x_label, y_label = labels(filetype)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.savefig(outputdir + "/" + graphname, dpi=300, bbox_inches='tight')

def load_data(filepath, filetype):
	'''
	loads the data for plotting the graph into a dictionary and returns it.

	The dictionary would look something like:
	{
		...,

		"census_small_sex" : {
								"fairflow" : {"x":[//k value], "y":[//runtimes]},
								"fmmd_ilp" : {"x":[//k value], "y":[//runtimes]}
							},

		"adult_small_race" : {
								"fairflow" : {"x":[//k value], "y":[//runtimes]},
								"fmmd_ilp" : {"x":[//k value], "y":[//runtimes]}
							},
		...
	}
	'''
	data = {}
	with open(filepath) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		lc = 0
		for row in csv_reader:
			print("reading line: ", lc)
			# Ignore first row
			if lc != 0:

				try:
					if filetype == "vary_k":
						dataset = row[0].lower()
						group = row[1].lower()
						alg = row[4].lower()
						if alg == "alg1":
							alg = "sfdm1"
						elif alg == "alg2":
							alg = "sfdm2"
						x_val = int(row[3])
						y_val = float(row[10])
					elif filetype == "vary_k_div":
						dataset = row[0].lower()
						group = row[1].lower()
						alg = row[4].lower()
						x_val = int(row[3])
						y_val = float(row[10])
				except:
					print("csv has missing columns")
				key = dataset + "_" + group
				# Look at the dataset and the group
				if key not in data:
					data[key] = {alg: {"x":[x_val], "y": [y_val]}}
				else:
					# Look at the algorithm run on the dataset and group
					if alg not in data[key]:
						data[key][alg] = {"x": [x_val], "y": [y_val]}
					else:
						data[key][alg]["x"].append(x_val)
						data[key][alg]["y"].append(y_val)
			lc += 1
	return data

def plotgraphsfromfile(filepath, filetype, outputdir):
	data = load_data(filepath, filetype)
	for graphname in data:
		print("Plotting - ", graphname)
		data_per_alg = data[graphname]
		plot_graph(graphname, filetype, data_per_alg, outputdir)



# main
def main():
	outputdir = "./graphs"
	if not os.path.exists(outputdir):
		os.makedirs(outputdir)
	plotgraphsfromfile("./results/results_vary_k.csv","vary_k","./graphs")
	plotgraphsfromfile("./results/results_vary_k.csv","vary_k_div","./graphs")



if __name__ == "__main__":
	main()