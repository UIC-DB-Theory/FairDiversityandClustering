import numpy as np
from sklearn.datasets import make_blobs
import os


def generate_blobs(size: int, group: int, dim: int) -> None:
	X, y = make_blobs(n_samples=size, centers=10, n_features=dim, random_state=17)
	g = np.random.randint(group, size=size, dtype=np.uint8)
	if not os.path.exists("./data/"):
		os.makedirs("./data/")
	with open("./data/blobs_n" + str(size) + "_c" + str(group) + ".csv", "a") as fileobj:
		for i in range(size):
			fileobj.write(str(i) + ",")
			fileobj.write(str(g[i]) + ",")
			for j in range(dim - 1):
				fileobj.write(format(X[i][j], ".6f") + ",")
			fileobj.write(format(X[i][dim - 1], ".6f") + "\n")


for c in range(2,21,2):
	generate_blobs(1000, c, 2)

generate_blobs(10000000, 2, 2)
generate_blobs(10000000, 10, 2)
