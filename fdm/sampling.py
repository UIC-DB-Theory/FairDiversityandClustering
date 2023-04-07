import scipy.sparse as sp
import random

# sampling from adult.csv
output = open("adult_small.csv", "a")
with open("adult.csv", "r") as fileobj:
    lines = fileobj.readlines()
    random.Random(0).shuffle(lines)
    output.writelines(lines[:1000])
output.flush()
output.close()

# sampling for celebA
csr_sparse = sp.load_npz('celebA_csr_sparse.npz')
a = random.Random(0).sample(range(202599), 1000)
sp.save_npz('celebA_small_csr_sparse.npz', csr_sparse[a])
