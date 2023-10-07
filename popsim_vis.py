# Import the popsim dataset
from datasets.utils import read_dataset


# Select the dataset directory 
datasetdir = "./datasets/popsim"

# Set the feilds for features and colors
color_fields = ['race']
feature_fields = ['x', 'y', 'z']

# Read the dataset
dataset = read_dataset(datasetdir, feature_fields, color_fields, unique=False, normalize=False)

#
features = dataset["features"]
colors = dataset["colors"]

race_color = {
    'White' : 'red',
    'Black/African American' : 'pink',
    'African American/Alaska Native' : 'yellow',
    'Asian' : 'blue',
    'Native Hawaiian/Other Pacific Islander' : 'green',
}

import matplotlib.pyplot as plt
from tqdm import trange
xs = []
ys = []
zs = []
cs = []
for i in trange(0, len(features)):
    color = race_color[colors[i]]
    x = features[i][0]
    y = features[i][1]
    z = features[i][2]
    xs.append(x)
    ys.append(y)
    zs.append(z)
    cs.append(color)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, zs, c = cs)
plt.savefig(f'popsim_vis', dpi=300, bbox_inches='tight')