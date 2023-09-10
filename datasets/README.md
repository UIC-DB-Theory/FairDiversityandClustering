# Datasets

Run the `init_datasets.sh` script to download all datasets

Example for importing a dataset - Adult
```

# Import
from datasets.utils import read_dataset

# Select the dataset directory 
datasetdir = "./datasets/adult"

# Set the feilds for features and colors
color_fields = ['race', 'sex']
feature_fields = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt', 'education-num']

# Read the dataset
dataset = read_dataset(datasetdir, feature_fields, color_fields)

#
features = dataset["features"]
colors = dataset["colors"]

```
