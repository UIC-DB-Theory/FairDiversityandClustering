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

Example for importing a dataset - Diabetes
```

# Import
from datasets.utils import read_dataset

# Select the dataset directory 
datasetdir = "./datasets/adult"

# Set the feilds for features and colors
color_fields = ["gender", "diabetesMed"]
feature_fields = ["num_procedures", "num_lab_procedures", "number_outpatient", "number_emergency", "number_inpatient" "number_diagnoses"]

# Read the dataset
dataset = read_dataset(datasetdir, feature_fields, color_fields)

#
features = dataset["features"]
colors = dataset["colors"]

```
