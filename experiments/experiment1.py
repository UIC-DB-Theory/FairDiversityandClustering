"""
Description:
    Measure runtime and diversity for varying k. Where k is the size of the output.
"""
import sys
import json
import numpy as np
sys.path.append("..")


log = {
    "datasets" : {
        "Adult" : {
            "data_dir" : "../datasets/adult",
            "color_fields" : [
                "race", 
                "sex"
            ],
            "feature_fields" : [
                "age", 
                "capital-gain", 
                "capital-loss", 
                "hours-per-week", 
                "fnlwgt", 
                "education-num"],
            "normalize" : 1
        }
    }
}

# algorithms = {
#     "SFDM-2" : 
#     lambda fs, cs, kis, gamma

# }

# Read all the datasets
from datasets.utils import read_dataset
datasets = {}
for dataset in log["datasets"]:
    datasets[dataset] = read_dataset(
        log["datasets"][dataset]["data_dir"],
        log["datasets"][dataset]["feature_fields"],
        log["datasets"][dataset]["color_fields"],
        normalize=True if log["datasets"][dataset]["normalize"] == 1 else False
    )
    log["datasets"][dataset]["points_per_color"] = datasets[dataset]["points_per_color"]


json_object = json.dumps(log, indent=8)
print(json_object)

