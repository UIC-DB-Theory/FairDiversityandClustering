import json
import csv

metadata_file = "adult.metadata"
metadata = {
    "filename" : "adult.data",
    "url" : "https://archive.ics.uci.edu/dataset/2/adult"
}


# fields can be found in adult.names file
fields = [
                "age",
                "workclass",
                "fnlwgt",
                "education",
                "education-num",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
                "native-country",
                "yearly-income"
        ]
data = []

# ---------------Read original dataset file--------------
# ALREADY IN REQUIRED FORMAT
# with open("./adult.data", 'r') as file:
#     csvreader = csv.reader(file)
#     i = 0
#     for row in csvreader:
#         print(row)

# ---------------Create file adult.metadata--------------
metadata["fields"] = fields

# Write to metadata file
json_object = json.dumps(metadata, indent=4)
with open(metadata_file, "w") as outfile:
    outfile.write(json_object)


# ---------------Create file adult.data---------------
# with open(metadata["filename"], "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(data)