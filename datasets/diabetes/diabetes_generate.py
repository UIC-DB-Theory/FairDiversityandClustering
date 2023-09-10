import json
import csv

metadata_file = "diabetes.metadata"
metadata = {
    "filename" : "diabetes.data",
    "url" : "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"
}
fields = []
data = []

# ---------------Read original dataset file--------------
with open("./diabetic_data.csv", 'r') as file:
    csvreader = csv.reader(file)
    i = 0
    for row in csvreader:
        if i == 0:
            fields = row
        else:
            data.append(row)
        i = i + 1

# ---------------Create file diabetes.metadata--------------
metadata["fields"] = fields

# Write to metadata file
json_object = json.dumps(metadata, indent=4)
with open(metadata_file, "w") as outfile:
    outfile.write(json_object)


# ---------------Create file diabetes.data---------------
with open(metadata["filename"], "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)