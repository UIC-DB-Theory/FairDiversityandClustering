import json
import csv

metadata_file = "census.metadata"
metadata = {
    "filename" : "census.data",
    "url" : "https://archive.ics.uci.edu/dataset/2/adult"
}


# fields can be found in adult.names file
fields = [
                "id",
                "sex-color",
                "age-color",
                "age-sex-color",
                "attr1",
                "attr2",
                "attr3",
                "attr4",
                "attr5",
                "attr6",
                "attr7",
                "attr8",
                "attr9",
                "attr10",
                "attr11",
                "attr12",
                "attr13",
                "attr14",
                "attr15",
                "attr16",
                "attr17",
                "attr18",
                "attr19",
                "attr20",
                "attr21",
                "attr22",
                "attr23",
                "attr24",
                "attr25"
        ]
data = []

# ---------------Read original dataset file--------------
with open('census.csv', 'r') as file:
    csvreader = csv.reader(file)
    i = 0
    for row in csvreader:
        data.append(row)

# ---------------Create file adult.metadata--------------
metadata["fields"] = fields

# Write to metadata file
json_object = json.dumps(metadata, indent=4)
with open(metadata_file, "w") as outfile:
    outfile.write(json_object)


# ---------------Create file census.data---------------
with open(metadata["filename"], "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)