import json
import csv
import math

metadata_file = "popsim.metadata"
metadata = {
    "filename" : "popsim.data",
    "url" : "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"
}
fields = []
data = []

# ---------------Read original dataset file--------------
# P1_003N, P1_004N, P1_005N, P1_006N, P1_007N
# white, black or African American, American Indian and Alaska Native, Asian, Native Hawaiian, Other Pacific Islander
race_mappings = {
    'P1_003N' : 'White',
    'P1_004N' : 'Black/African American',
    'P1_005N' : 'African American/Alaska Native',
    'P1_006N' : 'Asian',
    'P1_007N' : 'Native Hawaiian/Other Pacific Islander',
}
with open("./popsim_5m.csv", 'r') as file:
    csvreader = csv.reader(file)
    i = -1
    for row in csvreader:
        i = i + 1
        if i == 0:
            fields = ['race', 'x', 'y', 'z']
            continue
        # only append rows with race mappings
        if row[2] in race_mappings:
            # Convert lat lon to 2-D points
            lat = math.radians(float(row[4]))
            lon = math.radians(float(row[3]))
            x = 6371*math.cos(lat)*math.cos(lon)
            y = 6371*math.cos(lat)*math.sin(lon)
            z = 6371*math.sin(lat)
            data.append([race_mappings[row[2]], x, y, z])


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